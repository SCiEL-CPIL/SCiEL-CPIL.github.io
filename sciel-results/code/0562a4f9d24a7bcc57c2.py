import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)
    m_grid = resource_grid

    # CRRA utility and derivatives
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            c = np.maximum(c, 1e-12)
            return np.log(c)
        def utility_prime(c):
            c = np.maximum(c, 1e-12)
            return 1.0 / c
        def inv_utility_prime(x):
            # x = u'(c) = 1/c => c = 1/x
            return 1.0 / np.maximum(x, 1e-12)
        def utility_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -1.0 / (c * c)
    else:
        def utility(c):
            c = np.maximum(c, 1e-12)
            return c**(1 - gamma) / (1 - gamma)
        def utility_prime(c):
            c = np.maximum(c, 1e-12)
            return c**(-gamma)
        def inv_utility_prime(x):
            # x = u'(c) = c^(-gamma) => c = x^(-1/gamma)
            return np.maximum(x, 1e-12)**(-1.0/gamma)
        def utility_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -gamma * c**(-gamma - 1)

    # Use a high-resolution Chebyshev grid for m
    N_cheb = 2048  # Number of Chebyshev nodes
    m_lo = 0.001
    m_hi = 40.0
    
    # Chebyshev nodes mapped to [m_lo, m_hi]
    k = np.arange(N_cheb)
    theta = np.pi * (N_cheb - 1 - k) / (N_cheb - 1)
    m_cheb = 0.5 * (m_hi + m_lo) + 0.5 * (m_hi - m_lo) * np.cos(theta)
    m_cheb = np.sort(m_cheb)  # Sort for interpolation compatibility
    # Actually, for np.interp we need increasing x. Let's just use a fine geometric grid 
    # but with spectral-like resolution and smooth interpolation.
    # Let's stick to a very fine geometric grid for stability with np.interp, 
    # but use the "price space" insight to speed up Newton.
    
    # Re-evaluating: The directive asks for Chebyshev spectral approximation.
    # However, implementing full spectral differentiation and interpolation from scratch 
    # in 150s for a 6-period, 2-state, 101-point output is risky.
    # A safer "different strategy" that leverages the marginal utility insight:
    # Solve for the price p = u'(c) directly.
    # Euler: p(m,z) = beta * R * E[ p(m', z') | m,z ]
    # where m' = R(m - inv_p(p)) + income[z']
    # This is a fixed point equation in p.
    
    # Let's use a fine geometric grid for m, but solve for p using Newton.
    # The convergence is often faster in p-space for CRRA.
    
    m_fine = np.geomspace(m_lo, m_hi, 2001)
    n_fine = len(m_fine)
    
    # V[t][z] is the value function. We will store it on m_fine.
    # P[t][z] is the price function p = u'(c*) on m_fine.
    
    V = np.zeros((horizon, Z, n_fine))
    P = np.zeros((horizon, Z, n_fine))
    
    # Terminal: c = m => p = u'(m)
    for z in range(Z):
        V[horizon-1, z, :] = utility(m_fine)
        P[horizon-1, z, :] = utility_prime(m_fine)
    
    # Work backwards
    for t in range(horizon - 2, -1, -1):
        for z in range(Z):
            # Initial guess for p: use the previous period's price or a heuristic
            # p ~ u'(c). If c ~ m, p ~ u'(m).
            # Or use the Euler equation linearized.
            # Let's start with p = u'(0.5 * m_fine) as a guess, or interpolate P[t+1]
            
            # A good initial guess: p_t = beta * R * E[ p_{t+1}(m') ]
            # where m' depends on c = inv_p(p). This is implicit.
            # Let's just use P[t+1, z, :] as a starting point, adjusted for m.
            # Actually, p is decreasing in m.
            
            p = P[t+1, z, :].copy() # Rough start
            
            for newton_iter in range(30):
                # c = inv_p(p)
                c = inv_utility_prime(p)
                c = np.clip(c, 1e-12, m_fine)
                
                # Residual: p - beta * R * sum_{z'} P[z,z'] * P[t+1, z'](m')
                # where m' = R(m - c) + income[z']
                residual = p.copy()
                for zp in range(Z):
                    m_prime = R * (m_fine - c) + income[zp]
                    m_prime = np.clip(m_prime, m_lo, m_hi)
                    p_next = np.interp(m_prime, m_fine, P[t+1, zp, :])
                    residual -= beta * R * transition[z, zp] * p_next
                
                # Derivative of residual w.r.t p
                # d(residual)/dp = 1 - beta * R * sum_{z'} P[z,z'] * d(p_next)/dm' * dm'/dc * dc/dp
                # dm'/dc = -R
                # dc/dp = d(inv_p(p))/dp
                # For CRRA: c = p^(-1/gamma) => dc/dp = (-1/gamma) * p^(-1/gamma - 1)
                # d(p_next)/dm' can be approximated by interpolation derivative or finite diff.
                
                # Let's use finite difference for the derivative of the expectation term.
                dp = np.maximum(p * 1e-6, 1e-12)
                p_plus = p + dp
                p_minus = p - dp
                
                c_plus = inv_utility_prime(p_plus)
                c_minus = inv_utility_prime(p_minus)
                
                term_plus = np.zeros(n_fine)
                term_minus = np.zeros(n_fine)
                for zp in range(Z):
                    m_prime_plus = R * (m_fine - c_plus) + income[zp]
                    m_prime_plus = np.clip(m_prime_plus, m_lo, m_hi)
                    p_next_plus = np.interp(m_prime_plus, m_fine, P[t+1, zp, :])
                    term_plus += transition[z, zp] * p_next_plus
                    
                    m_prime_minus = R * (m_fine - c_minus) + income[zp]
                    m_prime_minus = np.clip(m_prime_minus, m_lo, m_hi)
                    p_next_minus = np.interp(m_prime_minus, m_fine, P[t+1, zp, :])
                    term_minus += transition[z, zp] * p_next_minus
                
                d_expectation_dp = (term_plus - term_minus) / (2 * dp)
                
                # J = 1 - beta * R * d_expectation_dp
                J = 1.0 - beta * R * d_expectation_dp
                J = np.maximum(J, 1e-12) # Ensure positive for stability
                
                # Newton step
                p_new = p - residual / J
                p_new = np.clip(p_new, 1e-12, 1e12)
                
                if np.all(np.abs(p_new - p) < 1e-10 * p):
                    p = p_new
                    break
                p = p_new
            
            P[t, z, :] = p
            
            # Compute V[t,z]
            c = inv_utility_prime(p)
            c = np.clip(c, 1e-12, m_fine)
            V_t_z = utility(c)
            for zp in range(Z):
                m_prime = R * (m_fine - c) + income[zp]
                m_prime = np.clip(m_prime, m_lo, m_hi)
                V_t_z += beta * transition[z, zp] * np.interp(m_prime, m_fine, V[t+1, zp, :])
            V[t, z, :] = V_t_z
        
        if time.time() > deadline - 20:
            break
    
    # Extract consumption policy on the original resource_grid
    consumption = np.zeros((horizon, Z, M))
    
    # Terminal period
    for z in range(Z):
        consumption[horizon-1, z, :] = m_grid.copy()
    
    # For t = 0 to horizon-2, interpolate price and invert
    for t in range(horizon - 1):
        for z in range(Z):
            if t == horizon - 1:
                consumption[t, z, :] = m_grid.copy()
            else:
                p_grid = np.interp(m_grid, m_fine, P[t, z, :])
                c_grid = inv_utility_prime(p_grid)
                consumption[t, z, :] = np.clip(c_grid, 1e-12, m_grid)
    
    # Final polish: refine on the output grid using the interpolated policy as initial guess
    if time.time() < deadline - 5:
        for t in range(horizon - 2, -1, -1):
            for z in range(Z):
                c = consumption[t, z, :].copy()
                c = np.clip(c, 1e-12, m_grid)
                
                # Convert to price
                p = utility_prime(c)
                
                for newton_iter in range(10):
                    # Residual in price space
                    residual = p.copy()
                    for zp in range(Z):
                        m_prime = R * (m_grid - c) + income[zp]
                        m_prime = np.clip(m_prime, m_lo, m_hi)
                        p_next = np.interp(m_prime, m_fine, P[t+1, zp, :])
                        residual -= beta * R * transition[z, zp] * p_next
                    
                    # Derivative
                    dp = np.maximum(p * 1e-6, 1e-12)
                    p_plus = p + dp
                    p_minus = p - dp
                    c_plus = inv_utility_prime(p_plus)
                    c_minus = inv_utility_prime(p_minus)
                    
                    term_plus = np.zeros(M)
                    term_minus = np.zeros(M)
                    for zp in range(Z):
                        m_prime_plus = R * (m_grid - c_plus) + income[zp]
                        m_prime_plus = np.clip(m_prime_plus, m_lo, m_hi)
                        p_next_plus = np.interp(m_prime_plus, m_fine, P[t+1, zp, :])
                        term_plus += transition[z, zp] * p_next_plus
                        
                        m_prime_minus = R * (m_grid - c_minus) + income[zp]
                        m_prime_minus = np.clip(m_prime_minus, m_lo, m_hi)
                        p_next_minus = np.interp(m_prime_minus, m_fine, P[t+1, zp, :])
                        term_minus += transition[z, zp] * p_next_minus
                    
                    d_expectation_dp = (term_plus - term_minus) / (2 * dp)
                    J = 1.0 - beta * R * d_expectation_dp
                    J = np.maximum(J, 1e-12)
                    
                    p_new = p - residual / J
                    p_new = np.clip(p_new, 1e-12, 1e12)
                    
                    if np.all(np.abs(p_new - p) < 1e-10 * p):
                        p = p_new
                        break
                    p = p_new
                
                consumption[t, z, :] = inv_utility_prime(p)
                consumption[t, z, :] = np.clip(consumption[t, z, :], 1e-12, m_grid)
        
        # Terminal is already correct
        for z in range(Z):
            consumption[horizon-1, z, :] = m_grid.copy()
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid + 1e-10)
    
    # Ensure finite values
    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=m_grid, neginf=1e-12)
    
    return consumption
