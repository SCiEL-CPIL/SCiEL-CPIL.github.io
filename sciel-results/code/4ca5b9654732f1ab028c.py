import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

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
        def utility_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -1.0 / (c * c)
        def inv_utility_prime(x):
            # u'(c) = 1/c => c = 1/x
            return 1.0 / np.maximum(x, 1e-12)
    else:
        def utility(c):
            c = np.maximum(c, 1e-12)
            return c**(1 - gamma) / (1 - gamma)
        def utility_prime(c):
            c = np.maximum(c, 1e-12)
            return c**(-gamma)
        def utility_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -gamma * c**(-gamma - 1)
        def inv_utility_prime(x):
            # u'(c) = c^(-gamma) => c = x^(-1/gamma)
            return np.maximum(x, 1e-12)**(-1.0/gamma)

    # Create a fine grid for m values
    m_fine = np.geomspace(0.001, 40, 801)
    n_fine = len(m_fine)
    
    # V[t][z] will be a 1D array of length n_fine
    V = np.zeros((horizon, Z, n_fine))
    
    # Terminal: V[horizon-1][z][i] = u(m_fine[i])
    for z in range(Z):
        V[horizon-1, z, :] = utility(m_fine)
    
    # Precompute interpolation helpers
    # For each z, we'll store V[t+1, z, :] for interpolation
    
    # Work backwards from t = horizon-2 down to 0
    # Use Euler equation with Newton's method
    # F(c) = u'(c) - beta * R * sum_{z'} P[z,z'] * u'(c'(m', z')) = 0
    # where m' = R*(m-c) + income[z']
    # and c'(m', z') is the optimal consumption at (m', z') in period t+1
    
    # For the continuation, we need the optimal consumption policy at t+1
    # We'll store c_policy[t][z][i] for the fine grid
    
    c_policy = np.zeros((horizon, Z, n_fine))
    
    # Terminal: consume all
    for z in range(Z):
        c_policy[horizon-1, z, :] = m_fine.copy()
    
    # For t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        for z in range(Z):
            c_t_z = np.zeros(n_fine)
            V_t_z = np.zeros(n_fine)
            
            for i in range(n_fine):
                m_val = m_fine[i]
                
                # Initial guess: proportional consumption
                # For CRRA, consumption is roughly proportional to wealth
                c = max(0.5 * m_val, 1e-8)
                c = min(c, m_val * 0.999)
                
                # Newton's method on the Euler equation
                # F(c) = u'(c) - beta * R * sum_{z'} P[z,z'] * u'(c_next(m', z'))
                # where m' = R*(m-c) + income[z']
                # c_next(m', z') is interpolated from c_policy[t+1, z', :]
                
                for newton_iter in range(15):
                    # Compute F(c)
                    F = 0.0
                    for zp in range(Z):
                        m_prime = R * (m_val - c) + income[zp]
                        m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                        c_next = np.interp(m_prime, m_fine, c_policy[t+1, zp, :])
                        c_next = max(c_next, 1e-12)
                        F += transition[z, zp] * utility_prime(c_next)
                    F = utility_prime(c) - beta * R * F
                    
                    # Compute F'(c) for Newton step
                    # F(c) = u'(c) - beta*R*sum_{z'} P[z,z']*u'(c'(m',z'))
                    # F'(c) = u''(c) - beta*R*sum_{z'} P[z,z']*u''(c'(m',z')) * dc'/dm' * dm'/dc
                    # dm'/dc = -R
                    # dc'/dm' = 1 / (1 - beta*R*sum_{z''} P[z',z'']*u''(c'')*R*...) -- complex
                    # Approximate F'(c) using u''(c) as dominant term
                    # Better: use a simple bisection-like approach or secant
                    
                    # Use secant method: keep previous c
                    if newton_iter == 0:
                        c_prev = c * 0.95
                        F_prev = 0.0
                        for zp in range(Z):
                            m_prime_prev = R * (m_val - c_prev) + income[zp]
                            m_prime_prev = np.clip(m_prime_prev, m_fine[0], m_fine[-1])
                            c_next_prev = np.interp(m_prime_prev, m_fine, c_policy[t+1, zp, :])
                            c_next_prev = max(c_next_prev, 1e-12)
                            F_prev += transition[z, zp] * utility_prime(c_next_prev)
                        F_prev = utility_prime(c_prev) - beta * R * F_prev
                    else:
                        c_prev = c_old
                        F_prev = F_old
                    
                    # Secant step
                    if abs(F - F_prev) > 1e-15:
                        c_new = c - F * (c - c_prev) / (F - F_prev)
                    else:
                        c_new = c
                    
                    # Clamp to valid range
                    c_new = np.clip(c_new, 1e-12, m_val * (1 - 1e-10))
                    
                    c_old = c
                    F_old = F
                    c = c_new
                    
                    if abs(F) < 1e-10:
                        break
                
                c_t_z[i] = c
                
                # Compute V_t_z[i]
                cont_val = 0.0
                for zp in range(Z):
                    m_prime = R * (m_val - c) + income[zp]
                    m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                    cont_val += transition[z, zp] * np.interp(m_prime, m_fine, V[t+1, zp, :])
                V_t_z[i] = utility(c) + beta * cont_val
            
            c_policy[t, z, :] = c_t_z
            V[t, z, :] = V_t_z
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    # Extract consumption policy on the original resource_grid
    consumption = np.zeros((horizon, Z, M))
    
    # Terminal period
    for z in range(Z):
        consumption[horizon-1, z, :] = m_grid.copy()
    
    # For t = 0 to horizon-2, interpolate from the fine grid policy
    for t in range(horizon - 1):
        for z in range(Z):
            consumption[t, z, :] = np.interp(m_grid, m_fine, c_policy[t, z, :])
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid + 1e-10)
    
    # Ensure finite values
    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=m_grid, neginf=1e-12)
    
    return consumption
