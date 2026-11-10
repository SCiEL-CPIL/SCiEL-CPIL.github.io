"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 150.0 - 10.0  # safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    N = len(resource_grid)
    T = horizon
    Z = len(income)
    
    # Policy array: c[t, z, i]
    c_policy = np.zeros((T, Z, N))
    
    # Last period: consume everything
    c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    def util_prime(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)
    
    # Backward induction with Newton-Raphson on the Euler equation
    # For each t from T-2 down to 0:
    #   c* satisfies: u'(c) = beta * E_z'[ R * u'(c') ]
    # where c' = R*(m - c) + income[z']
    
    for t in range(T - 2, -1, -1):
        if time.time() > deadline:
            break
        
        for z in range(Z):
            # For each grid point m
            for i in range(N):
                m = resource_grid[i]
                
                # Try Newton-Raphson to solve Euler equation
                # f(c) = u'(c) - beta * sum_{z'} P(z'|z) * u'(R*(m-c) + income[z'])
                # f'(c) = -u''(c) + beta * sum_{z'} P(z'|z) * R * u''(R*(m-c) + income[z'])
                
                # Initial guess: use a fraction of m
                # For CRRA, c* ≈ m * (1 - (beta*R)^(1/gamma)) for log utility
                if np.isclose(gamma, 1.0):
                    # Log utility: c* = m / (1 + 1/(beta*R)) approximately for infinite horizon
                    # For finite horizon, start with a reasonable guess
                    c_guess = m / (1.0 + 1.0 / (beta * R))
                else:
                    c_guess = m * (1.0 - (beta * R) ** (1.0 / gamma))
                    if c_guess <= 0:
                        c_guess = m * 0.5
                
                c_guess = np.clip(c_guess, 1e-10, m * 0.99)
                
                # Newton iterations
                c_new = c_guess
                for _ in range(20):
                    c_curr = c_new
                    
                    # Compute f(c) and f'(c)
                    u_prime_c = util_prime(c_curr)
                    
                    # Expected continuation utility derivative
                    f_val = u_prime_c
                    f_prime_val = -u_prime_c / c_curr  # -u''(c) = -u'(c)/c for log, adjust for CRRA
                    
                    if np.isclose(gamma, 1.0):
                        # u''(c) = -1/c^2, so -u''(c) = 1/c^2
                        f_prime_val = 1.0 / (c_curr * c_curr)
                    else:
                        # u''(c) = -gamma * c^(-gamma-1), so -u''(c) = gamma * c^(-gamma-1)
                        f_prime_val = gamma * c_curr ** (-gamma - 1)
                    
                    # Continuation part
                    for zp in range(Z):
                        p_zp = transition[z, zp]
                        if p_zp == 0:
                            continue
                        
                        m_next = R * (m - c_curr) + income[zp]
                        if m_next <= 0:
                            # Can't have negative resources; this c is infeasible
                            # Push c down
                            c_new = c_curr * 0.5
                            break
                        
                        u_prime_m_next = util_prime(m_next)
                        f_val -= beta * p_zp * u_prime_m_next
                        
                        # f'(c) contribution: beta * p_zp * R * u''(m_next)
                        # u''(m_next) is negative, so this adds to f'
                        if np.isclose(gamma, 1.0):
                            u_double_prime = -1.0 / (m_next * m_next)
                        else:
                            u_double_prime = -gamma * m_next ** (-gamma - 1)
                        
                        f_prime_val += beta * p_zp * R * u_double_prime
                    
                    if c_new * 0.5 > 1e-10 and c_new != c_curr:
                        # Check if we broke out due to infeasibility
                        if m_next <= 0 if 'm_next' in dir() else True:
                            pass
                    
                    if abs(f_prime_val) < 1e-30:
                        break
                    
                    c_new = c_curr - f_val / f_prime_val
                    
                    # Ensure feasibility
                    c_new = np.clip(c_new, 1e-10, m * 0.999)
                    
                    if abs(c_new - c_curr) < 1e-12 * max(1.0, abs(c_curr)):
                        break
                
                c_policy[t, z, i] = c_new
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Ensure last period is correct
    c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    # Validate: 0 < c <= m + 1e-10
    for t in range(T):
        for z in range(Z):
            for i in range(N):
                m = resource_grid[i]
                if c_policy[t, z, i] <= 0:
                    c_policy[t, z, i] = 1e-10
                if c_policy[t, z, i] > m + 1e-10:
                    c_policy[t, z, i] = m
    
    return c_policy
# EVOLVE-BLOCK-END
