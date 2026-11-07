import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

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

    # Use a fine grid for m values
    # Increased to 2001 to improve interpolation accuracy, enabled by faster Newton steps
    m_fine = np.geomspace(0.001, 40, 2001)
    n_fine = len(m_fine)
    
    # V[t][z] will be a 1D array of length n_fine
    V = np.zeros((horizon, Z, n_fine))
    # D[t][z] stores the derivative of consumption w.r.t m, dc/dm
    D = np.zeros((horizon, Z, n_fine))
    
    # Terminal: V[horizon-1][z][i] = u(m_fine[i])
    for z in range(Z):
        V[horizon-1, z, :] = utility(m_fine)
        D[horizon-1, z, :] = 1.0  # c = m, so dc/dm = 1
    
    # c_policy[t][z][i] stores the optimal consumption at m_fine[i] for state z at time t
    c_policy = np.zeros((horizon, Z, n_fine))
    
    # Terminal: consume all
    for z in range(Z):
        c_policy[horizon-1, z, :] = m_fine.copy()
    
    # Work backwards from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        for z in range(Z):
            c_t_z = np.zeros(n_fine)
            V_t_z = np.zeros(n_fine)
            D_t_z = np.zeros(n_fine)
            
            for i in range(n_fine):
                m_val = m_fine[i]
                
                # Initial guess: proportional consumption
                c = max(0.5 * m_val, 1e-8)
                c = min(c, m_val * 0.999)
                
                # Newton's method on the Euler equation
                # F(c) = u'(c) - beta * R * sum_{z'} P[z,z'] * u'(c'(m', z'))
                # F'(c) = u''(c) + beta * R^2 * sum_{z'} P[z,z'] * u''(c'(m', z')) * D'(m', z')
                
                for newton_iter in range(15):
                    F = 0.0
                    F_prime = 0.0
                    u_prime_c = utility_prime(c)
                    u_double_prime_c = utility_double_prime(c)
                    F = u_prime_c
                    F_prime = u_double_prime_c
                    
                    for zp in range(Z):
                        p = transition[z, zp]
                        if p == 0: continue
                        
                        m_prime = R * (m_val - c) + income[zp]
                        m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                        
                        c_next = np.interp(m_prime, m_fine, c_policy[t+1, zp, :])
                        c_next = max(c_next, 1e-12)
                        
                        # Derivative of c_next w.r.t m_prime
                        D_next = np.interp(m_prime, m_fine, D[t+1, zp, :])
                        
                        u_prime_next = utility_prime(c_next)
                        u_double_prime_next = utility_double_prime(c_next)
                        
                        F -= beta * R * p * u_prime_next
                        # dm'/dc = -R
                        # d/dc [ u'(c_next(m')) ] = u''(c_next) * D_next * (-R)
                        # Term in F is - beta * R * ...
                        # So derivative is - beta * R * ... * (-R) * D_next * u''(c_next)
                        # = beta * R^2 * ... * D_next * u''(c_next)
                        F_prime += beta * R * R * p * u_double_prime_next * D_next
                    
                    if abs(F_prime) < 1e-15:
                        break
                    
                    c_new = c - F / F_prime
                    c_new = np.clip(c_new, 1e-12, m_val * (1 - 1e-10))
                    
                    if abs(c_new - c) < 1e-12:
                        c = c_new
                        break
                    c = c_new
                
                c_t_z[i] = c
                
                # Compute V_t_z[i]
                cont_val = 0.0
                for zp in range(Z):
                    m_prime = R * (m_val - c) + income[zp]
                    m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                    cont_val += transition[z, zp] * np.interp(m_prime, m_fine, V[t+1, zp, :])
                V_t_z[i] = utility(c) + beta * cont_val
                
                # Compute D_t_z[i] using Envelope Theorem / Implicit Function Theorem
                # V_t(m) = u(c) + beta E[V_{t+1}(m')]
                # dV/dm = u'(c) * dc/dm + beta E[ V'_{t+1}(m') * dm'/dm ]
                # Also from Euler eq: u'(c) = beta R E[u'(c')]
                # So dV/dm = beta R E[u'(c')] * dc/dm + beta E[ V'_{t+1}(m') * R ]
                # Note V'_{t+1}(m') = u'(c_{t+1}(m'))
                # So dV/dm = beta R E[ u'(c') * (dc/dm + 1) ] ? No.
                
                # Let's use the standard formula for D = dc/dm
                # F(c, m) = u'(c) - beta R E[u'(c'(m'))] = 0
                # df/dm = u'(c) * 0 - beta R E[ u''(c') D' * dm'/dm ] = - beta R E[ u''(c') D' (-R) ] = beta R^2 E[ u''(c') D' ]
                # df/dc = u''(c) + beta R^2 E[ u''(c') D' ]  (Wait, this is F_prime from above)
                # Actually, F(c,m) = u'(c) - beta R Sum P u'(c')
                # dF/dm = - beta R Sum P [ u''(c') D' * (-R) ] = beta R^2 Sum P u''(c') D'
                # dF/dc = u''(c) + beta R^2 Sum P u''(c') D'  (This matches F_prime calculated above? 
                # Let's re-verify F_prime.
                # F = u'(c) - beta R Sum P u'(c')
                # dF/dc = u''(c) - beta R Sum P u''(c') D' (-R) = u''(c) + beta R^2 Sum P u''(c') D'
                # Yes, it matches.
                
                # By Implicit Function Theorem: dc/dm = - (dF/dm) / (dF/dc)
                # dF/dm = beta R^2 Sum P u''(c') D'
                # dF/dc = F_prime (calculated above)
                
                # We need to calculate dF/dm separately or reuse terms
                # dF/dm = beta * R * R * sum(transition[z, zp] * u_double_prime_next * D_next)
                
                # Let's recompute the sum part for dF/dm
                sum_term = 0.0
                for zp in range(Z):
                    p = transition[z, zp]
                    if p == 0: continue
                    m_prime = R * (m_val - c) + income[zp]
                    m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                    c_next = np.interp(m_prime, m_fine, c_policy[t+1, zp, :])
                    c_next = max(c_next, 1e-12)
                    D_next = np.interp(m_prime, m_fine, D[t+1, zp, :])
                    u_double_prime_next = utility_double_prime(c_next)
                    sum_term += p * u_double_prime_next * D_next
                
                dF_dm = beta * R * R * sum_term
                dF_dc = F_prime # From the last Newton iteration
                
                if abs(dF_dc) > 1e-15:
                    D_t_z[i] = -dF_dm / dF_dc
                else:
                    D_t_z[i] = 1.0 # Fallback
                
                # Clamp D to reasonable range
                D_t_z[i] = np.clip(D_t_z[i], 0.0, 5.0)
            
            c_policy[t, z, :] = c_t_z
            V[t, z, :] = V_t_z
            D[t, z, :] = D_t_z
        
        # Check time
        if time.time() > deadline - 5:
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
