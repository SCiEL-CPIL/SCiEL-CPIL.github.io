import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill

    # Extract parameters
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    # Output grid
    m_out = resource_grid
    n_out = len(m_out)

    # Use a dense internal grid for accurate interpolation
    # 2000 points for m, 400 points for s (save fraction)
    m_int = np.geomspace(0.001, 40, 2000)
    s_int = np.linspace(0.001, 1.0, 400)

    n_m = len(m_int)
    n_s = len(s_int)

    # Initialize policy and value arrays
    policy_int = np.zeros((horizon, 2, n_m))
    value_int = np.zeros((horizon, 2, n_m))

    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)

    # Backward induction
    # Terminal period t = horizon - 1
    t = horizon - 1
    for z in range(2):
        policy_int[t, z, :] = m_int
        value_int[t, z, :] = utility(m_int)

    # Previous periods
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline - 5:
            break
        
        for z in range(2):
            # Create meshgrid of m and s
            M_grid, S_grid = np.meshgrid(m_int, s_int, indexing='ij')
            C_grid = M_grid * S_grid
            
            # Pre-compute expected continuation value
            cont_val = np.zeros_like(C_grid)
            
            for zp in range(2):
                prob = transition[z, zp]
                if prob == 0:
                    continue
                y_zp = income[zp]
                m_prime = R * (M_grid - C_grid) + y_zp
                # Clip m_prime to valid range for interpolation
                m_prime = np.clip(m_prime, m_int[0], m_int[-1])
                v_next = np.interp(m_prime, m_int, value_int[t+1, zp, :])
                cont_val += prob * v_next
            
            # Total utility
            u_c = utility(C_grid)
            total_util = u_c + beta * cont_val
            
            # Find the best c for each m
            best_s_idx = np.argmax(total_util, axis=1)
            best_c = C_grid[np.arange(n_m), best_s_idx]
            
            # Store policy and value
            policy_int[t, z, :] = best_c
            value_int[t, z, :] = total_util[np.arange(n_m), best_s_idx]

    # Interpolate the internal policies to the output grid
    policy_out = np.zeros((horizon, 2, n_out))
    
    for t in range(horizon):
        for z in range(2):
            policy_out[t, z, :] = np.interp(m_out, m_int, policy_int[t, z, :])
            
            # Enforce constraints: 0 < c <= m + 1e-10
            policy_out[t, z, :] = np.maximum(policy_out[t, z, :], 1e-12)
            policy_out[t, z, :] = np.minimum(policy_out[t, z, :], m_out + 1e-10)
            
            # Special case for t = horizon - 1: c should be exactly m
            if t == horizon - 1:
                policy_out[t, z, :] = m_out

    return policy_out
