import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']
    
    M = np.array(resource_grid)
    num_m = len(M)
    num_z = len(income)
    H = horizon
    
    consumption = np.zeros((H, num_z, num_m))
    
    # CRRA utility and derivative
    if abs(gamma - 1.0) < 1e-10:
        def u(c):
            return np.log(np.maximum(c, 1e-12))
        def u_prime(c):
            return 1.0 / np.maximum(c, 1e-12)
    else:
        def u(c):
            return np.maximum(c, 1e-12) ** (1.0 - gamma) / (1.0 - gamma)
        def u_prime(c):
            return np.maximum(c, 1e-12) ** (-gamma)
    
    # Last period: c = m, V = u(m)
    consumption[H-1, :, :] = M[np.newaxis, :, np.newaxis] * np.ones((num_z, 1))
    V_next = u(M)  # shape (num_m,)
    
    # Backward induction
    for t in range(H - 2, -1, -1):
        if time.time() > deadline - 5.0:
            break
        
        V_curr = np.zeros((num_z, num_m))
        
        for z in range(num_z):
            probs = transition[z, :]  # shape (num_z,)
            
            # Vectorized bisection on FOC
            c_low = np.full(num_m, 1e-12)
            c_high = M.copy()
            
            for _ in range(45):
                c_mid = 0.5 * (c_low + c_high)
                cp = u_prime(c_mid)
                m_next = R * (M[np.newaxis, :] - c_mid[np.newaxis, :]) + income[:, np.newaxis]
                m_next = np.maximum(m_next, 1e-12)
                
                h = 1e-5 * np.maximum(m_next, 1.0)
                V_plus = np.array([np.interp(m_next[zp, :] + h[zp, :], M, V_next) for zp in range(num_z)])
                V_minus = np.array([np.interp(m_next[zp, :] - h[zp, :], M, V_next) for zp in range(num_z)])
                dVdm = (V_plus - V_minus) / (2.0 * h)
                
                expected_dVdm = np.dot(probs, dVdm)
                fp = cp - beta * R * expected_dVdm
                
                c_low = np.where(fp > 0, c_mid, c_low)
                c_high = np.where(fp > 0, c_high, c_mid)
            
            c_opt = 0.5 * (c_low + c_high)
            consumption[t, z, :] = c_opt
            
            m_next = R * (M[np.newaxis, :] - c_opt[np.newaxis, :]) + income[:, np.newaxis]
            m_next = np.maximum(m_next, 1e-12)
            V_next_vals = np.array([np.interp(m_next[zp, :], M, V_next) for zp in range(num_z)])
            V_curr[z, :] = u(c_opt) + beta * np.dot(probs, V_next_vals)
        
        V_next = V_curr
    
    # Ensure validity
    consumption[H-1, :, :] = M[np.newaxis, :, np.newaxis] * np.ones((num_z, 1))
    
    for t in range(H):
        for z in range(num_z):
            c = consumption[t, z, :]
            c = np.maximum(c, 1e-12)
            c = np.minimum(c, M + 1e-10)
            consumption[t, z, :] = c
    
    consumption[H-1, :, :] = M[np.newaxis, :, np.newaxis] * np.ones((num_z, 1))
    
    return consumption
