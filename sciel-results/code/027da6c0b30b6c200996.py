import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Target grid
    m_grid = resource_grid
    N_m = len(m_grid)
    
    # Fine grid for value function interpolation
    N_fine = 1000
    m_fine = np.geomspace(0.001, 40, N_fine)
    
    # Utility function
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    
    # Store value functions on fine grid
    # V[t][z] will be an array of size N_fine
    V = [None] * horizon
    V[horizon - 1] = [None] * 2
    for z in range(2):
        V[horizon - 1][z] = u(m_fine)
        
    # Store policies on fine grid (optional, for re-solving on target grid)
    # Actually, we only need V to re-solve on target grid.
    
    # Backward induction on fine grid to get accurate V functions
    for t in range(horizon - 2, -1, -1):
        V[t] = [None] * 2
        for z in range(2):
            # Candidates for c
            # c in (0, m)
            eps = 1e-6
            N_c = 500
            c_fracs = np.linspace(eps, 1.0 - eps, N_c)
            
            # m_fine shape (N_fine,), c_fracs shape (N_c,)
            # C_cand shape (N_fine, N_c)
            C_cand = m_fine[:, None] * c_fracs[None, :]
            
            # Utility of consumption
            J = u(C_cand)
            
            # Continuation value
            for zp in range(2):
                p = transition[z, zp]
                if p < 1e-12:
                    continue
                # m' = R(m - c) + income[zp]
                m_next = R * (m_fine[:, None] - C_cand) + income[zp]
                # Interpolate V[t+1][zp]
                # Ensure m_next is within bounds for interp, though it should be
                # m_next min: R(0) + 0.5 = 0.5 (if c=m)
                # m_next max: R(40) + 1.5 = 42.3
                # m_fine is 0.001 to 40.
                # If m_next > 40, interp will clamp to V(40).
                # If m_next < 0.001, interp will clamp to V(0.001).
                V_next_vals = np.interp(m_next, m_fine, V[t+1][zp])
                J += beta * p * V_next_vals
                
            # Find max
            idx = np.argmax(J, axis=1)
            V[t][z] = J[np.arange(N_fine), idx]
            
        # Time check
        if time.time() > deadline:
            break

    # Now, compute the policy on the requested resource_grid
    # We need to re-solve for each m in resource_grid using the V functions from fine grid
    policy = np.zeros((horizon, 2, N_m))
    
    for t in range(horizon):
        if t == horizon - 1:
            # c = m
            for z in range(2):
                policy[t, z, :] = m_grid
        else:
            for z in range(2):
                # Re-solve for each m in m_grid
                eps = 1e-6
                N_c = 500
                c_fracs = np.linspace(eps, 1.0 - eps, N_c)
                
                # m_grid shape (N_m,)
                C_cand = m_grid[:, None] * c_fracs[None, :]
                
                J = u(C_cand)
                
                for zp in range(2):
                    p = transition[z, zp]
                    if p < 1e-12:
                        continue
                    m_next = R * (m_grid[:, None] - C_cand) + income[zp]
                    V_next_vals = np.interp(m_next, m_fine, V[t+1][zp])
                    J += beta * p * V_next_vals
                    
                idx = np.argmax(J, axis=1)
                policy[t, z, :] = C_cand[np.arange(N_m), idx]
                
                # Enforce constraints
                policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, m_grid)

    return policy
