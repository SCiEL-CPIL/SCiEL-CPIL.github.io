import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Grids
    M_res = resource_grid # 101 points
    # V_grid needs to cover max possible m'.
    # max m = 40. max m' = 1.02*40 + 1.5 = 42.3.
    # min m = 0.001. min m' = 0.001*1.02 + 0.5 = 0.501.
    # Let's use 0.001 to 50.
    N_V = 1000
    V_grid = np.geomspace(0.001, 50, N_V)
    
    # Initialize Value Function V_t(m, z)
    # V[0] is V_5, V[1] is V_4, etc.
    # We store V as array of shape (2, N_V) for each t.
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    # V_next: shape (2, N_V)
    # t = horizon - 1 = 5
    # V_5(m, z) = util(m)
    V_next = np.empty((2, N_V))
    for z in range(2):
        V_next[z] = util(V_grid)
        
    # Policy output: shape (horizon, 2, len(M_res))
    policy = np.zeros((horizon, 2, len(M_res)))
    
    # Fill t = horizon - 1 (index 5)
    # c = m
    for z in range(2):
        policy[horizon-1, z, :] = M_res
        
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline:
             # Return best effort (current policy)
             break
             
        V_curr = np.empty((2, N_V))
        
        for z in range(2):
            # 1. Compute V_curr on V_grid
            # 2. Compute Policy on M_res
            
            # We will solve for optimal c for a given m using a candidate grid
            # Candidate c: m * linspace(0.001, 1, 200)
            
            # To be efficient, we can vectorize over the grid points.
            # However, the candidate c grid depends on m.
            # Let's loop over the grid points (V_grid and M_res) or combine them.
            
            # Combined grid for solving:
            # We need V_curr on V_grid.
            # We need Policy on M_res.
            # Let's just loop over V_grid to get V_curr, and store c_opt.
            # Then extract Policy from V_curr's c_opt? No, M_res != V_grid.
            
            # Let's solve on a union or just solve on V_grid and interpolate c_opt to M_res?
            # Or solve on M_res separately.
            
            # Let's solve on V_grid first.
            c_opts_V = np.zeros(N_V)
            vals_V = np.zeros(N_V)
            
            for i, m in enumerate(V_grid):
                # Candidate c
                c_cands = m * np.linspace(0.001, 1.0, 200)
                
                # Next wealths
                m_prime_0 = R * (m - c_cands) + income[0]
                m_prime_1 = R * (m - c_cands) + income[1]
                
                # Interpolate V_next
                # Ensure m_prime is within V_grid bounds
                # V_grid is 0.001 to 50. m_prime min ~0.5, max ~42. Safe.
                v0 = np.interp(m_prime_0, V_grid, V_next[0])
                v1 = np.interp(m_prime_1, V_grid, V_next[1])
                
                # Expected continuation value
                cont_val = transition[z, 0] * v0 + transition[z, 1] * v1
                
                # Total value
                total_val = util(c_cands) + beta * cont_val
                
                # Find max
                idx_max = np.argmax(total_val)
                vals_V[i] = total_val[idx_max]
                c_opts_V[i] = c_cands[idx_max]
                
            V_curr[z] = vals_V
            
            # Now, get policy for M_res
            # We can either interpolate c_opts_V from V_grid to M_res, or re-solve.
            # Interpolation is faster and likely accurate enough given N_V=1000.
            # c(m) is smooth.
            policy[t, z, :] = np.interp(M_res, V_grid, c_opts_V)
            
        V_next = V_curr

    return policy
