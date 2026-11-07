import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    c_min = 1e-12
    T = horizon
    n_z = 2
    grid_out = resource_grid
    n_out = len(grid_out)
    
    def utility(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
            
    def utility_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    def utility_double_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (c ** 2)
        else:
            return -gamma * c ** (-gamma - 1)

    # Internal grid for value function approximation
    # 1000 points is a good balance. Geomspace handles the wide range well.
    n_inner = 1000
    grid = np.geomspace(0.001, 40, n_inner)
    n_grid = len(grid)
    
    V = [None] * T
    P = [None] * T
    
    # Terminal condition
    V[T-1] = utility(grid)[np.newaxis, :].repeat(n_z, axis=0)
    P[T-1] = grid[np.newaxis, :].repeat(n_z, axis=0)
    
    # Precompute finite difference coefficients for derivatives
    # We need dV and d2V of V_next to use in Newton refinement
    h_minus = grid[1:-1] - grid[:-2]
    h_plus = grid[2:] - grid[1:-1]
    
    # Interior coefficients
    L0p_int = h_plus / (h_minus * (h_minus + h_plus))
    L1p_int = (h_plus - h_minus) / (h_minus * h_plus)
    L2p_int = h_minus / ((h_minus + h_plus) * h_plus)
    d2V_coeff_int = 2.0 / (h_plus + h_minus)
    
    # Edge i=0
    h0 = grid[1] - grid[0]
    h1 = grid[2] - grid[1]
    d01 = h0
    d02 = h0 + h1
    d12 = h1
    L0p_edge0 = ( -d02 - d01 ) / ( -d01 * -d02 )
    L1p_edge0 = -d02 / ( d01 * -d12 )
    L2p_edge0 = -d01 / ( d02 * d12 )
    d2V_coeff_edge0 = 2.0 / (h0 + h1)
    
    # Edge i=n_grid-1
    h_m1 = grid[-1] - grid[-2]
    h_m2 = grid[-2] - grid[-3]
    d_m1 = h_m1
    d_m2 = h_m2
    d_m12 = h_m1 + h_m2
    L_0p_edge1 = d_m1 / (d_m2 * d_m12)
    L_1p_edge1 = -d_m12 / (d_m2 * d_m1)
    L_2p_edge1 = 1.0 / d_m1
    d2V_coeff_edge1 = 2.0 / (h_m1 + h_m2)

    def compute_derivatives(V_z):
        dV = np.zeros(n_grid)
        d2V = np.zeros(n_grid)
        
        if n_grid > 2:
            dV[1:-1] = V_z[:-2]*L0p_int + V_z[1:-1]*L1p_int + V_z[2:]*L2p_int
            d2V[1:-1] = d2V_coeff_int * ( (V_z[2:] - V_z[1:-1]) / h_plus - (V_z[1:-1] - V_z[:-2]) / h_minus )
            
            dV[0] = V_z[0]*L0p_edge0 + V_z[1]*L1p_edge0 + V_z[2]*L2p_edge0
            d2V[0] = d2V_coeff_edge0 * ( (V_z[1] - V_z[0]) / h0 - (V_z[2] - V_z[1]) / h1 )
            
            dV[-1] = V_z[-3]*L_0p_edge1 + V_z[-2]*L_1p_edge1 + V_z[-1]*L_2p_edge1
            d2V[-1] = d2V_coeff_edge1 * ( (V_z[-1] - V_z[-2]) / h_m1 - (V_z[-2] - V_z[-3]) / h_m2 )
            
        return dV, d2V

    # Number of consumption fraction candidates for the grid search
    # 200 is sufficient to locate the optimum, Newton will refine it.
    n_fracs = 200
    fracs = np.linspace(0.001, 1.0, n_fracs)
    
    # Backward induction
    for t in range(T-2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: use previous period's policy
            for t_rem in range(t, -1, -1):
                V[t_rem] = V[t+1].copy()
                P[t_rem] = P[t+1].copy()
            break
            
        V_t = np.zeros((n_z, n_grid))
        P_t = np.zeros((n_z, n_grid))
        V_next = V[t+1]
        
        # Precompute derivatives for next period
        dV_next = [None] * n_z
        d2V_next = [None] * n_z
        for zp in range(n_z):
            dV_next[zp], d2V_next[zp] = compute_derivatives(V_next[zp])
            
        for z in range(n_z):
            # 1. Grid Search for Initial Guess
            # Create mesh of c candidates
            # c[i, j] = grid[i] * fracs[j]
            c_cands = grid[:, None] * fracs[None, :]
            c_cands = np.maximum(c_cands, c_min)
            
            # Compute Utility
            U_c = utility(c_cands)
            
            # Compute Continuation Value
            cont_val = np.zeros_like(c_cands)
            for zp in range(n_z):
                m_next = R * (grid[:, None] - c_cands) + income[zp]
                m_next_clipped = np.clip(m_next, grid[0], grid[-1])
                
                # Interpolate V_next[zp]
                # Flatten for np.interp, then reshape
                V_interp = np.interp(m_next_clipped.flatten(), grid, V_next[zp])
                V_interp = V_interp.reshape(n_grid, n_fracs)
                
                cont_val += transition[z, zp] * V_interp
            
            Obj = U_c + beta * cont_val
            
            # Find best fraction index for each m
            best_idx = np.argmax(Obj, axis=1)
            best_c = c_cands[np.arange(n_grid), best_idx]
            
            # 2. Newton Refinement
            # We refine best_c using the Euler equation:
            # u'(c) = beta * sum_zp pi_zp * V'_next[zp](m')
            # Where m' = R(m-c) + income[zp]
            
            c_refined = best_c.copy()
            
            for _ in range(5):
                G_c = np.zeros(n_grid)
                G_prime_c = np.zeros(n_grid)
                
                for zp in range(n_z):
                    next_m = R * (grid - c_refined) + income[zp]
                    next_m_clipped = np.clip(next_m, grid[0], grid[-1])
                    
                    dv = np.interp(next_m_clipped, grid, dV_next[zp])
                    d2v = np.interp(next_m_clipped, grid, d2V_next[zp])
                    
                    G_c += transition[z, zp] * dv
                    G_prime_c += transition[z, zp] * d2v * (-R)
                
                G_c *= beta
                G_prime_c *= beta
                
                f_val = utility_prime(c_refined) - G_c
                f_prime_val = utility_double_prime(c_refined) - G_prime_c
                
                # Newton step
                mask = np.abs(f_prime_val) > 1e-10
                step = np.zeros(n_grid)
                step[mask] = f_val[mask] / f_prime_val[mask]
                
                c_new = c_refined - step
                c_new = np.clip(c_new, c_min, grid)
                
                if np.max(np.abs(c_new - c_refined)) < 1e-10:
                    c_refined = c_new
                    break
                c_refined = c_new
                
            P_t[z, :] = c_refined
            
            # Calculate value at optimal c
            cont_val_final = np.zeros(n_grid)
            for zp in range(n_z):
                next_m = R * (grid - c_refined) + income[zp]
                next_m_clipped = np.clip(next_m, grid[0], grid[-1])
                cont_val_final += transition[z, zp] * np.interp(next_m_clipped, grid, V_next[zp])
            
            V_t[z, :] = utility(c_refined) + beta * cont_val_final
        
        V[t] = V_t
        P[t] = P_t
    
    # Interpolate policy back to the original grid
    result = np.zeros((T, n_z, n_out))
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.interp(grid_out, grid, P[t][z])
    
    # Enforce constraints
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.clip(result[t, z, :], c_min, grid_out + 1e-10)
            
    # Terminal condition must be exact on the output grid
    result[T-1, :, :] = grid_out[np.newaxis, :]
    
    return result
