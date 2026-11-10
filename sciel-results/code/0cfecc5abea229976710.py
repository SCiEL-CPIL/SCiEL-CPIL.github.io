"""Consumption-saving policy via vectorized backward induction with parabolic refinement and local re-search."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def solve_consumption(model, resource_grid, budget):
    """
    Solve the consumption-saving problem via backward induction with
    vectorized objective evaluation and parabolic interpolation refinement.
    
    Parameters:
    -----------
    model : dict with keys 'beta', 'gamma', 'R', 'income', 'transition', 'horizon'
    resource_grid : np.array of shape (101,)
    budget : int (number of consumption grid points per state)
    
    Returns:
    --------
    consumption : np.array of shape (6, 2, 101)
    """
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)
    m_min = m_grid[0]
    m_max = m_grid[-1]
    
    # Precompute utility
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            return np.log(np.maximum(c, 1e-12))
    else:
        one_minus_gamma = 1.0 - gamma
        def utility(c):
            return np.power(np.maximum(c, 1e-12), one_minus_gamma) / one_minus_gamma
    
    # Terminal: c[horizon-1] = m (consume all)
    consumption = np.zeros((horizon, n_z, n_m))
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # V_next[z, m_idx] = value at t+1
    V_next = np.tile(utility(m_grid)[np.newaxis, :], (n_z, 1))  # shape (n_z, n_m)
    
    # Number of consumption points to scan
    n_scan = max(budget, 200)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fill remaining with consume-all heuristic
            for tt in range(t, horizon - 1):
                consumption[tt] = m_grid[np.newaxis, :]
            break
        
        V_curr = np.zeros((n_z, n_m))
        C_opt = np.zeros((n_z, n_m))
        
        for z in range(n_z):
            trans_z = transition[z]
            
            # Use fractional grid: c = frac * m, where frac in [frac_min, 1]
            frac_min = 0.001
            fracs = np.linspace(frac_min, 1.0, n_scan)
            
            # c_candidates[m_idx, scan_idx] = fracs[scan_idx] * m_grid[m_idx]
            c_candidates = fracs[np.newaxis, :] * m_grid[:, np.newaxis]  # shape (n_m, n_scan)
            
            # m_minus_c shape: (n_m, n_scan)
            m_minus_c = m_grid[:, np.newaxis] - c_candidates  # (n_m, n_scan)
            m_minus_c = np.maximum(m_minus_c, 0.0)
            
            # m_prime for each z': shape (n_z, n_m, n_scan)
            m_prime_all = R * m_minus_c[np.newaxis, :, :] + income[:, np.newaxis, np.newaxis]
            m_prime_all = np.clip(m_prime_all, m_min, m_max)
            
            # Compute continuation value for each z', m, c
            cont_all = np.zeros((n_z, n_m, n_scan))
            for zp in range(n_z):
                mp_flat = m_prime_all[zp].ravel()
                cont_flat = np.interp(mp_flat, m_grid, V_next[zp, :])
                cont_all[zp] = cont_flat.reshape(n_m, n_scan)
            
            # Weighted continuation: sum over z' of trans_z[z'] * cont_all[z']
            cont_weighted = np.tensordot(trans_z, cont_all, axes=(0, 0))  # (n_m, n_scan)
            
            # Utility of consumption: shape (n_m, n_scan)
            util_c = utility(c_candidates)
            
            # Total objective: (n_m, n_scan)
            obj_matrix = util_c + beta * cont_weighted
            
            # Find best scan index for each m
            best_idx = np.argmax(obj_matrix, axis=1)  # (n_m,)
            
            # Get best c from scan
            best_c_scan = c_candidates[np.arange(n_m), best_idx]
            
            # Vectorized parabolic interpolation refinement
            refined_c = best_c_scan.copy()
            
            interior_mask = (best_idx > 0) & (best_idx < n_scan - 1)
            
            if np.any(interior_mask):
                idxs = np.where(interior_mask)[0]
                bi = best_idx[idxs]
                
                c0_arr = c_candidates[idxs, bi - 1]
                c1_arr = c_candidates[idxs, bi]
                c2_arr = c_candidates[idxs, bi + 1]
                
                f0_arr = obj_matrix[idxs, bi - 1]
                f1_arr = obj_matrix[idxs, bi]
                f2_arr = obj_matrix[idxs, bi + 1]
                
                d01 = c1_arr - c0_arr
                d12 = c2_arr - c1_arr
                
                valid = (np.abs(d01) > 1e-15) & (np.abs(d12) > 1e-15)
                
                if np.any(valid):
                    idx_valid = idxs[valid]
                    d01_v = d01[valid]
                    d12_v = d12[valid]
                    f0_v = f0_arr[valid]
                    f1_v = f1_arr[valid]
                    f2_v = f2_arr[valid]
                    c1_v = c1_arr[valid]
                    c0_v = c0_arr[valid]
                    c2_v = c2_arr[valid]
                    
                    num = 0.5 * (d01_v**2 * (f1_v - f2_v) - d12_v**2 * (f1_v - f0_v))
                    den = d01_v * (f1_v - f2_v) - d12_v * (f1_v - f0_v)
                    
                    valid2 = np.abs(den) > 1e-15
                    if np.any(valid2):
                        x_star = c1_v[valid2] + num[valid2] / den[valid2]
                        in_bounds = (x_star >= c0_v[valid2]) & (x_star <= c2_v[valid2])
                        
                        if np.any(in_bounds):
                            m_idx_valid = idx_valid[valid2][in_bounds]
                            x_star_valid = x_star[in_bounds]
                            
                            # Evaluate objective at x_star
                            m_vals = m_grid[m_idx_valid]
                            c_vals = x_star_valid
                            mc_vals = np.maximum(0.0, m_vals - c_vals)
                            mp_vals = R * mc_vals[:, np.newaxis] + income[np.newaxis, :]
                            mp_vals = np.clip(mp_vals, m_min, m_max)
                            
                            cont_vals = np.zeros(len(m_idx_valid))
                            for zp in range(n_z):
                                cont_vals += trans_z[zp] * np.interp(mp_vals[:, zp], m_grid, V_next[zp, :])
                            f_star = utility(c_vals) + beta * cont_vals
                            
                            # Accept if better
                            f1_ref = f1_v[valid2][in_bounds]
                            accept = f_star > f1_ref
                            if np.any(accept):
                                refined_c[m_idx_valid[accept]] = x_star_valid[accept]
            
            # Clamp refined_c
            refined_c = np.clip(refined_c, 1e-12, m_grid)
            
            # Local re-search: finer grid around refined_c
            # Use a small window around the refined point
            local_n = 50
            m_vals = m_grid
            c_centers = refined_c
            # Window: 2% of m on each side, but at least 1e-6
            window = np.maximum(0.02 * m_vals, 1e-6)
            c_local_min = np.maximum(c_centers - window, 1e-12)
            c_local_max = np.minimum(c_centers + window, m_vals)
            
            # Only do local search where the center is not at boundary
            local_mask = (c_local_min < c_local_max) & (c_centers > 1e-11)
            
            if np.any(local_mask):
                lm_idx = np.where(local_mask)[0]
                n_local_pts = len(lm_idx)
                
                # For each m_idx in lm_idx, create local grid
                c_local = np.zeros((n_local_pts, local_n))
                for i, mi in enumerate(lm_idx):
                    lo = c_local_min[mi]
                    hi = c_local_max[mi]
                    c_local[i] = np.linspace(lo, hi, local_n)
                
                # Compute objective for local candidates
                m_loc = m_grid[lm_idx]
                mc_loc = np.maximum(0.0, m_loc[:, np.newaxis] - c_local)
                mp_loc = R * mc_loc[:, np.newaxis] + income[np.newaxis, :]
                mp_loc = np.clip(mp_loc, m_min, m_max)
                
                cont_loc = np.zeros((n_local_pts, local_n))
                for zp in range(n_z):
                    cont_loc += trans_z[zp] * np.interp(mp_loc[:, zp], m_grid, V_next[zp, :])
                
                util_loc = utility(c_local)
                obj_loc = util_loc + beta * cont_loc
                
                best_local_idx = np.argmax(obj_loc, axis=1)
                best_local_c = c_local[np.arange(n_local_pts), best_local_idx]
                
                # Compare with current refined_c
                current_c = refined_c[lm_idx]
                current_obj = obj_matrix[lm_idx, best_idx[lm_idx]]
                local_obj = obj_loc[np.arange(n_local_pts), best_local_idx]
                
                # Accept local if better
                accept_local = local_obj > current_obj
                if np.any(accept_local):
                    refined_c[lm_idx[accept_local]] = best_local_c[accept_local]
            
            # Final clamp
            refined_c = np.clip(refined_c, 1e-12, m_grid)
            C_opt[z] = refined_c
            
            # Compute V_curr[z] using the refined c
            mc = np.maximum(0.0, m_grid - refined_c)
            mp = R * mc[:, np.newaxis] + income[np.newaxis, :]  # (n_m, n_z)
            mp = np.clip(mp, m_min, m_max)
            
            cont_val = np.zeros(n_m)
            for zp in range(n_z):
                cont_val += trans_z[zp] * np.interp(mp[:, zp], m_grid, V_next[zp, :])
            
            V_curr[z] = utility(refined_c) + beta * cont_val
        
        consumption[t] = C_opt
        V_next = V_curr
    
    # Ensure terminal condition
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # Validate: 0 < c <= m + 1e-10, finite
    for t in range(horizon):
        for z in range(n_z):
            for m_idx in range(n_m):
                m_val = m_grid[m_idx]
                c_val = consumption[t, z, m_idx]
                if not np.isfinite(c_val) or c_val <= 0:
                    consumption[t, z, m_idx] = max(1e-12, m_val)
                if c_val > m_val + 1e-10:
                    consumption[t, z, m_idx] = m_val
    
    return consumption
# EVOLVE-BLOCK-END
