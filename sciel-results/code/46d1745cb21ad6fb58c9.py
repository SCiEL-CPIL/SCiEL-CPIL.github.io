"""Consumption-saving policy via vectorized backward induction with parabolic refinement."""
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
    
    # Precompute interpolation for speed: use np.interp vectorized
    def interp_V(m_prime_array, z_prime):
        """Interpolate V_next[z_prime] at m_prime_array (array of m values)."""
        return np.interp(m_prime_array, m_grid, V_next[z_prime, :])
    
    # Number of consumption points to scan
    # Use more points for better accuracy
    n_scan = max(budget, 400)
    
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
            
            # For each m in m_grid, we need to find optimal c
            # c ranges from eps*m to m
            # Vectorize: for all m values, compute c candidates as fractions
            
            # Use fractional grid: c = frac * m, where frac in [frac_min, 1]
            # This handles the constraint c <= m naturally
            frac_min = 0.001
            fracs = np.linspace(frac_min, 1.0, n_scan)
            
            # For small m, the absolute step size is tiny, so we may want more points
            # But for large m, we might want fewer. Let's stick with uniform for now.
            
            # c_candidates[m_idx, scan_idx] = fracs[scan_idx] * m_grid[m_idx]
            c_candidates = fracs[np.newaxis, :] * m_grid[:, np.newaxis]  # shape (n_m, n_scan)
            
            # For each m and each c candidate, compute the objective
            # m_prime[z'] = R * (m - c) + income[z']
            # We need to compute this for all m, c, z'
            
            # m_minus_c shape: (n_m, n_scan)
            m_minus_c = m_grid[:, np.newaxis] - c_candidates  # (n_m, n_scan)
            m_minus_c = np.maximum(m_minus_c, 0.0)
            
            # m_prime for each z': shape (n_z, n_m, n_scan)
            m_prime_all = R * m_minus_c[np.newaxis, :, :] + income[:, np.newaxis, np.newaxis]
            m_prime_all = np.clip(m_prime_all, m_min, m_max)
            
            # Compute continuation value for each z', m, c
            # cont[z', m, c] = V_next[z', interp(m_prime[z', m, c])]
            cont_all = np.zeros((n_z, n_m, n_scan))
            for zp in range(n_z):
                # m_prime_all[zp] has shape (n_m, n_scan)
                # Need to interpolate V_next[zp] at each point
                # Reshape for np.interp which works on 1D
                mp_flat = m_prime_all[zp].ravel()
                cont_flat = np.interp(mp_flat, m_grid, V_next[zp, :])
                cont_all[zp] = cont_flat.reshape(n_m, n_scan)
            
            # Weighted continuation: sum over z' of trans_z[z'] * cont_all[z']
            # Shape: (n_m, n_scan)
            cont_weighted = np.tensordot(trans_z, cont_all, axes=(0, 0))  # (n_m, n_scan)
            
            # Utility of consumption: shape (n_m, n_scan)
            util_c = utility(c_candidates)
            
            # Total objective: (n_m, n_scan)
            obj_matrix = util_c + beta * cont_weighted
            
            # Find best scan index for each m
            best_idx = np.argmax(obj_matrix, axis=1)  # (n_m,)
            
            # Get best c from scan
            best_c_scan = c_candidates[np.arange(n_m), best_idx]
            
            # Parabolic interpolation refinement around best point
            # Vectorized version for speed
            refined_c = best_c_scan.copy()
            
            # Get indices for interior points
            interior_mask = (best_idx > 0) & (best_idx < n_scan - 1)
            
            if np.any(interior_mask):
                idxs = np.where(interior_mask)[0]
                
                # Get c values and f values for the three points
                # c0, c1, c2 for each m_idx in idxs
                c0_arr = c_candidates[idxs, best_idx[idxs] - 1]
                c1_arr = c_candidates[idxs, best_idx[idxs]]
                c2_arr = c_candidates[idxs, best_idx[idxs] + 1]
                
                f0_arr = obj_matrix[idxs, best_idx[idxs] - 1]
                f1_arr = obj_matrix[idxs, best_idx[idxs]]
                f2_arr = obj_matrix[idxs, best_idx[idxs] + 1]
                
                d01 = c1_arr - c0_arr
                d12 = c2_arr - c1_arr
                
                valid_den = np.abs(d01) > 1e-15
                valid_den &= np.abs(d12) > 1e-15
                
                num = 0.5 * (d01**2 * (f1_arr - f2_arr) - d12**2 * (f1_arr - f0_arr))
                den = d01 * (f1_arr - f2_arr) - d12 * (f1_arr - f0_arr)
                
                valid_den &= np.abs(den) > 1e-15
                
                if np.any(valid_den):
                    x_star = c1_arr[valid_den] + num[valid_den] / den[valid_den]
                    
                    # Check bounds
                    c0_valid = c0_arr[valid_den]
                    c2_valid = c2_arr[valid_den]
                    in_bounds = (x_star >= c0_valid) & (x_star <= c2_valid)
                    
                    if np.any(in_bounds):
                        # Evaluate objective at x_star for valid candidates
                        m_idx_valid = idxs[valid_den][in_bounds]
                        x_star_valid = x_star[in_bounds]
                        
                        # Compute objective for these points
                        m_vals = m_grid[m_idx_valid]
                        c_vals = x_star_valid
                        mc_vals = np.maximum(0.0, m_vals - c_vals)
                        
                        # m_prime for each z'
                        mp_vals = R * mc_vals[:, np.newaxis] + income[np.newaxis, :]
                        mp_vals = np.clip(mp_vals, m_min, m_max)
                        
                        # Continuation value
                        cont_vals = np.zeros(len(m_idx_valid))
                        for zp in range(n_z):
                            cont_vals += trans_z[zp] * np.interp(mp_vals[:, zp], m_grid, V_next[zp, :])
                        
                        f_star_vals = utility(c_vals) + beta * cont_vals
                        f1_vals = f1_arr[valid_den][in_bounds]
                        
                        # Update if better
                        better_mask = f_star_vals > f1_vals
                        if np.any(better_mask):
                            refined_c[m_idx_valid[better_mask]] = x_star_valid[better_mask]
            
            # Clamp refined_c
            refined_c = np.clip(refined_c, 1e-12, m_grid)
            
            C_opt[z] = refined_c
            
            # Compute V_curr[z] using the refined c
            # Re-evaluate objective at refined_c
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
    
    # Validate: 0 < c <= m + 1e-10, finite (vectorized)
    for t in range(horizon):
        for z in range(n_z):
            c_vals = consumption[t, z, :]
            m_vals = m_grid
            
            # Fix non-finite or non-positive
            bad_mask = ~np.isfinite(c_vals) | (c_vals <= 0)
            if np.any(bad_mask):
                c_vals[bad_mask] = np.maximum(1e-12, m_vals[bad_mask])
            
            # Fix c > m
            too_big_mask = c_vals > m_vals + 1e-10
            if np.any(too_big_mask):
                c_vals[too_big_mask] = m_vals[too_big_mask]
            
            consumption[t, z, :] = c_vals
    
    return consumption
# EVOLVE-BLOCK-END
