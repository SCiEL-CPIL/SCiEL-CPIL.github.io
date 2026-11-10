import numpy as np
from scipy.interpolate import CubicSpline
import time

def solve_consumption(model, resource_grid, budget):
    """
    Solve the consumption-saving problem via backward induction using
    vectorized objective evaluation with Newton-Raphson refinement and
    Cubic Spline approximation for the value function.
    
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
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
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
    
    # Precompute utility and marginal utility
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            return np.log(np.maximum(c, 1e-12))
        def marginal_utility(c):
            c_safe = np.maximum(c, 1e-12)
            return 1.0 / c_safe
        def second_marginal_utility(c):
            c_safe = np.maximum(c, 1e-12)
            return -1.0 / (c_safe * c_safe)
    else:
        one_minus_gamma = 1.0 - gamma
        def utility(c):
            return np.power(np.maximum(c, 1e-12), one_minus_gamma) / one_minus_gamma
        def marginal_utility(c):
            c_safe = np.maximum(c, 1e-12)
            return np.power(c_safe, -gamma)
        def second_marginal_utility(c):
            c_safe = np.maximum(c, 1e-12)
            return -gamma * np.power(c_safe, -gamma - 1)
    
    # Terminal condition: at t=horizon-1, consume all resources
    consumption = np.zeros((horizon, n_z, n_m))
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # Value function at terminal: V(m, z) = u(m) for all z
    V_next = np.tile(utility(m_grid)[np.newaxis, :], (n_z, 1))
    
    # Initialize splines for V_next
    V_splines = []
    for z in range(n_z):
        try:
            V_splines.append(CubicSpline(m_grid, V_next[z], bc_type='natural'))
        except Exception:
            # Fallback to linear if spline fails (e.g., non-monotonic issues, though V should be monotonic)
            # We can use a simple linear interp wrapper or just use np.interp in the step
            V_splines.append(None)
    
    # Number of consumption scan points - use a fine grid for initial scan
    n_scan = max(budget, 300)
    
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
            m_minus_c = m_grid[:, np.newaxis] - c_candidates # (n_m, n_scan)
            m_minus_c = np.maximum(m_minus_c, 0.0)
            
            # m_prime for each z': shape (n_z, n_m, n_scan)
            m_prime_all = R * m_minus_c[np.newaxis, :, :] + income[:, np.newaxis, np.newaxis]
            m_prime_all = np.clip(m_prime_all, m_min, m_max)
            
            # Compute continuation value for each z', m, c
            # We need V_next[z'](m')
            cont_all = np.zeros((n_z, n_m, n_scan))
            for zp in range(n_z):
                mp_flat = m_prime_all[zp].ravel()
                if V_splines[zp] is not None:
                    cont_flat = V_splines[zp](mp_flat)
                else:
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
            
            # Newton-Raphson refinement
            refined_c = best_c_scan.copy()
            
            # Precompute V' and V'' for next period using splines
            # We need to evaluate these at m_prime which depends on c
            # So we do this inside the loop for each m, but vectorized over m?
            # No, m_prime depends on c, which is different for each m.
            # However, we can vectorize the Newton steps for all m simultaneously.
            
            c_curr = best_c_scan.copy()
            
            # Newton iterations
            for _newton_iter in range(10):
                if time.time() > deadline:
                    break
                
                # m_prime for current c: (n_z, n_m)
                m_minus_c_curr = np.maximum(m_grid[np.newaxis, :] - c_curr[np.newaxis, :], 0.0)
                m_prime_curr = R * m_minus_c_curr + income[:, np.newaxis]
                m_prime_curr = np.clip(m_prime_curr, m_min, m_max)
                
                # V' and V'' at m_prime: (n_z, n_m)
                V_prime_mp = np.zeros((n_z, n_m))
                V_double_prime_mp = np.zeros((n_z, n_m))
                
                for zp in range(n_z):
                    mp_flat = m_prime_curr[zp]
                    if V_splines[zp] is not None:
                        V_prime_mp[zp] = V_splines[zp](mp_flat, 1)
                        V_double_prime_mp[zp] = V_splines[zp](mp_flat, 2)
                    else:
                        # Fallback: central differences
                        # This is slow, but splines should work
                        d1 = (V_next[zp, 1] - V_next[zp, 0]) / (m_grid[1] - m_grid[0])
                        d2 = (V_next[zp, -1] - V_next[zp, -2]) / (m_grid[-1] - m_grid[-2])
                        V_prime_mp[zp] = np.interp(mp_flat, m_grid, V_prime_mp[zp] if False else np.interp(m_grid, m_grid, (V_next[zp, 1:] - V_next[zp, :-1]) / (m_grid[1:] - m_grid[:-1])))
                        # Actually, just use linear interp derivative approx
                        # For safety, let's just use a small epsilon diff if spline fails
                        eps = 1e-6
                        mp_up = np.clip(mp_flat + eps, m_min, m_max)
                        mp_dn = np.clip(mp_flat - eps, m_min, m_max)
                        V_up = np.interp(mp_up, m_grid, V_next[zp, :])
                        V_dn = np.interp(mp_dn, m_grid, V_next[zp, :])
                        V_mid = np.interp(mp_flat, m_grid, V_next[zp, :])
                        V_prime_mp[zp] = (V_up - V_dn) / (mp_up - mp_dn)
                        V_double_prime_mp[zp] = (V_up - 2*V_mid + V_dn) / (eps*eps)
                
                # Weighted V' and V''
                # E[V'] = sum_z' trans[z'] V'(m')
                # E[V''] = sum_z' trans[z'] V''(m')
                E_V_prime = np.tensordot(trans_z, V_prime_mp, axes=(0, 0)) # (n_m,)
                E_V_double_prime = np.tensordot(trans_z, V_double_prime_mp, axes=(0, 0)) # (n_m,)
                
                # FOC: f(c) = u'(c) - beta * R * E[V']
                # f'(c) = u''(c) - beta * R * d/dc(E[V'])
                # d/dc(E[V']) = E[V''] * d/dc(m') = E[V''] * (-R)
                # So f'(c) = u''(c) + beta * R^2 * E[V'']
                
                mu_c = marginal_utility(c_curr)
                mu2_c = second_marginal_utility(c_curr)
                
                f_c = mu_c - beta * R * E_V_prime
                f_prime_c = mu2_c + beta * R * R * E_V_double_prime
                
                # Newton step: c_new = c - f/f'
                # Safeguard: if f' is close to 0 or step is too large, use bisection-like step
                # Since problem is concave, f' should be negative (u'' < 0, E[V''] > 0 but term is positive? 
                # Wait. u'' < 0. E[V''] > 0. 
                # f'(c) = u''(c) + beta R^2 E[V''].
                # Usually u'' dominates, so f' < 0.
                # If f' > 0, we are in a region where the objective is convex (should not happen for concave utility/monotonic V).
                # Let's check signs.
                # u(c) concave => u'' < 0.
                # V(m) concave => V'' < 0.
                # So E[V''] < 0.
                # f'(c) = u''(c) + beta R^2 E[V''] < 0.
                # So f' is always negative.
                
                # Step size
                with np.errstate(divide='ignore', invalid='ignore'):
                    step = f_c / f_prime_c
                
                # Clamp step to avoid overshooting
                # Max step is half the distance to bounds
                max_step_up = m_grid - c_curr
                max_step_dn = c_curr - 1e-12
                # We want c_new = c_curr - step
                # If step > 0, c decreases. Max decrease is c_curr - 1e-12.
                # If step < 0, c increases. Max increase is m_grid - c_curr.
                
                step = np.where(step > 0, np.minimum(step, max_step_dn * 0.5), step)
                step = np.where(step < 0, np.maximum(step, -max_step_up * 0.5), step)
                
                c_new = c_curr - step
                c_new = np.clip(c_new, 1e-12, m_grid)
                
                # Check convergence
                if np.max(np.abs(c_new - c_curr)) < 1e-10 * m_max:
                    c_curr = c_new
                    break
                
                c_curr = c_new
            
            refined_c = c_curr
            C_opt[z, :] = refined_c
            
            # Compute V_curr using the refined consumption
            # Vectorized calculation of V_curr
            m_minus_c_final = np.maximum(m_grid[np.newaxis, :] - C_opt[z, np.newaxis, :], 0.0) # (1, n_m)
            m_prime_final = R * m_minus_c_final + income[:, np.newaxis] # (n_z, n_m)
            m_prime_final = np.clip(m_prime_final, m_min, m_max)
            
            cont_final = np.zeros((n_z, n_m))
            for zp in range(n_z):
                if V_splines[zp] is not None:
                    cont_final[zp] = V_splines[zp](m_prime_final[zp])
                else:
                    cont_final[zp] = np.interp(m_prime_final[zp], m_grid, V_next[zp, :])
            
            cont_val = np.tensordot(trans_z, cont_final, axes=(0, 0)) # (n_m,)
            util_final = utility(C_opt[z, :])
            V_curr[z, :] = util_final + beta * cont_val
        
        consumption[t] = C_opt
        V_next = V_curr
        
        # Update splines for next iteration
        V_splines = []
        for z in range(n_z):
            try:
                V_splines.append(CubicSpline(m_grid, V_next[z], bc_type='natural'))
            except Exception:
                V_splines.append(None)
        
    # Ensure terminal condition: c[5] = grid
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # Validate: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            for m_idx in range(n_m):
                m_val = m_grid[m_idx]
                c_val = consumption[t, z, m_idx]
                if c_val <= 0:
                    consumption[t, z, m_idx] = 1e-12
                if c_val > m_val + 1e-10:
                    consumption[t, z, m_idx] = m_val
    
    return consumption
