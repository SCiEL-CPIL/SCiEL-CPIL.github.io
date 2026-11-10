"""Consumption-saving policy via vectorized backward induction with Newton refinement on splines."""
import numpy as np
import time
from scipy.interpolate import CubicSpline

# EVOLVE-BLOCK-START
def solve_consumption(model, resource_grid, budget):
    """
    Solve the consumption-saving problem via backward induction using
    vectorized objective evaluation with Newton-Raphson refinement
    and Cubic Spline approximation for the value function.
    
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
    deadline = start_time + 138.0  # safety margin before 150s kill
    
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
    
    # Build initial splines
    V_splines = []
    for z in range(n_z):
        try:
            V_splines.append(CubicSpline(m_grid, V_next[z], bc_type='natural'))
        except Exception:
            V_splines.append(None)
    
    # Number of consumption scan points
    n_scan = max(budget, 300)
    
    def eval_V_prime(zp, mp):
        """Evaluate V' for state zp at resource level mp (scalar or array)."""
        if V_splines[zp] is not None:
            return V_splines[zp](mp, 1)
        else:
            return np.interp(mp, m_grid, (V_next[zp, 1:] - V_next[zp, :-1]) / (m_grid[1:] - m_grid[:-1]))
    
    def eval_V_double_prime(zp, mp):
        """Evaluate V'' for state zp at resource level mp (scalar or array)."""
        if V_splines[zp] is not None:
            return V_splines[zp](mp, 2)
        else:
            # Approximate second derivative via central difference of first derivative
            eps = 1e-4
            mp_up = np.clip(mp + eps, m_min, m_max)
            mp_dn = np.clip(mp - eps, m_min, m_max)
            return (eval_V_prime(zp, mp_up) - eval_V_prime(zp, mp_dn)) / (mp_up - mp_dn)
    
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
            
            # Newton-Raphson refinement (vectorized over m)
            c_curr = best_c_scan.copy()
            
            for _newton_iter in range(8):
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
                    V_prime_mp[zp] = eval_V_prime(zp, m_prime_curr[zp])
                    V_double_prime_mp[zp] = eval_V_double_prime(zp, m_prime_curr[zp])
                
                # Weighted derivatives
                w_prime = trans_z @ V_prime_mp  # (n_m,)
                w_double_prime = trans_z @ V_double_prime_mp  # (n_m,)
                
                # FOC: u'(c) - beta * R * sum_z' trans[z'][z'] * V'(m') = 0
                # Hessian: u''(c) - beta * R^2 * sum_z' trans[z'][z'] * V''(m') = 0
                mu = marginal_utility(c_curr)  # (n_m,)
                mu_prime = second_marginal_utility(c_curr)  # (n_m,)
                
                f = mu - beta * R * w_prime  # (n_m,)
                df = mu_prime - beta * R * R * w_double_prime  # (n_m,)
                
                # Newton step with damping
                # Avoid division by zero
                df_safe = np.where(np.abs(df) < 1e-30, 1e-30 * np.sign(df), df)
                step = f / df_safe
                
                # Damp the step
                c_new = c_curr - step
                c_new = np.clip(c_new, 1e-12, m_grid)
                
                # Check convergence
                if np.max(np.abs(c_new - c_curr)) < 1e-12 * m_max:
                    c_curr = c_new
                    break
                c_curr = c_new
            
            C_opt[z, :] = c_curr
            
            # Compute V_curr using the refined consumption
            for m_idx in range(n_m):
                m_val = m_grid[m_idx]
                c_val = C_opt[z, m_idx]
                c_safe = np.maximum(c_val, 1e-12)
                m_minus_c = np.maximum(m_val - c_safe, 0.0)
                m_prime = R * m_minus_c + income
                m_prime_clipped = np.clip(m_prime, m_min, m_max)
                
                cont = np.zeros(n_z)
                for zp in range(n_z):
                    if V_splines[zp] is not None:
                        cont[zp] = V_splines[zp](m_prime_clipped[zp])
                    else:
                        cont[zp] = np.interp(m_prime_clipped[zp], m_grid, V_next[zp, :])
                cont_val = np.dot(trans_z, cont)
                
                V_curr[z, m_idx] = utility(c_safe) + beta * cont_val
        
        consumption[t] = C_opt
        V_next = V_curr
        
        # Rebuild splines for next iteration
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
# EVOLVE-BLOCK-END
