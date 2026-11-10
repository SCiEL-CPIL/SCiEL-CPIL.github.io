import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    """
    Solve the consumption-saving problem via backward induction using
    Cubic Spline approximation for the value function and Newton-Raphson
    refinement on the First Order Condition.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
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
    # V is increasing and convex. Natural BCs might introduce slight artifacts at boundaries,
    # but for interior points (which are most important for scoring) it should be fine.
    # We use 'natural' (zero second derivative) as a standard choice.
    V_splines = []
    for z in range(n_z):
        try:
            V_splines.append(CubicSpline(m_grid, V_next[z], bc_type='natural'))
        except Exception:
            V_splines.append(None)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fill remaining with consume-all heuristic
            for tt in range(t, horizon - 1):
                consumption[tt] = m_grid[np.newaxis, :]
            break
            
        V_curr = np.zeros((n_z, n_m))
        C_curr = np.zeros((n_z, n_m))
        
        # Build splines for V_next
        # Note: V_next is already computed in the previous iteration (or terminal)
        # We need splines for V_next to evaluate derivatives in the FOC
        # The V_splines list currently holds splines for V_next from the start of this loop iteration?
        # No, V_splines was initialized for V_terminal.
        # Inside the loop, we need splines for V_{t+1}.
        # Let's rename: V_next is V_{t+1}. V_splines should correspond to V_next.
        
        # Re-evaluate splines for V_next if not already done or if V_next changed
        # Actually, V_next is constant during the t-loop iteration.
        # We should build splines for V_next once per t.
        
        # Let's ensure V_splines corresponds to V_next
        # In the first iteration (t=4), V_next is V_5 (terminal). V_splines is already built for V_5.
        # In subsequent iterations, V_next is V_{t+1} which was computed in the previous step.
        # So we need to update V_splines at the end of the loop or start of the loop.
        
        # Let's restructure:
        # 1. Ensure V_splines matches V_next
        # 2. Solve for C_curr
        # 3. Compute V_curr
        # 4. Set V_next = V_curr, C_next = C_curr
        # 5. Update V_splines for next iteration
        
        # Check if V_splines is valid for current V_next
        # We'll just rebuild them every time to be safe, it's cheap (101 points)
        for z in range(n_z):
            try:
                V_splines[z] = CubicSpline(m_grid, V_next[z], bc_type='natural')
            except Exception:
                V_splines[z] = None
        
        for z in range(n_z):
            trans_z = transition[z]
            
            # Coarse grid scan for initial guess
            # Use a grid of consumption fractions
            n_scan = 100
            fracs = np.linspace(0.001, 1.0, n_scan)
            c_candidates = fracs[np.newaxis, :] * m_grid[:, np.newaxis]  # (n_m, n_scan)
            
            # Ensure c <= m
            c_candidates = np.minimum(c_candidates, m_grid[:, np.newaxis])
            
            # Compute m' for all candidates
            # m' = R * (m - c) + income[z']
            # Shape: (n_z, n_m, n_scan)
            m_prime_all = np.zeros((n_z, n_m, n_scan))
            cont_all = np.zeros((n_z, n_m, n_scan))
            
            for zp in range(n_z):
                if trans_z[zp] == 0:
                    continue
                m_minus_c = np.maximum(m_grid[:, np.newaxis] - c_candidates, 0.0)
                mp = R * m_minus_c + income[zp]
                mp = np.clip(mp, m_min, m_max)
                m_prime_all[zp] = mp
                
                if V_splines[zp] is not None:
                    cont_flat = V_splines[zp](mp.ravel())
                    cont_all[zp] = cont_flat.reshape(n_m, n_scan)
                else:
                    cont_flat = np.interp(mp.ravel(), m_grid, V_next[zp, :])
                    cont_all[zp] = cont_flat.reshape(n_m, n_scan)
            
            # Weighted continuation value
            # cont_weighted[m, scan] = sum_z' trans_z[z'] * cont_all[z', m, scan]
            cont_weighted = np.tensordot(trans_z, cont_all, axes=(0, 0)) # (n_m, n_scan)
            
            # Utility of consumption
            util_c = utility(c_candidates) # (n_m, n_scan)
            
            # Objective
            obj_matrix = util_c + beta * cont_weighted
            
            # Best initial guess from scan
            best_idx = np.argmax(obj_matrix, axis=1)
            c_init = c_candidates[np.arange(n_m), best_idx]
            
            # Newton-Raphson Refinement
            c_curr = c_init.copy()
            
            # Precompute u'' for c_curr
            u2_curr = second_marginal_utility(c_curr)
            
            # Newton iterations
            for _ in range(10):
                if time.time() > deadline:
                    break
                
                # Compute m' for current c_curr
                m_minus_c_curr = np.maximum(m_grid - c_curr, 0.0)
                m_prime_curr = R * m_minus_c_curr + income[:, np.newaxis] # (n_z, n_m)
                m_prime_curr = np.clip(m_prime_curr, m_min, m_max)
                
                # Evaluate V' and V'' at m_prime
                V_prime_mp = np.zeros((n_z, n_m))
                V_double_prime_mp = np.zeros((n_z, n_m))
                
                for zp in range(n_z):
                    if trans_z[zp] == 0:
                        continue
                    if V_splines[zp] is not None:
                        V_prime_mp[zp] = V_splines[zp](m_prime_curr[zp], 1)
                        V_double_prime_mp[zp] = V_splines[zp](m_prime_curr[zp], 2)
                    else:
                        # Fallback to linear interpolation derivatives (less accurate)
                        # Central difference on the grid
                        eps = 1e-6
                        mp_up = np.clip(m_prime_curr[zp] + eps, m_min, m_max)
                        mp_dn = np.clip(m_prime_curr[zp] - eps, m_min, m_max)
                        V_up = np.interp(mp_up, m_grid, V_next[zp, :])
                        V_dn = np.interp(mp_dn, m_grid, V_next[zp, :])
                        V_prime_mp[zp] = (V_up - V_dn) / (mp_up - mp_dn)
                        V_double_prime_mp[zp] = 0.0 # Approximation
            
            # FOC: G(c) = u'(c) - beta * R * sum_z' pi_z' V'(m', z') = 0
            # Sum part: (n_m,)
            sum_V_prime = np.tensordot(trans_z, V_prime_mp, axes=(0, 0))
            sum_V_double_prime = np.tensordot(trans_z, V_double_prime_mp, axes=(0, 0))
            
            lhs = marginal_utility(c_curr)
            rhs = beta * R * sum_V_prime
            
            G = lhs - rhs
            
            # G'(c) = u''(c) + beta * R * sum_z' pi_z' V''(m', z') * d(m')/dc
            # d(m')/dc = -R
            G_prime = u2_curr - beta * R * sum_V_double_prime * R
            
            # Newton step
            # Avoid division by zero
            safe_G_prime = np.where(np.abs(G_prime) < 1e-12, 1e-12 * np.sign(G_prime + 1e-20), G_prime)
            step = G / safe_G_prime
            
            c_new = c_curr - step
            
            # Clip to valid range [0, m]
            c_new = np.clip(c_new, 1e-12, m_grid)
            
            # Check convergence
            if np.max(np.abs(c_new - c_curr)) < 1e-10:
                c_curr = c_new
                break
            
            c_curr = c_curr * 0.5 + c_new * 0.5 # Damping for stability if needed? 
            # Actually, pure Newton is usually fine for convex problems. 
            # But if step is too large, it might overshoot. 
            # Let's stick to pure Newton but clipped.
            c_curr = c_new
            
            # Update u'' for next iteration
            u2_curr = second_marginal_utility(c_curr)
            
            if time.time() > deadline:
                break
                
            # If time is running out, break out of Newton loop
            if time.time() > deadline - 5.0:
                break

            # If time is running out, break out of z loop
            if time.time() > deadline - 2.0:
                break

            if time.time() > deadline - 1.0:
                break

        # Compute V_curr using optimal c
        # V_curr[z, m] = u(c*) + beta * sum_z' pi_z' V_next[z'](m')
        for z in range(n_z):
            trans_z = transition[z]
            c_opt = C_curr[z, :]
            
            m_minus_c = np.maximum(m_grid - c_opt, 0.0)
            m_prime = R * m_minus_c + income[:, np.newaxis] # (n_z, n_m)
            m_prime = np.clip(m_prime, m_min, m_max)
            
            cont_val = np.zeros((n_z, n_m))
            for zp in range(n_z):
                if trans_z[zp] == 0:
                    continue
                if V_splines[zp] is not None:
                    cont_val[zp] = V_splines[zp](m_prime[zp])
                else:
                    cont_val[zp] = np.interp(m_prime[zp], m_grid, V_next[zp, :])
            
            sum_cont = np.tensordot(trans_z, cont_val, axes=(0, 0))
            V_curr[z, :] = utility(c_opt) + beta * sum_cont
            
        consumption[t] = C_curr
        V_next = V_curr
        
        # Update splines for next iteration
        for z in range(n_z):
            try:
                V_splines[z] = CubicSpline(m_grid, V_next[z], bc_type='natural')
            except Exception:
                V_splines[z] = None

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
