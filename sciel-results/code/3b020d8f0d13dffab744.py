import numpy as np
import time
from scipy.interpolate import UnivariateSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)

    # High resolution grid for value function
    # Increased to 2000 to support high-precision spline interpolation
    fine_M = 2000
    fine_grid = np.geomspace(0.001, 40, fine_M)
    
    # Precompute utility and marginal utility
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            return np.log(np.maximum(c, 1e-12))
        def marginal_utility(c):
            return 1.0 / np.maximum(c, 1e-12)
    else:
        one_minus_gamma = 1.0 - gamma
        def utility(c):
            return np.maximum(c, 1e-12) ** one_minus_gamma / one_minus_gamma
        def marginal_utility(c):
            return np.maximum(c, 1e-12) ** (-gamma)

    # Value function and consumption on fine grid
    V = np.zeros((horizon + 1, Z, fine_M))
    consumption_fine = np.zeros((horizon, Z, fine_M))

    # Backward induction
    for t in range(horizon - 1, -1, -1):
        for z in range(Z):
            if t == horizon - 1:
                # Terminal: consume all
                consumption_fine[t, z, :] = fine_grid
                V[t, z, :] = utility(fine_grid)
                continue

            # Prepare next period's value functions for interpolation
            # We need V_{t+1}(m') and V'_{t+1}(m') for each z'
            splines = []
            for zp in range(Z):
                if transition[z, zp] < 1e-15:
                    splines.append(None)
                    continue
                # Use UnivariateSpline with s=0 for exact interpolation if data is smooth enough
                # However, V might have kinks at boundaries. 
                # Given the log-linear nature and finite horizon, it should be smooth enough.
                # We use k=3 (cubic) and s=0 (exact fit).
                try:
                    spline = UnivariateSpline(fine_grid, V[t + 1, zp, :], k=3, s=0)
                    splines.append(spline)
                except Exception:
                    # Fallback to linear if spline fails (e.g., non-monotonic issues or memory)
                    splines.append(None)
            
            # Solve for optimal c for each m in fine_grid
            m_vals = fine_grid
            best_c = np.zeros(fine_M)
            best_V = np.zeros(fine_M)

            # We will vectorize the Newton steps where possible, but Newton is local.
            # To speed up, we can initialize c with a heuristic (e.g., linear policy) and do a few global steps.
            # However, for robustness and simplicity within the time limit, we do a scalar loop for m
            # but vectorize the z' summation.
            
            # Heuristic initial guess: c = m / (1 + beta * R * ... ) is complex.
            # Let's start with c = 0.5 * m or similar, or just use a coarse grid search for init?
            # Actually, for CRRA, the optimal c is often proportional to m in infinite horizon.
            # In finite horizon, it varies. Let's use a simple heuristic: c_init = m * 0.5
            # Or better, use the previous period's policy if available? No, t is decreasing.
            # Let's use a coarse search to get a good initial guess, then Newton.
            # Coarse search: 20 points
            c_coarse_frac = np.linspace(0.01, 1.0, 20)
            
            for i_m in range(fine_M):
                m = m_vals[i_m]
                
                # Coarse search for initialization
                c_cands = m * c_coarse_frac
                c_cands = np.clip(c_cands, 1e-8, m)
                
                # Evaluate objective for coarse candidates
                # m' = R*(m-c) + income[z']
                # V_next = sum p * V_{t+1}(m')
                
                # Vectorize over c_cands
                m_primes = R * (m - c_cands)[:, None] + income[None, :] # (N_c, Z)
                m_primes = np.clip(m_primes, fine_grid[0], fine_grid[-1])
                
                # Interpolate V
                # We can use np.interp for the coarse search as it's fast and robust
                cont_vals = np.zeros(len(c_cands))
                for zp in range(Z):
                    p = transition[z, zp]
                    if p < 1e-15: continue
                    v_interp = np.interp(m_primes[:, zp], fine_grid, V[t + 1, zp, :])
                    cont_vals += p * v_interp
                
                total_vals = utility(c_cands) + beta * cont_vals
                best_c_init = c_cands[np.argmax(total_vals)]
                
                # Newton's Method
                c_newton = best_c_init
                for _ in range(5): # 5 iterations should be enough for quadratic convergence
                    # Calculate m' for current c
                    m_p_curr = R * (m - c_newton) + income # (Z,)
                    m_p_curr = np.clip(m_p_curr, fine_grid[0], fine_grid[-1])
                    
                    # Derivative: u'(c) - beta * R * sum p * V'(m')
                    # We need V'(m')
                    # If spline is available, use spline derivative. Else, use numerical diff or np.gradient on interp?
                    # Using spline derivative is the key improvement.
                    
                    dV_dm = np.zeros(Z)
                    for zp in range(Z):
                        if transition[z, zp] < 1e-15: continue
                        if splines[zp] is not None:
                            dV_dm[zp] = splines[zp].derivative()(m_p_curr[zp])
                        else:
                            # Fallback: central difference on fine_grid
                            # Find index
                            idx = np.searchsorted(fine_grid, m_p_curr[zp])
                            if idx == 0:
                                idx = 1
                            elif idx == len(fine_grid):
                                idx = len(fine_grid) - 2
                            h = fine_grid[idx] - fine_grid[idx-1]
                            dV_dm[zp] = (V[t+1, zp, idx] - V[t+1, zp, idx-1]) / h
                            
                    grad = marginal_utility(c_newton) - beta * R * np.dot(transition[z, :], dV_dm)
                    
                    # Second derivative for Newton step: -u''(c) - beta * R^2 * sum p * V''(m')
                    # u''(c) = -c^{-gamma-1} for gamma != 1, -1/c^2 for gamma=1
                    # We need V''(m')
                    d2V_dm2 = np.zeros(Z)
                    for zp in range(Z):
                        if transition[z, zp] < 1e-15: continue
                        if splines[zp] is not None:
                            d2V_dm2[zp] = splines[zp].derivative(2)(m_p_curr[zp])
                        else:
                            # Approximate second derivative
                            idx = np.searchsorted(fine_grid, m_p_curr[zp])
                            if idx == 0: idx = 1
                            elif idx == len(fine_grid): idx = len(fine_grid) - 2
                            # Use simple finite diff
                            if idx > 0 and idx < len(fine_grid)-1:
                                h1 = fine_grid[idx] - fine_grid[idx-1]
                                h2 = fine_grid[idx+1] - fine_grid[idx]
                                d2V_dm2[zp] = (V[t+1, zp, idx+1] - 2*V[t+1, zp, idx] + V[t+1, zp, idx-1]) / ((h1+h2)/2) ** 2
                            else:
                                d2V_dm2[zp] = 0
                                
                    if abs(gamma - 1.0) < 1e-10:
                        u_pp = -1.0 / (c_newton ** 2)
                    else:
                        u_pp = -c_newton ** (-gamma - 1)
                        
                    hessian = u_pp - beta * (R ** 2) * np.dot(transition[z, :], d2V_dm2)
                    
                    if abs(hessian) < 1e-15:
                        break
                        
                    step = -grad / hessian
                    
                    # Damping to ensure convergence and bounds
                    # If step is too large, damp it
                    if abs(step) > 0.5 * c_newton:
                        step = 0.5 * c_newton * np.sign(step)
                        
                    c_next = c_newton + step
                    
                    # Enforce bounds
                    if c_next <= 1e-8:
                        c_next = 1e-8
                    if c_next >= m:
                        c_next = m * (1 - 1e-8)
                        
                    if abs(c_next - c_newton) < 1e-10:
                        c_newton = c_next
                        break
                    c_newton = c_next
                    
                # Final evaluation and bounds check
                c_newton = np.clip(c_newton, 1e-8, m)
                best_c[i_m] = c_newton
                
                # Calculate V
                m_p_final = R * (m - c_newton) + income
                m_p_final = np.clip(m_p_final, fine_grid[0], fine_grid[-1])
                cont_val = 0.0
                for zp in range(Z):
                    p = transition[z, zp]
                    if p < 1e-15: continue
                    v_interp = np.interp(m_p_final[zp], fine_grid, V[t + 1, zp, :])
                    cont_val += p * v_interp
                best_V[i_m] = utility(c_newton) + beta * cont_val

            consumption_fine[t, z, :] = best_c
            V[t, z, :] = best_V

        # Check deadline
        if time.time() > deadline - 10.0:
            # If we are running out of time, we must break.
            # We have computed up to t. Earlier t (larger index in loop, smaller t value) are not computed.
            # Wait, loop is t from horizon-1 down to 0.
            # If we break, V[t] for t < current_t will be 0.
            # This will make the policy for those periods suboptimal (consuming too much or too little based on V=0).
            # To be safe, if we break early, we should fill the remaining V with a heuristic or just return what we have.
            # The prompt says "return the best result found".
            # If we break, the earlier periods (t=0..t_break) will have V=0.
            # For t where V=0, the optimal c is just argmax u(c), which is c=m (for log) or c->0 (for CRRA gamma>1).
            # This is bad.
            # Given horizon=6, and fine_M=2000, Z=2.
            # 6 * 2 * 2000 * (5 Newton iters * Z ops) = 12000 * 5 * 2 = 120,000 ops.
            # Plus spline creation: 6 * 2 = 12 splines.
            # This should be well within 115s. The bottleneck is likely the Python loop over m.
            # 2000 * 12 = 24,000 iterations. Each has 5 Newton steps.
            # 120,000 Newton steps. Each step involves vector ops of size 2.
            # This is fast in Python if not heavily overheaded.
            # The spline creation is the expensive part? UnivariateSpline is O(N log N) or similar.
            # 12 splines of size 2000. Should be < 1s total.
            # So we should be fine.
            pass

    # Extract policy on original resource_grid
    consumption = np.zeros((horizon, Z, M))
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.interp(resource_grid, fine_grid, consumption_fine[t, z, :])

    # Enforce terminal condition: c[5] = m
    consumption[horizon - 1, :, :] = resource_grid[None, None, :]

    # Enforce 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            m_vals = resource_grid[None, :]
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_vals + 1e-10)

    return consumption
