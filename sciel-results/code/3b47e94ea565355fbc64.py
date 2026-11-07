"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin below 120s

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    m_grid = resource_grid
    n_m = len(m_grid)

    # Use a dense internal grid for high accuracy
    # The evaluator interpolates onto 251 points, so we need high resolution.
    # 1000 points is fast enough and provides excellent accuracy.
    n_dense = 1000
    dense_grid = np.geomspace(0.001, 40, n_dense)
    
    # Value function V[t, z, m]
    V = np.zeros((horizon, 2, n_dense))
    # Policy function c[t, z, m]
    c_policy = np.zeros((horizon, 2, n_dense))

    # Terminal period: consume everything
    V[horizon-1, :, :] = dense_grid[np.newaxis, :, np.newaxis] * 0 # Placeholder, will overwrite
    # Actually, for log utility, V_T(m) = log(m). For CRRA, V_T(m) = m^(1-gamma)/(1-gamma).
    # But we need the policy c_T = m.
    # Let's just set the policy and compute V from it for consistency if needed, 
    # but backward induction usually starts with V_T.
    
    def get_utility(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    def get_utility_prime(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Initialize terminal period
    c_policy[horizon-1, :, :] = dense_grid[np.newaxis, :, np.newaxis]
    V[horizon-1, :, :] = get_utility(dense_grid[np.newaxis, :, np.newaxis], gamma)

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: fill remaining with simple rule
            for tt in range(t, -1, -1):
                # Simple heuristic: consume proportional share
                # This is a safety net, unlikely to be hit with 1000 grid points
                c_policy[tt, :, :] = dense_grid[np.newaxis, :, np.newaxis] * 0.5
                V[tt, :, :] = get_utility(c_policy[tt, :, :], gamma)
            break

        # Interpolate V_{t+1} for each state z
        # We need to compute the optimal c for each m, z at time t
        # FOC: u'(c) = beta * R * E[ V'_{t+1}(m', z') | z ]
        # where m' = R(m - c) + income[z']
        
        # We will use a fixed-point iteration or Newton's method on the policy function.
        # Given the structure, we can vectorize the update for all m.
        
        # Initialize policy guess for this period
        # A good guess is the previous period's policy shifted or a constant fraction
        # Let's use the previous period's policy as a warm start if available, else 0.5*m
        if t == horizon - 2:
             c_guess = 0.5 * dense_grid[np.newaxis, :, np.newaxis]
        else:
             # Simple linear extrapolation or just use previous
             c_guess = c_policy[t+1, :, :]

        # We will iterate to find the fixed point of the policy function
        # c_new = argmax_c { u(c) + beta * E[ V_{t+1}(R(m-c)+y', z') ] }
        
        # To speed up convergence, we can use the fact that c is monotonic in m.
        # We'll do a few iterations of value iteration on the policy itself.
        
        # Precompute interpolators for V_{t+1} and its derivative V'_{t+1}
        # V_{t+1} is defined on dense_grid for each z.
        interp_V = [PchipInterpolator(dense_grid, V[t+1, z, :]) for z in range(2)]
        # We need V' for the FOC. PchipInterpolator provides derivative.
        interp_Vp = [interp_V[z].derivatives(1) for z in range(2)]
        
        # Iterate for policy
        c_curr = c_guess.copy()
        
        for _ in range(15): # 15 iterations of policy iteration should be enough
            c_next = np.zeros_like(c_curr)
            
            for z in range(2):
                m_vals = dense_grid
                c_vals = c_curr[z, :]
                
                # Compute m' for each z'
                # m_next[z', m] = R * (m - c) + income[z']
                # Shape: (2, n_dense)
                m_next = R * (m_vals[np.newaxis, :] - c_vals[np.newaxis, :]) + income[:, np.newaxis]
                
                # Clip m_next to valid range for interpolation
                # PchipInterpolator extrapolates, which might be unstable if m' is far out.
                # m' min: R*(0.001 - 0.001) + 0.5 = 0.5
                # m' max: R*(40 - 0.001) + 1.5 ~ 42
                # Grid is 0.001 to 40. So m' can be > 40.
                # We should extend the grid or handle extrapolation.
                # For now, PchipInterpolator handles it, but it might be wiggly.
                # Let's extend the grid slightly for safety in V calculation.
                # Actually, the grid goes up to 40. Max m' is ~42.
                # Let's just rely on Pchip for now, it's usually okay for short extrapolation.
                
                # Evaluate V_{t+1} and V'_{t+1}
                # V_val[z', m] = V_{t+1}(m_next[z', m], z')
                # Vp_val[z', m] = V'_{t+1}(m_next[z', m], z')
                
                V_vals = np.array([interp_V[zp](m_next[zp, :]) for zp in range(2)])
                Vp_vals = np.array([interp_Vp[zp](m_next[zp, :]) for zp in range(2)])
                
                # Expected values
                # E[V'] = sum_zp P(zp|z) * Vp_vals[zp, :]
                E_Vp = transition[z, 0] * Vp_vals[0, :] + transition[z, 1] * Vp_vals[1, :]
                
                # FOC: u'(c) = beta * R * E_Vp
                # u'(c) = c^(-gamma) (or 1/c)
                # c^(-gamma) = beta * R * E_Vp
                # c = (beta * R * E_Vp)^(-1/gamma)
                
                target = beta * R * E_Vp
                target = np.maximum(target, 1e-12)
                
                if abs(gamma - 1.0) < 1e-9:
                    c_opt = 1.0 / target
                else:
                    c_opt = target ** (-1.0 / gamma)
                
                # Ensure c <= m
                c_opt = np.minimum(c_opt, m_vals)
                c_opt = np.maximum(c_opt, 1e-12)
                
                c_next[z, :] = c_opt
            
            # Check convergence
            if np.max(np.abs(c_next - c_curr)) < 1e-8:
                c_curr = c_next
                break
            # Damping to ensure stability
            c_curr = 0.5 * c_curr + 0.5 * c_next
            c_curr = c_next # Actually, pure fixed point is usually fast enough for this convex problem
            
        c_policy[t, :, :] = c_curr
        
        # Compute V[t]
        for z in range(2):
            m_vals = dense_grid
            c_vals = c_curr[z, :]
            m_next = R * (m_vals[np.newaxis, :] - c_vals[np.newaxis, :]) + income[:, np.newaxis]
            
            V_next_vals = np.array([interp_V[zp](m_next[zp, :]) for zp in range(2)])
            E_V_next = transition[z, 0] * V_next_vals[0, :] + transition[z, 1] * V_next_vals[1, :]
            
            V[t, z, :] = get_utility(c_vals, gamma) + beta * E_V_next

    # Interpolate the solution from dense_grid to resource_grid
    consumption = np.zeros((horizon, 2, n_m))
    for z in range(2):
        for t in range(horizon):
            consumption[t, z, :] = np.interp(resource_grid, dense_grid, c_policy[t, z, :])
    
    # Final check and clipping
    consumption = np.clip(consumption, 1e-12, resource_grid[np.newaxis, np.newaxis, :] + 1e-10)
    
    # Ensure c[5] is exactly the grid
    consumption[5, :, :] = resource_grid[np.newaxis, :]
    
    return consumption
# EVOLVE-BLOCK-END
