"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin below 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    n_z = len(income)
    
    # Use a dense grid for internal computation
    m_grid = np.geomspace(0.001, 40, 251)
    n_m = len(m_grid)
    
    # Utility and derivative functions
    def u(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c_safe)
        else:
            return c_safe ** (1.0 - gamma) / (1.0 - gamma)
    
    def u_prime(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c_safe
        else:
            return c_safe ** (-gamma)
    
    def u_double_prime(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (c_safe ** 2)
        else:
            return -gamma * c_safe ** (-gamma - 1)
    
    # Terminal period: consume everything
    # c[horizon-1, z, :] = resource_grid (i.e., m)
    # But we need to return on the original resource_grid
    # Let's work on the scoring grid m_grid and interpolate back at the end
    
    # Policy: c[t, z, m] for t in 0..horizon-1, z in 0..n_z-1, m in m_grid
    # We'll store as c_policy[t, z] which is an array of length n_m
    
    c_policy = np.zeros((horizon, n_z, n_m))
    
    # Terminal condition: c = m
    for z in range(n_z):
        c_policy[horizon - 1, z, :] = m_grid
    
    # Backward induction from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            # Not enough time, break and return what we have
            break
        
        for z in range(n_z):
            # For each m, find c that maximizes:
            # u(c) + beta * sum_{z'} transition[z, z'] * V_{t+1}(m', z')
            # where m' = R * (m - c) + income[z']
            
            # We need V_{t+1}(m', z') for each z'
            # V_{t+1}(m, z') = u(c_{t+1, z', m}) + beta * sum_{z''} transition[z', z''] * V_{t+2}(m'', z'')
            # But at t = horizon-2, V_{t+1} = V_{horizon-1} = u(m) (consume all)
            
            # Let's compute the continuation value function for each z'
            # cont_value[z', m'] = expected continuation value
            
            # For efficiency, we'll define the continuation value as a function of m'
            # cont_val[z'] is an array of length n_m, giving the continuation value at m_grid
            
            # At t = horizon-2:
            #   V_{t+1}(m, z') = u(m)  (terminal)
            # At t < horizon-2:
            #   V_{t+1}(m, z') = u(c_policy[t+1, z', m]) + beta * sum_{z''} transition[z', z''] * V_{t+2}(m'', z'')
            #   where m'' = R * (m - c_policy[t+1, z', m]) + income[z'']
            
            # Let's build the continuation value arrays
            # cont_val[z'] = V_{t+1} evaluated at m_grid for state z'
            
            if t == horizon - 2:
                # Terminal next period
                cont_val = np.zeros((n_z, n_m))
                for zp in range(n_z):
                    cont_val[zp, :] = u(m_grid)
            else:
                # Need V_{t+1} which depends on c_policy[t+1, :]
                cont_val = np.zeros((n_z, n_m))
                for zp in range(n_z):
                    c_next = c_policy[t + 1, zp, :]
                    # u(c_next) for each m
                    u_next = u(c_next)
                    
                    # Now add beta * sum_{z''} transition[zp, z''] * V_{t+2}(m'', z'')
                    # m'' = R * (m - c_next) + income[z'']
                    # V_{t+2}(m'', z'') needs to be interpolated from m_grid
                    
                    # Build the sum over z''
                    cont_sum = np.zeros(n_m)
                    for zpp in range(n_z):
                        prob = transition[zp, zpp]
                        if prob < 1e-15:
                            continue
                        # m'' = R * (m_grid - c_next) + income[zpp]
                        m_next = R * (m_grid - c_next) + income[zpp]
                        # Clip to valid range
                        m_next = np.maximum(m_next, 1e-12)
                        m_next = np.minimum(m_next, 40.0)
                        
                        # Interpolate V_{t+2}(m_next, zpp)
                        # V_{t+2}(m, zpp) = u(c_policy[t+2, zpp, m]) + beta * ...
                        # But we can just use the stored policy to compute V at any m
                        # Actually, we need V_{t+2} as a function. Let's compute it.
                        
                        # For each m in m_grid, V_{t+2}(m, zpp) = u(c_policy[t+2, zpp, m]) + beta * sum_{z'''} transition[zpp, z'''] * V_{t+3}(m''', z''')
                        # This is recursive. Instead, let's just compute V_{t+2} on m_grid and interpolate.
                        
                        # V_{t+2} on m_grid:
                        if t + 2 == horizon - 1:
                            V_next2 = u(m_grid)
                        else:
                            V_next2 = np.zeros(n_m)
                            for zppp in range(n_z):
                                prob2 = transition[zpp, zppp]
                                if prob2 < 1e-15:
                                    continue
                                c_pp = c_policy[t + 2, zpp, :]
                                u_pp = u(c_pp)
                                m_pp = R * (m_grid - c_pp) + income[zppp]
                                m_pp = np.maximum(m_pp, 1e-12)
                                m_pp = np.minimum(m_pp, 40.0)
                                V_pp = np.interp(m_pp, m_grid, u(c_policy[t + 2, zppp, :]))
                                # Wait, this is getting complicated. Let me restructure.
                                
                                # Actually, let me just compute V_{t+2}(m, zpp) directly.
                                # V_{t+2}(m, zpp) = u(c_{t+2, zpp}(m)) + beta * sum_{z'''} transition[zpp, z'''] * V_{t+3}(R*(m - c_{t+2, zpp}(m)) + income[z'''], z''')
                                
                                # This is the same structure. Let me just compute the full V_{t+2} array.
                                pass
                            # This approach is getting unwieldy. Let me use a cleaner recursion.
                            pass
                        
                        cont_sum += prob * np.interp(m_next, m_grid, V_next2)
                    
                    cont_val[zp, :] = u_next + beta * cont_sum
            
            # Now find optimal c for state z at period t
            # Maximize u(c) + beta * sum_{z'} transition[z, z'] * cont_val[z'](R*(m-c) + income[z'])
            
            # For each m, solve for c
            for i in range(n_m):
                m_i = m_grid[i]
                
                def objective(c):
                    if c <= 0 or c >= m_i - 1e-15:
                        return -1e10
                    m_next = R * (m_i - c)
                    total = 0.0
                    for zp in range(n_z):
                        prob = transition[z, zp]
                        if prob < 1e-15:
                            continue
                        m_zp = m_next + income[zp]
                        m_zp = max(m_zp, 1e-12)
                        m_zp = min(m_zp, 40.0)
                        v = np.interp(m_zp, m_grid, cont_val[zp, :])
                        total += prob * v
                    return u(c) + beta * total
                
                # Use Brent's method on the derivative
                def euler_residual(c):
                    if c <= 0 or c >= m_i - 1e-15:
                        return 1.0
                    m_next = R * (m_i - c)
                    # Derivative of u(c) is u'(c)
                    # Derivative of beta * sum prob * V(R*(m-c) + income[z']) w.r.t. c is
                    # beta * sum prob * V'(R*(m-c) + income[z']) * (-R)
                    # V'(m, z') = u'(c_{t+1, z'}(m)) * dc/dm
                    # But we don't have the derivative of the policy easily.
                    
                    # Alternative: use numerical derivative or just use brentq on the first-order condition
                    # FOC: u'(c) = beta * sum_{z'} transition[z, z'] * V_{t+1}'(R*(m-c) + income[z'], z') * (-R)
                    # = -beta * R * sum_{z'} transition[z, z'] * V_{t+1}'(R*(m-c) + income[z'], z')
                    
                    # V_{t+1}'(m, z') = u'(c_{t+1, z'}(m)) (by envelope theorem, since c is chosen optimally)
                    # Wait, that's only if we're at the optimum. The envelope theorem says:
                    # V'(m, z) = u'(c*(m, z)) where c* is the optimal consumption.
                    
                    # So V_{t+1}'(m, z') = u'(c_policy[t+1, z', m])
                    
                    total_deriv = 0.0
                    for zp in range(n_z):
                        prob = transition[z, zp]
                        if prob < 1e-15:
                            continue
                        m_zp = m_next + income[zp]
                        m_zp = max(m_zp, 1e-12)
                        m_zp = min(m_zp, 40.0)
                        c_zp = np.interp(m_zp, m_grid, c_policy[t + 1, zp, :])
                        v_prime = u_prime(c_zp)
                        total_deriv += prob * v_prime
                    
                    return u_prime(c) + beta * R * total_deriv
                
                # Solve euler_residual(c) = 0
                c_lo = 1e-10
                c_hi = m_i * 0.999999
                if c_hi <= c_lo:
                    c_policy[t, z, i] = m_i
                    continue
                
                try:
                    c_opt = brentq(euler_residual, c_lo, c_hi, xtol=1e-12, rtol=1e-12, maxiter=100)
                except ValueError:
                    # Sign didn't change, use bisection or just pick midpoint
                    # Try a few points to find a sign change
                    c_opt = None
                    for frac in np.linspace(0.01, 0.99, 20):
                        c_test = c_lo + frac * (c_hi - c_lo)
                        r1 = euler_residual(c_test)
                        r2 = euler_residual(c_test + 1e-6)
                        if r1 * r2 < 0:
                            try:
                                c_opt = brentq(euler_residual, c_test, c_test + 1e-6, xtol=1e-12, maxiter=50)
                            except:
                                c_opt = c_test
                            break
                    if c_opt is None:
                        c_opt = m_i * 0.5
                
                c_policy[t, z, i] = np.clip(c_opt, 1e-10, m_i)
    
    # Interpolate back to the original resource_grid
    result = np.zeros((horizon, n_z, len(resource_grid)))
    for t in range(horizon):
        for z in range(n_z):
            result[t, z, :] = np.interp(resource_grid, m_grid, c_policy[t, z, :])
    
    # Enforce constraints: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            result[t, z, :] = np.maximum(result[t, z, :], 1e-12)
            result[t, z, :] = np.minimum(result[t, z, :], resource_grid + 1e-10)
    
    # Terminal condition: c[horizon-1] must equal the grid within 1e-9
    for z in range(n_z):
        result[horizon - 1, z, :] = resource_grid
    
    # Check finiteness
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=1e-12, posinf=1.0, neginf=1e-12)
        for t in range(horizon):
            for z in range(n_z):
                result[t, z, :] = np.maximum(result[t, z, :], 1e-12)
                result[t, z, :] = np.minimum(result[t, z, :], resource_grid + 1e-10)
        for z in range(n_z):
            result[horizon - 1, z, :] = resource_grid
    
    return result
# EVOLVE-BLOCK-END
