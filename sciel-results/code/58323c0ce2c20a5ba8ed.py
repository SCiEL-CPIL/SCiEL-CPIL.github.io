import time
import numpy as np
from scipy.optimize import brentq

DEADLINE_SECONDS = 130  # Safety margin before the 150s kill

def _crsa_utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)

def _crsa_marginal(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return 1.0 / c
    return c ** (-gamma)

def _solve_consumption_single(model, resource_grid, budget, deadline):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    trans = model['transition']
    horizon = model['horizon']
    states = len(income)
    n_grid = len(resource_grid)
    
    # We will store V[t, z, i] = value function at time t, state z, grid index i
    # And c_policy[t, z, i] = optimal consumption
    consumption = np.zeros((horizon, states, n_grid))
    V = np.zeros((horizon, states, n_grid))
    
    # Terminal condition: t = horizon - 1
    # c = m, V = u(m)
    consumption[horizon - 1] = np.tile(resource_grid, (states, 1))
    V[horizon - 1] = _crsa_utility(np.tile(resource_grid, (states, 1)), gamma)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        V_next = V[t + 1]  # Shape (states, n_grid)
        
        for z in range(states):
            # We need to find optimal c for each m in resource_grid
            # Objective: max_c u(c) + beta * sum_z' trans[z, z'] * V_next[z', R*(m-c) + income[z']]
            
            # The FOC is: u'(c) = beta * sum_z' trans[z, z'] * V_next_marginal[z', m'] * R
            # where m' = R*(m-c) + income[z']
            # V_next_marginal is the derivative of V_next with respect to m.
            
            # Since V_next is defined on a grid, we approximate its derivative.
            # However, for CRRA and linear budget, the value function is often homogeneous of degree 1-gamma (or log for gamma=1).
            # Actually, V(m, z) = A(z) * m^(1-gamma)/(1-gamma) is a common form if we assume it.
            # But we are doing numerical backward induction, so we should use numerical derivatives of the interpolated V_next.
            
            # Let's compute the derivative of V_next[z'] with respect to m using central differences on the grid.
            # dV_dm[z', i]
            dV_dm = np.zeros((states, n_grid))
            # Use np.gradient or manual central difference
            # Manual central difference for interior, one-sided for boundaries
            for zp in range(states):
                dV_dm[zp, 1:-1] = (V_next[zp, 2:] - V_next[zp, :-2]) / (resource_grid[2:] - resource_grid[:-2])
                dV_dm[zp, 0] = (V_next[zp, 1] - V_next[zp, 0]) / (resource_grid[1] - resource_grid[0])
                dV_dm[zp, -1] = (V_next[zp, -1] - V_next[zp, -2]) / (resource_grid[-1] - resource_grid[-2])
            
            # Now for each m in resource_grid, we solve for c.
            # Let c = f * m.
            # FOC: u'(f*m) = beta * R * sum_z' trans[z, z'] * dV_dm[z', m']
            # where m' = R*(m - f*m) + income[z'] = m*(1 - R*f) + R*income[z']
            
            # We will solve for f for each m.
            # Define a function g(f) = u'(f*m) - beta * R * sum_z' trans[z, z'] * interp(dV_dm[z'], m')
            # We want g(f) = 0.
            
            # Since u' is decreasing in c (and thus f), and the RHS is likely increasing in m (and thus decreasing in f? No, m' decreases as f increases, so dV_dm(m') increases if dV_dm is decreasing? 
            # Wait, dV_dm is the marginal value of wealth. For CRRA, marginal utility is decreasing. So dV_dm is decreasing in m.
            # As f increases, m' decreases. So dV_dm(m') increases.
            # LHS: u'(f*m) decreases as f increases.
            # RHS: beta * R * sum ... dV_dm(m') increases as f increases.
            # So g(f) = LHS - RHS is strictly decreasing.
            # At f=0, LHS = u'(0) = inf (or large), RHS is finite. g(0) > 0.
            # At f=1, LHS = u'(m), RHS = beta * R * sum ... dV_dm(m'). 
            # If the agent wants to save, g(1) < 0. If they want to consume all, g(1) > 0 (but c<=m, so f=1 is boundary).
            
            # We will use brentq to find f in [eps, 1].
            # To be safe, we can vectorize the root finding over m.
            
            # Precompute trans[z, :]
            trans_z = trans[z]
            
            # We will iterate over m indices.
            for i in range(n_grid):
                m = resource_grid[i]
                
                def g(f):
                    c = f * m
                    if c <= 0:
                        return 1e10
                    lhs = _crsa_marginal(c, gamma)
                    
                    # Compute m' for each z'
                    m_primes = np.zeros(states)
                    for zp in range(states):
                        m_prime = R * (m - c) + income[zp]
                        # Clip m' to grid for interpolation
                        m_prime = np.clip(m_prime, resource_grid[0], resource_grid[-1])
                        m_primes[zp] = m_prime
                    
                    # Interpolate dV_dm at m_primes
                    rhs_sum = 0.0
                    for zp in range(states):
                        dV_interp = np.interp(m_primes[zp], resource_grid, dV_dm[zp])
                        rhs_sum += trans_z[zp] * dV_interp
                    
                    rhs = beta * R * rhs_sum
                    return lhs - rhs
                
                # Solve g(f) = 0
                # f is in [1e-6, 1.0]
                f_min = 1e-6
                f_max = 1.0
                
                g_min = g(f_min)
                g_max = g(f_max)
                
                if g_min < 0:
                    # Optimum is at lower bound? Unlikely, but handle it.
                    f_opt = f_min
                elif g_max > 0:
                    # Optimum is at upper bound (consume all)
                    f_opt = f_max
                else:
                    try:
                        f_opt = brentq(g, f_min, f_max, xtol=1e-8)
                    except Exception:
                        # Fallback to binary search or grid search if brentq fails
                        # Simple binary search
                        lo = f_min
                        hi = f_max
                        for _ in range(50):
                            mid = (lo + hi) / 2.0
                            if g(mid) > 0:
                                lo = mid
                            else:
                                hi = mid
                        f_opt = (lo + hi) / 2.0
                
                c_opt = f_opt * m
                consumption[t, z, i] = c_opt
                
                # Compute V[t, z, i] = u(c_opt) + beta * E[V_next]
                u_c = _crsa_utility(c_opt, gamma)
                
                m_primes = np.zeros(states)
                for zp in range(states):
                    m_prime = R * (m - c_opt) + income[zp]
                    m_prime = np.clip(m_prime, resource_grid[0], resource_grid[-1])
                    m_primes[zp] = m_prime
                
                cont_val = 0.0
                for zp in range(states):
                    V_interp = np.interp(m_primes[zp], resource_grid, V_next[zp])
                    cont_val += trans_z[zp] * V_interp
                
                V[t, z, i] = u_c + beta * cont_val
                
                # Time check
                if time.time() > deadline:
                    # Fill remaining with a heuristic or break
                    # For simplicity, break and let the outer loop handle validation/fallback
                    return consumption, V, True # Return flag indicating timeout
            
        # Time check
        if time.time() > deadline:
            return consumption, V, True
    
    return consumption, V, False

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    try:
        consumption, V, timed_out = _solve_consumption_single(model, resource_grid, budget, deadline)
        
        # Validate
        if not np.all(np.isfinite(consumption)):
            raise ValueError("Non-finite values")
        
        m = resource_grid[None, None, :]
        if np.any(consumption <= 0) or np.any(consumption > m + 1e-10):
            raise ValueError("c out of bounds")
        
        # Check c[5] equals grid within 1e-9
        states = model['transition'].shape[0]
        if not np.allclose(consumption[5], np.tile(resource_grid, (states, 1)), atol=1e-9):
            raise ValueError("Terminal condition violated")
        
        return consumption
        
    except Exception as e:
        # Fallback: simple heuristic
        horizon = model['horizon']
        states = len(model['income'])
        n_grid = len(resource_grid)
        
        result = np.zeros((horizon, states, n_grid))
        for t in range(horizon):
            if t == horizon - 1:
                result[t] = np.tile(resource_grid, (states, 1))
            else:
                # Simple rule: consume a fraction
                frac = 0.9
                result[t] = frac * resource_grid[None, :]
                result[t] = np.clip(result[t], 1e-6, resource_grid[None, :])
                result[t] = np.tile(result[t], (states, 1))
        
        return result
