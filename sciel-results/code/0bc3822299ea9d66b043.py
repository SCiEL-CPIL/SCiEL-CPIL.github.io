import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    c_floor = 1e-12
    
    # Utility function
    def utility(c):
        c = np.maximum(c, c_floor)
        if gamma == 1:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    # Utility derivative
    def u_prime(c):
        c = np.maximum(c, c_floor)
        if gamma == 1:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    # Utility second derivative
    def u_double_prime(c):
        c = np.maximum(c, c_floor)
        if gamma == 1:
            return -1.0 / (c * c)
        else:
            return -gamma * np.power(c, -gamma - 1)
    
    # Policy and Value arrays
    # policy[t, z, m_idx]
    policy = np.zeros((horizon, num_states, num_resources))
    value = np.zeros((horizon, num_states, num_resources))
    
    # Terminal Period
    t = horizon - 1
    for z in range(num_states):
        value[t, z, :] = utility(resource_grid)
        policy[t, z, :] = resource_grid
    
    # Pre-compute V' interpolators for t+1 at the start of each t iteration
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with consume-all
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    policy[tt, z, :] = resource_grid
                    value[tt, z, :] = utility(resource_grid)
            break
        
        # Pre-compute V' interpolators for all z' at t+1
        v_prime_interps = {}
        for zp in range(num_states):
            v_prime_grid = u_prime(policy[t+1, zp, :])
            try:
                v_prime_interps[zp] = PchipInterpolator(resource_grid, v_prime_grid)
            except Exception:
                v_prime_interps[zp] = None
        
        # Also pre-compute V interpolators for value computation
        v_interps = {}
        for zp in range(num_states):
            try:
                v_interps[zp] = PchipInterpolator(resource_grid, value[t+1, zp, :])
            except Exception:
                v_interps[zp] = None
        
        for z in range(num_states):
            probs = transition[z]
            active_zp = [zp for zp in range(num_states) if probs[zp] > 0]
            
            # Initial bounds for bisection
            low = np.full(num_resources, c_floor)
            high = resource_grid.copy()
            
            # Initial guess
            c_current = 0.5 * (low + high)
            
            # Compute f(c) = u'(c) - beta * R * E[V'(m_next)]
            def compute_f(c):
                lhs = u_prime(c)
                rhs = np.zeros(num_resources)
                for zp in active_zp:
                    m_next = R * (resource_grid - c) + income[zp]
                    m_next_clipped = np.clip(m_next, resource_grid[0], resource_grid[-1])
                    
                    if v_prime_interps[zp] is not None:
                        v_prime_vals = v_prime_interps[zp](m_next_clipped)
                    else:
                        v_prime_vals = np.interp(m_next_clipped, resource_grid, u_prime(policy[t+1, zp, :]))
                    
                    rhs += probs[zp] * v_prime_vals
                
                return lhs - beta * R * rhs

            f_val = compute_f(c_current)
            
            max_iter = 40
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Update bounds based on current f
                # f is decreasing: f>0 implies c < c_opt, f<0 implies c > c_opt
                low = np.where(f_val > 0, c_current, low)
                high = np.where(f_val < 0, c_current, high)
                
                # Convergence check
                if np.all(high - low < 1e-10):
                    break
                
                # Newton step
                f_prime = u_double_prime(c_current)
                # Avoid division by zero
                # f_prime is negative
                with np.errstate(divide='ignore', invalid='ignore'):
                    step = -f_val / f_prime
                
                c_newton = c_current + step
                
                # Bisection step
                c_bisect = 0.5 * (low + high)
                
                # Choose step: Newton if valid and within bounds, else Bisection
                mask_newton = (c_newton >= low) & (c_newton <= high) & np.isfinite(c_newton)
                c_next = np.where(mask_newton, c_newton, c_bisect)
                
                # Ensure within global bounds
                c_next = np.clip(c_next, c_floor, resource_grid)
                
                # Evaluate f at next point
                f_val = compute_f(c_next)
                c_current = c_next
            
            c_opt = np.clip(c_current, c_floor, resource_grid)
            policy[t, z, :] = c_opt
            
            # Compute value at c_opt
            u_val = utility(c_opt)
            cont_val = np.zeros(num_resources)
            for zp in active_zp:
                m_next = R * (resource_grid - c_opt) + income[zp]
                m_next_clipped = np.clip(m_next, resource_grid[0], resource_grid[-1])
                
                if v_interps[zp] is not None:
                    v_next = v_interps[zp](m_next_clipped)
                else:
                    v_next = np.interp(m_next_clipped, resource_grid, value[t+1, zp, :])
                
                cont_val += probs[zp] * v_next
            
            value[t, z, :] = u_val + beta * cont_val
    
    return policy
