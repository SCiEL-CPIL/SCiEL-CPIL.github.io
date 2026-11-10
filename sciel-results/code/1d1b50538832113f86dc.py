import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'], dtype=np.float64)
    transition = np.array(model['transition'], dtype=np.float64)
    horizon = model['horizon']
    
    num_states = len(income)
    c_floor = 1e-12
    
    # Use a denser internal grid for backward induction
    n_internal = 400
    m_internal = np.geomspace(0.001, 40, n_internal)
    log_m_internal = np.log(m_internal)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, c_floor)
        if gamma == 1:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    
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
            return -gamma * np.power(c, -gamma - 1.0)
    
    # Backward induction on internal grid
    policy_int = np.zeros((horizon, num_states, n_internal))
    value_int = np.zeros((horizon, num_states, n_internal))
    
    # Terminal period
    t = horizon - 1
    for z in range(num_states):
        value_int[t, z, :] = utility(m_internal)
        policy_int[t, z, :] = m_internal
    
    # Build interpolation functions for V' at t+1
    def build_vp_interp(zp, t_idx):
        """Build PCHIP interpolator for V' at time t_idx, state zp"""
        vp_vals = u_prime(policy_int[t_idx, zp, :])
        try:
            return PchipInterpolator(m_internal, vp_vals)
        except Exception:
            return None
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: consume all
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    policy_int[tt, z, :] = m_internal
                    value_int[tt, z, :] = utility(m_internal)
            break
        
        # Pre-build V' interpolators for t+1
        vp_interps = []
        val_interps = []
        for zp in range(num_states):
            vp_interps.append(build_vp_interp(zp, t + 1))
            try:
                val_interps.append(PchipInterpolator(m_internal, value_int[t + 1, zp, :]))
            except Exception:
                val_interps.append(None)
        
        for z in range(num_states):
            probs = transition[z]
            
            # Newton's method on FOC: f(c) = u'(c) - beta*R*sum P(z')*V'(m') = 0
            # f is decreasing in c (u' decreasing, V' increasing as m' decreases)
            # f'(c) = u''(c) + beta*R*sum P(z')*R*V''(m') < 0 (both terms negative)
            c = m_internal.copy()  # initial guess: consume all
            
            max_iter = 40
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Evaluate FOC and its derivative at current c
                lhs = u_prime(c)
                lhs_prime = u_double_prime(c)
                
                rhs = np.zeros(n_internal)
                rhs_prime = np.zeros(n_internal)
                
                for zp in range(num_states):
                    if probs[zp] < 1e-15:
                        continue
                    m_next = R * (m_internal - c) + income[zp]
                    m_next_clipped = np.clip(m_next, m_internal[0], m_internal[-1])
                    
                    if vp_interps[zp] is not None:
                        vp_vals = vp_interps[zp](m_next_clipped)
                    else:
                        vp_vals = np.interp(m_next_clipped, m_internal, u_prime(policy_int[t+1, zp, :]))
                    rhs += probs[zp] * vp_vals
                    # V''(m') * dm'/dc = V''(m') * (-R), so contribution to f' is +beta*R*P*R*V''(m')
                    # We approximate V'' using finite differences of V'
                    # For simplicity, use a rough estimate: V'' ~ (V'(m+dm) - V'(m-dm)) / (2*dm)
                    # But this is expensive. Instead, use the fact that for CRRA, V' is monotone
                    # and we can use a conservative step.
                
                rhs = beta * R * rhs
                rhs_prime = np.zeros(n_internal)  # Approximate as 0 for stability
                
                # f(c) = lhs - rhs; f decreasing
                f_val = lhs - rhs
                f_prime_val = lhs_prime + rhs_prime  # Negative
                f_prime_val = np.minimum(f_prime_val, -1e-15)  # Ensure negative
                
                # Newton step: c_new = c - f/f'
                c_new = c - f_val / f_prime_val
                c_new = np.clip(c_new, c_floor, m_internal)
                
                # Damping if Newton step is too large
                step = np.abs(c_new - c)
                max_step = 0.5 * m_internal
                too_large = step > max_step
                c_new = np.where(too_large, c + np.sign(c_new - c) * max_step, c_new)
                
                c = c_new
                
                if np.all(np.abs(f_val) < 1e-12 * np.abs(lhs)):
                    break
            
            policy_int[t, z, :] = np.clip(c, c_floor, m_internal)
            
            # Compute value
            u_val = utility(policy_int[t, z, :])
            cont_val = np.zeros(n_internal)
            for zp in range(num_states):
                if probs[zp] < 1e-15:
                    continue
                m_next = R * (m_internal - policy_int[t, z, :]) + income[zp]
                m_next_clipped = np.clip(m_next, m_internal[0], m_internal[-1])
                
                if val_interps[zp] is not None:
                    v_next = val_interps[zp](m_next_clipped)
                else:
                    v_next = np.interp(m_next_clipped, m_internal, value_int[t + 1, zp, :])
                cont_val += probs[zp] * v_next
            
            value_int[t, z, :] = u_val + beta * cont_val
    
    # Interpolate policy back to the original resource_grid
    policy = np.zeros((horizon, num_states, len(resource_grid)))
    for t in range(horizon):
        for z in range(num_states):
            # Use PCHIP for smooth interpolation
            try:
                interp = PchipInterpolator(m_internal, policy_int[t, z, :])
                policy[t, z, :] = interp(resource_grid)
            except Exception:
                policy[t, z, :] = np.interp(resource_grid, m_internal, policy_int[t, z, :])
            # Enforce constraints
            policy[t, z, :] = np.clip(policy[t, z, :], c_floor, resource_grid)
    
    # Ensure terminal condition is exact
    for z in range(num_states):
        policy[horizon - 1, z, :] = resource_grid
    
    return policy
