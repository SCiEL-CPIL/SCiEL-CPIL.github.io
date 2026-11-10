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
            return -gamma * np.power(c, -gamma - 1)
    
    # Backward induction on internal grid
    policy_int = np.zeros((horizon, num_states, n_internal))
    value_int = np.zeros((horizon, num_states, n_internal))
    
    # Terminal period
    t = horizon - 1
    for z in range(num_states):
        value_int[t, z, :] = utility(m_internal)
        policy_int[t, z, :] = m_internal
    
    # Build interpolation functions for V' at t+1 using log-space
    def build_vp_interp_log(zp, t_idx):
        """Build PCHIP interpolator for V' in log-space at time t_idx, state zp"""
        vp_vals = u_prime(policy_int[t_idx, zp, :])
        # Use log-space interpolation for smoother behavior across scales
        try:
            interp = PchipInterpolator(log_m_internal, np.log(vp_vals + 1e-30))
            return interp
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
        
        # Pre-build V' interpolators for t+1 in log-space
        vp_interps = []
        val_interps = []
        for zp in range(num_states):
            vp_interps.append(build_vp_interp_log(zp, t + 1))
            try:
                val_interps.append(PchipInterpolator(log_m_internal, value_int[t + 1, zp, :]))
            except Exception:
                val_interps.append(None)
        
        for z in range(num_states):
            probs = transition[z]
            
            # Newton-Raphson with bisection fallback on FOC: f(c) = u'(c) - beta*R*sum P(z')*V'(m') = 0
            c = m_internal.copy()  # initial guess
            c_low = np.full(n_internal, c_floor)
            c_high = m_internal.copy()
            
            max_iter = 40
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Evaluate FOC and its derivative at current c
                lhs = u_prime(c)
                rhs = np.zeros(n_internal)
                rhs_prime = np.zeros(n_internal)
                
                for zp in range(num_states):
                    if probs[zp] < 1e-15:
                        continue
                    m_next = R * (m_internal - c) + income[zp]
                    m_next_clipped = np.clip(m_next, m_internal[0], m_internal[-1])
                    
                    if vp_interps[zp] is not None:
                        log_vp = vp_interps[zp](np.log(m_next_clipped))
                        vp_vals = np.exp(log_vp)
                    else:
                        vp_vals = np.interp(m_next_clipped, m_internal, u_prime(policy_int[t+1, zp, :]))
                    rhs += probs[zp] * vp_vals
                    
                    # For derivative, we need V''(m_next) * dm_next/dc = V''(m_next) * (-R)
                    # Approximate V'' from the interpolator if possible
                    # For now, use a finite difference estimate
                    dm = 1e-6 * m_next_clipped
                    m_next_plus = np.clip(m_next_clipped + dm, m_internal[0], m_internal[-1])
                    if vp_interps[zp] is not None:
                        log_vp_plus = vp_interps[zp](np.log(m_next_plus))
                        vp_vals_plus = np.exp(log_vp_plus)
                    else:
                        vp_vals_plus = np.interp(m_next_plus, m_internal, u_prime(policy_int[t+1, zp, :]))
                    vpp_approx = (vp_vals_plus - vp_vals) / dm
                    rhs_prime += probs[zp] * vpp_approx * (-R)
                
                rhs = beta * R * rhs
                rhs_prime = beta * R * rhs_prime
                
                # f(c) = lhs - rhs; f'(c) = u''(c) - rhs_prime
                f_val = lhs - rhs
                f_prime = u_double_prime(c) - rhs_prime
                
                # Newton step with bisection safeguard
                newton_step = -f_val / np.where(np.abs(f_prime) > 1e-15, f_prime, 1e-15)
                c_newton = c + newton_step
                
                # Bisection update
                too_high = f_val > 0
                c_low = np.where(too_high, c, c_low)
                c_high = np.where(~too_high, c, c_high)
                c_bisect = (c_low + c_high) / 2.0
                
                # Use Newton if it stays in bounds, otherwise bisect
                c_new = np.where(
                    (c_newton >= c_floor) & (c_newton <= m_internal),
                    c_newton,
                    c_bisect
                )
                c = np.clip(c_new, c_floor, m_internal)
                
                if np.all(c_high - c_low < 1e-12 * m_internal):
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
                    v_next = val_interps[zp](np.log(m_next_clipped))
                else:
                    v_next = np.interp(m_next_clipped, m_internal, value_int[t + 1, zp, :])
                cont_val += probs[zp] * v_next
            
            value_int[t, z, :] = u_val + beta * cont_val
    
    # Interpolate policy back to the original resource_grid
    policy = np.zeros((horizon, num_states, len(resource_grid)))
    for t in range(horizon):
        for z in range(num_states):
            # Use PCHIP in log-space for smooth interpolation
            try:
                log_policy = np.log(policy_int[t, z, :] + 1e-30)
                interp = PchipInterpolator(log_m_internal, log_policy)
                policy[t, z, :] = np.exp(interp(np.log(resource_grid)))
            except Exception:
                policy[t, z, :] = np.interp(resource_grid, m_internal, policy_int[t, z, :])
            # Enforce constraints
            policy[t, z, :] = np.clip(policy[t, z, :], c_floor, resource_grid)
    
    # Ensure terminal condition is exact
    for z in range(num_states):
        policy[horizon - 1, z, :] = resource_grid
    
    return policy
