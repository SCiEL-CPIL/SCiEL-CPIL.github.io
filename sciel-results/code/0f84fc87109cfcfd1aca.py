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
    
    # Use a denser internal grid to improve interpolation accuracy for V and V'
    # 600 points on [0.001, 40] is much better than 300 for PCHIP
    n_internal = 600
    m_internal = np.geomspace(0.001, 40, n_internal)
    
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

    # Backward induction storage
    policy_int = np.zeros((horizon, num_states, n_internal))
    value_int = np.zeros((horizon, num_states, n_internal))
    
    # Terminal period: consume all
    t = horizon - 1
    for z in range(num_states):
        value_int[t, z, :] = utility(m_internal)
        policy_int[t, z, :] = m_internal
    
    # Pre-build interpolators for terminal period
    val_interps = [None] * num_states
    vp_interps = [None] * num_states
    vpp_interps = [None] * num_states
    
    for z in range(num_states):
        try:
            val_interps[z] = PchipInterpolator(m_internal, value_int[t, z, :])
            vp_interps[z] = PchipInterpolator(m_internal, u_prime(policy_int[t, z, :]))
            # V'' is the derivative of V'. We can approximate it by differentiating the PCHIP of V'
            # Or simpler: V''(m) = -u''(c) / (u'(c) + beta*R*sum P V'(m')) ... no, that's complex.
            # Let's just use numerical derivative of the PCHIP V' or a second PCHIP on V'' values?
            # Actually, for Newton's method on F(c) = u'(c) - beta*R*E[V'(m')], we need F'(c).
            # F'(c) = u''(c) - beta*R*E[V''(m')] * (-R) = u''(c) + beta*R^2*E[V''(m')]
            # We need V''(m'). We can get this by differentiating the PCHIP of V'.
            vpp_interps[z] = vp_interps[z].derivatives(1)
        except Exception:
            pass

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: consume all
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    policy_int[tt, z, :] = m_internal
                    value_int[tt, z, :] = utility(m_internal)
            break
        
        # Update interpolators for t+1
        # val_interps, vp_interps, vpp_interps currently hold t+1 data
        
        for z in range(num_states):
            probs = transition[z]
            
            # Initial guess: proportional to m
            c = m_internal * 0.5
            c = np.clip(c, c_floor, m_internal)
            
            max_iter = 20
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Calculate F(c) = u'(c) - beta*R*E[V'(m')]
                lhs = u_prime(c)
                rhs = np.zeros(n_internal)
                rhs_deriv = np.zeros(n_internal) # For Newton: d(rhs)/dc = -beta*R^2*E[V''(m')]
                
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
                    
                    # For Newton's method, we need the derivative of the RHS with respect to c
                    # d/dc (V'(m')) = V''(m') * dm'/dc = V''(m') * (-R)
                    if vpp_interps[zp] is not None:
                        vpp_vals = vpp_interps[zp](m_next_clipped)
                    else:
                        # Fallback: numerical differentiation or just 0 (gradient descent)
                        vpp_vals = np.zeros(n_internal)
                        
                    rhs_deriv -= probs[zp] * vpp_vals * R
                
                rhs = beta * R * rhs
                # F(c) = u'(c) - beta*R*E[V'(m')]
                F_val = lhs - rhs
                
                # F'(c) = u''(c) - d/dc(beta*R*E[V'(m')])
                # d/dc(beta*R*E[V'(m')]) = beta*R * E[V''(m') * (-R)] = -beta*R^2*E[V''(m')]
                # So F'(c) = u''(c) - (-beta*R^2*E[V''(m')]) = u''(c) + beta*R^2*E[V''(m')]
                # Wait, my rhs_deriv above was d(rhs)/dc where rhs = beta*R*E[V'].
                # d(rhs)/dc = beta * d/dc(R*E[V']) = beta * E[V'' * (-R)] = -beta*R*E[V'']
                # So F'(c) = u''(c) - d(rhs)/dc = u''(c) - (-beta*R*E[V'']) = u''(c) + beta*R*E[V'']
                # Let's re-verify.
                # F(c) = u'(c) - G(c)
                # G(c) = beta * R * sum P V'(R(m-c)+y)
                # G'(c) = beta * R * sum P V''(R(m-c)+y) * (-R) = -beta * R^2 * sum P V''(...)
                # F'(c) = u''(c) - G'(c) = u''(c) + beta * R^2 * sum P V''(...)
                
                # In my code above:
                # rhs = beta * R * sum P V'
                # rhs_deriv (accumulated) = sum P V'' * (-R)
                # So d(rhs)/dc = beta * R * (sum P V'' * (-R)) = -beta * R^2 * sum P V''
                # This matches G'(c).
                # So F'(c) = u''(c) - d(rhs)/dc = u''(c) + beta * R^2 * sum P V''
                
                u2 = u_double_prime(c)
                F_prime = u2 - (beta * R * rhs_deriv) # Note: rhs_deriv is sum P V'' * (-R)
                # Wait. G'(c) = beta * R * rhs_deriv.
                # F'(c) = u''(c) - G'(c) = u''(c) - beta * R * rhs_deriv.
                
                # Newton step
                # c_new = c - F(c) / F'(c)
                # If F'(c) is close to 0 or step is too large, use bisection fallback
                valid_step = np.abs(F_prime) > 1e-12
                step = np.zeros(n_internal)
                step[valid_step] = F_val[valid_step] / F_prime[valid_step]
                
                # Limit step size to prevent divergence
                max_step = 0.1 * m_internal
                step = np.clip(step, -max_step, max_step)
                
                c_new = c - step
                c_new = np.clip(c_new, c_floor, m_internal)
                
                # Check convergence
                if np.max(np.abs(c_new - c)) < 1e-10 * np.max(m_internal):
                    c = c_new
                    break
                c = c_new
            
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
            
        # Update interpolators for next iteration (t -> t-1)
        for z in range(num_states):
            try:
                val_interps[z] = PchipInterpolator(m_internal, value_int[t, z, :])
                vp_interps[z] = PchipInterpolator(m_internal, u_prime(policy_int[t, z, :]))
                vpp_interps[z] = vp_interps[z].derivatives(1)
            except Exception:
                pass

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
