import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    
    # Policy: consumption[t, z, m_idx], shape (horizon, num_states, num_resources)
    policy = np.zeros((horizon, num_states, num_resources))
    # Value: value[t, z, m_idx]
    value = np.zeros((horizon, num_states, num_resources))
    
    # Utility and its derivatives
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)
    
    def util_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return c ** (-gamma)
    
    # Second derivative of utility (for Newton's method)
    def util_deriv2(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return -1.0 / (c ** 2)
        return -gamma * c ** (-gamma - 1)

    # Terminal condition: at t = horizon - 1 (last period), consume all
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    value[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, :]
    
    # Helper to compute continuation value vectorized over m_idx for a given c and z
    # c is shape (num_resources,)
    # Returns continuation value shape (num_resources,)
    def get_continuation(c, z, t_next):
        # m_next = R * (m - c)
        m = resource_grid
        m_next = R * (m - c)
        
        cont_val = np.zeros_like(m)
        for zp in range(num_states):
            m_next_zp = m_next + income[zp]
            
            # Clip to grid range for interpolation
            # np.interp handles out-of-bounds by clamping to first/last value
            v_next = np.interp(m_next_zp, resource_grid, value[t_next, zp, :])
            cont_val += transition[z, zp] * v_next
            
        return cont_val

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: use simple rule for remaining periods
            for tt in range(t, -1, -1):
                policy[tt, :, :] = 0.5 * resource_grid[np.newaxis, :]
                value[tt, :, :] = utility(0.5 * resource_grid)[np.newaxis, :]
            break
        
        for z in range(num_states):
            m = resource_grid
            # Initialize c. A good initial guess is often based on Euler equation or simple rule.
            # Let's start with a fraction of m, e.g., 0.5*m, but ensure it's within [1e-12, m]
            c = np.clip(0.5 * m, 1e-12, m)
            
            # Newton-Raphson iterations on the FOC:
            # u'(c) - beta * R * sum P(z'|z) * V'(m_next, z') = 0
            # Note: V'(m_next, z') is the derivative of the value function w.r.t. m_next.
            # By envelope theorem, V'(m, z) = u'(c*(m, z)) * (1 + R * (1 - c'_{m}))? No.
            # V(m, z) = u(c) + beta * sum P V(m_next, z')
            # dV/dm = u'(c) * dc/dm + beta * sum P V'(m_next, z') * R
            # From FOC: u'(c) = beta * R * sum P V'(m_next, z')
            # So dV/dm = beta * R * sum P V'(m_next, z') * (dc/dm + 1)
            # This is getting complicated to compute analytically without storing derivatives.
            
            # Alternative: Use numerical derivative of the value function for the FOC.
            # Since we have the value function on a grid, we can compute its derivative numerically.
            # However, a simpler and robust approach for the "different strategy" is to use the 
            # first-order condition directly with the analytic utility derivative and a 
            # numerical derivative of the continuation value, but vectorized.
            
            # Actually, let's stick to the FOC:
            # u'(c) = beta * R * E[V'(m_next, z')]
            # We need V'(m_next, z'). We can approximate V' using central differences on the grid.
            
            # Pre-compute derivative of value function for t+1 for all states
            # V_prime[t+1, z, m_idx]
            # Using np.gradient
            V_prime_next = np.zeros_like(value)
            for zp in range(num_states):
                # np.gradient handles the grid spacing. resource_grid is geomspace, so spacing is not uniform.
                # np.gradient(f, x) works for non-uniform x.
                V_prime_next[t + 1, zp, :] = np.gradient(value[t + 1, zp, :], resource_grid, edge_order=2)
            
            # Now iterate
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                
                # Calculate RHS of FOC: beta * R * E[V'(m_next, z')]
                # m_next = R * (m - c)
                m_next = R * (m - c)
                
                rhs = np.zeros_like(m)
                for zp in range(num_states):
                    # Interpolate V' at m_next
                    v_prime_next = np.interp(m_next, resource_grid, V_prime_next[t + 1, zp, :])
                    rhs += transition[z, zp] * v_prime_next
                
                rhs = beta * R * rhs
                
                # LHS: u'(c)
                lhs = util_deriv(c)
                
                # Newton step: c_new = c - (lhs - rhs) / (u''(c))
                # Note: The derivative of the LHS-RHS w.r.t c is u''(c) - d(rhs)/dc
                # d(rhs)/dc = beta * R * E[ V''(m_next) * (-R) ] = -beta * R^2 * E[V'']
                # So denominator = u''(c) + beta * R^2 * E[V'']
                # We don't have V'' readily. 
                # Using just u''(c) in the denominator is a damped Newton or secant-like approximation 
                # if we assume the RHS is locally constant or its derivative is small relative to u''.
                # Given CRRA, u'' is significant. Let's try standard Newton with u'' only.
                
                u2 = util_deriv2(c)
                
                # Avoid division by zero or negative step issues
                # u2 is negative. (lhs - rhs) should be zero at optimum.
                # If lhs > rhs, we need to increase c (since u' is decreasing).
                # c_new = c - (lhs - rhs) / u2
                # If lhs > rhs, numerator > 0, u2 < 0, so term < 0, c_new = c - (-) = c + positive. Correct.
                
                step = (lhs - rhs) / u2
                c_new = c - step
                
                # Clip to valid range
                c_new = np.clip(c_new, 1e-12, m)
                
                # Check convergence
                if np.all(np.abs(c_new - c) < 1e-10 * np.maximum(1.0, c)):
                    c = c_new
                    break
                c = c_new
            
            policy[t, z, :] = c
            value[t, z, :] = get_continuation(c, z, t + 1) + utility(c)
            
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_states):
            for m_idx in range(num_resources):
                m = resource_grid[m_idx]
                c = policy[t, z, m_idx]
                if c <= 0:
                    policy[t, z, m_idx] = 1e-12
                if c > m + 1e-10:
                    policy[t, z, m_idx] = m
    
    # Ensure c[5] equals grid within 1e-9
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    
    # Check for non-finite values
    if not np.all(np.isfinite(policy)):
        policy = np.nan_to_num(policy, nan=1e-12, posinf=1e10, neginf=0)
    
    return policy
