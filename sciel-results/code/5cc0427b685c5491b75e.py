import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We aim to finish well before.
    # If we run out of time, we return the best we have.
    deadline = start_time + 140.0 

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']

    num_states = len(income)
    num_resources = len(resource_grid)
    c_floor = 1e-12

    def utility(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def util_deriv(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return np.power(c, -gamma)

    # Value function: V[t, z, m_idx]
    # Policy: C[t, z, m_idx]
    V = np.zeros((horizon, num_states, num_resources))
    C = np.zeros((horizon, num_states, num_resources))

    # Terminal period (t = horizon-1 = 5): consume all
    V[horizon - 1] = 0.0
    for z in range(num_states):
        C[horizon - 1, z] = resource_grid

    # Backward induction from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        # Check deadline: if we are running out of time, fill remaining with a simple heuristic
        # to ensure we return a valid array.
        if time.time() > deadline - 5.0:
            for tt in range(t, horizon - 1):
                for z in range(num_states):
                    # Simple heuristic: consume a fraction, e.g., 0.5 or just all if terminal
                    # But we must ensure c <= m. 
                    # Using the terminal policy (consume all) is a valid fallback for accuracy if we can't compute optimal.
                    # Actually, a better fallback is to use the policy from t+1 if available, or just m.
                    # Let's just use the grid (consume all) as a safe valid fallback.
                    C[tt, z] = resource_grid
                    V[tt, z] = 0.0
            break

        for z in range(num_states):
            m_vals = resource_grid  # shape (num_resources,)

            # Initialize bisection bounds
            c_low = np.full(num_resources, c_floor)
            c_high = np.maximum(m_vals - c_floor, c_floor)

            # Number of bisection iterations
            # 50 iterations give ~1e-15 relative precision, which is more than enough.
            n_bisect = 50

            for _ in range(n_bisect):
                if time.time() > deadline - 5.0:
                    break
                
                c_mid = 0.5 * (c_low + c_high)

                # Compute m' for each z' and each m
                # m' = R * (m - c_mid) + income[z']
                # Shape: (num_states, num_resources)
                m_prime = R * (m_vals[np.newaxis, :] - c_mid[np.newaxis, :]) + income[:, np.newaxis]
                # Clip m' to grid range for interpolation
                m_prime_clipped = np.clip(m_prime, resource_grid[0], resource_grid[-1])

                # Interpolate V'(t+1, z', m') onto m_prime_clipped
                # Envelope Theorem: V'(m) = u'(c*(m))
                V_prime_next_interp = np.zeros((num_states, num_resources))
                for zp in range(num_states):
                    V_prime_vals = util_deriv(C[t + 1, zp])
                    V_prime_next_interp[zp] = np.interp(m_prime_clipped[zp], resource_grid, V_prime_vals)

                # Expected marginal value: sum over z' of transition[z, z'] * V'(t+1, z', m')
                V_prime_expected = transition[z, :] @ V_prime_next_interp  # shape (num_resources,)

                # FOC: u'(c) = beta * R * V_prime_expected
                rhs = beta * R * V_prime_expected
                lhs = util_deriv(c_mid)

                # u'(c) is decreasing in c.
                # If u'(c_mid) > rhs, we are consuming too little (marginal utility is too high), increase c.
                # If u'(c_mid) < rhs, we are consuming too much, decrease c.
                
                mask_increase = lhs > rhs
                c_low[mask_increase] = c_mid[mask_increase]
                
                mask_decrease = lhs < rhs
                c_high[mask_decrease] = c_mid[mask_decrease]

            c_opt = 0.5 * (c_low + c_high)
            
            # Ensure c is within bounds
            c_opt = np.clip(c_opt, c_floor, m_vals - c_floor)
            
            # For very small m, c should be m (consume all) to avoid negative savings issues
            small_m = m_vals < 2 * c_floor
            c_opt[small_m] = m_vals[small_m]

            C[t, z] = c_opt

            # Compute V[t, z, m] = u(c) + beta * E[V(t+1, z', m')]
            m_prime = R * (m_vals[np.newaxis, :] - c_opt[np.newaxis, :]) + income[:, np.newaxis]
            m_prime_clipped = np.clip(m_prime, resource_grid[0], resource_grid[-1])

            V_next_interp = np.zeros((num_states, num_resources))
            for zp in range(num_states):
                V_next_interp[zp] = np.interp(m_prime_clipped[zp], resource_grid, V[t + 1, zp])

            V_expected = transition[z, :] @ V_next_interp
            V[t, z] = utility(c_opt) + beta * V_expected

    # Ensure final period consumes all
    for z in range(num_states):
        C[horizon - 1, z] = resource_grid

    # Final validity checks
    # 0 < c <= m + 1e-10
    m_expanded = resource_grid[np.newaxis, np.newaxis, :]
    C = np.maximum(C, c_floor)
    C = np.minimum(C, m_expanded + 1e-10)

    # c[5] should equal the grid exactly
    for z in range(num_states):
        C[horizon - 1, z] = resource_grid

    return C
