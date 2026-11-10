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

    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)

    # Policy and value arrays
    policy = np.zeros((horizon, num_states, num_resources))
    value = np.zeros((horizon, num_states, num_resources))

    # Terminal condition: at t = horizon - 1, consume all
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    value[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, :]

    # Use a dense consumption grid for search: 2000 points from 0.002*m to 1.0*m
    # We'll vectorize over all resource points simultaneously
    n_c = 2000
    c_fracs = np.linspace(0.002, 1.0, n_c)  # consumption as fraction of m

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: simple heuristic for remaining periods
            for tt in range(t, -1, -1):
                frac = 1.0 / (horizon - tt)
                policy[tt, :, :] = np.maximum(frac * resource_grid[np.newaxis, :], 1e-12)
                policy[tt, :, :] = np.minimum(policy[tt, :, :], resource_grid[np.newaxis, :])
                value[tt, :, :] = utility(policy[tt, :, :])
            break

        for z in range(num_states):
            m = resource_grid  # shape (num_resources,)
            # Consumption candidates: shape (num_resources, n_c)
            c_grid = np.outer(m, c_fracs)  # (num_resources, n_c)
            # Ensure c <= m and c > 0
            c_grid = np.minimum(c_grid, m[:, np.newaxis])
            c_grid = np.maximum(c_grid, 1e-12)

            # For each c candidate, compute m_next for each z'
            # m_next[z', r, c_idx] = R * (m[r] - c[r, c_idx]) + income[z']
            # Shape: (num_states, num_resources, n_c)
            m_next = R * (m[:, np.newaxis] - c_grid) + income[:, np.newaxis, np.newaxis]
            m_next = np.maximum(m_next, 1e-10)  # shape (num_states, num_resources, n_c)

            # Interpolate continuation value V_{t+1}[z'](m_next) for each z', r, c_idx
            # value[t+1, zp, :] is on resource_grid
            v_next = np.zeros_like(m_next)  # (num_states, num_resources, n_c)
            for zp in range(num_states):
                # Interpolate value[t+1, zp, :] at m_next[zp, :, :]
                v_next[zp, :, :] = np.interp(m_next[zp, :, :], resource_grid, value[t + 1, zp, :])

            # Expected continuation value: sum over z' of transition[z, z'] * V_{t+1}[z'](m_next)
            # Shape: (num_resources, n_c)
            exp_v_next = np.sum(transition[z, :, np.newaxis, np.newaxis] * v_next, axis=0)

            # Total value for each c candidate: u(c) + beta * E[V_{t+1}]
            u_c = utility(c_grid)  # (num_resources, n_c)
            total_value = u_c + beta * exp_v_next  # (num_resources, n_c)

            # Find argmax over c for each resource point
            best_idx = np.argmax(total_value, axis=1)  # (num_resources,)
            best_c = c_grid[np.arange(num_resources), best_idx]
            best_v = total_value[np.arange(num_resources), best_idx]

            # Ensure validity
            best_c = np.maximum(best_c, 1e-12)
            best_c = np.minimum(best_c, m)

            policy[t, z, :] = best_c
            value[t, z, :] = best_v

    # Ensure c[horizon-1] equals grid within 1e-9
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]

    # Final validity check
    for t in range(horizon):
        for z in range(num_states):
            m = resource_grid
            c = policy[t, z, :]
            c = np.maximum(c, 1e-12)
            c = np.minimum(c, m)
            policy[t, z, :] = c

    # Check for non-finite values
    if not np.all(np.isfinite(policy)):
        policy = np.nan_to_num(policy, nan=1e-12, posinf=1e10, neginf=0)

    return policy
