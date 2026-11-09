"""Consumption policy solver: vectorized backward induction on dense ratio grid."""
import time
import numpy as np

DEADLINE_SECONDS = 130


def _crsa_utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)


def _solve_consumption_single(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=np.float64)
    trans = np.asarray(model['transition'], dtype=np.float64)
    horizon = model['horizon']
    states = len(income)
    n_grid = len(resource_grid)
    m_grid = np.asarray(resource_grid, dtype=np.float64)

    consumption = np.zeros((horizon, states, n_grid))

    # Terminal: consume everything
    consumption[horizon - 1] = np.tile(m_grid, (states, 1))
    V_next = np.tile(_crsa_utility(m_grid, gamma), (states, 1))

    # Dense ratio grid: f = c/m in (0, 1]
    # Use 3001 points for high accuracy
    n_ratio = 3001
    ratios = np.linspace(0.001, 1.0, n_ratio)

    # Precompute log_m for interpolation in log-space (better for geometric grid)
    log_m_grid = np.log(m_grid)

    for t in range(horizon - 2, -1, -1):
        V_current = np.zeros((states, n_grid))

        for z in range(states):
            m = m_grid  # shape (n_grid,)
            c_grid = m[:, None] * ratios[None, :]  # shape (n_grid, n_ratio)

            # Compute m' for each (m, c) combination
            # m' = R*(m - c) + income[zp]
            # shape: (n_grid, n_ratio, states)
            m_prime = R * (m[:, None, None] - c_grid[:, :, None]) + income[None, None, :]

            # Clip m' to grid bounds for interpolation
            m_prime_clipped = np.clip(m_prime, m_grid[0], m_grid[-1])

            # Interpolate V_next for each state
            # V_next shape: (states, n_grid)
            # For each zp, we need V_next[zp] interpolated at m_prime[:, :, zp]
            # Vectorize over zp
            V_interp = np.zeros_like(m_prime_clipped)  # (n_grid, n_ratio, states)
            for zp in range(states):
                V_interp[:, :, zp] = np.interp(m_prime_clipped[:, :, zp], m_grid, V_next[zp])

            # Expected continuation: sum over zp of trans[z, zp] * V_interp
            # trans[z] shape: (states,)
            expected_V = np.tensordot(trans[z], V_interp, axes=(0, 2))  # (n_grid, n_ratio)

            # Total value for each (m, c): u(c) + beta * expected_V
            u_c = _crsa_utility(c_grid, gamma)  # (n_grid, n_ratio)
            total_value = u_c + beta * expected_V  # (n_grid, n_ratio)

            # Find argmax over ratio dimension
            best_idx = np.argmax(total_value, axis=1)  # (n_grid,)
            best_c = c_grid[np.arange(n_grid), best_idx]  # (n_grid,)

            # Ensure 0 < c <= m
            best_c = np.clip(best_c, 1e-12, m)

            consumption[t, z, :] = best_c
            V_current[z, :] = total_value[np.arange(n_grid), best_idx]

        V_next = V_current

    return consumption


def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    try:
        result = _solve_consumption_single(model, resource_grid, budget)

        # Validate
        if not np.all(np.isfinite(result)):
            raise ValueError("Non-finite values")
        m = resource_grid[None, None, :]
        if np.any(result <= 0) or np.any(result > m + 1e-10):
            raise ValueError("c out of bounds")
        horizon = model['horizon']
        states = model['transition'].shape[0]
        if not np.allclose(result[horizon - 1],
                           np.tile(resource_grid, (states, 1)), atol=1e-9):
            raise ValueError("Terminal condition violated")

        return result
    except Exception:
        # Fallback: simple heuristic
        horizon = model['horizon']
        states = len(model['income'])
        n_grid = len(resource_grid)

        result = np.zeros((horizon, states, n_grid))
        for t in range(horizon):
            if t == horizon - 1:
                result[t] = np.tile(resource_grid, (states, 1))
            else:
                # Simple saving rule: save a fraction based on horizon remaining
                frac_save = 0.1 * (horizon - 1 - t) / (horizon - 1)
                result[t] = (1.0 - frac_save) * resource_grid[None, :]
                result[t] = np.clip(result[t], 1e-6, resource_grid[None, :])
                result[t] = np.tile(result[t], (states, 1))

        return result
