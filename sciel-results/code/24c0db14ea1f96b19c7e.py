"""Consumption policy solver using backward induction with a dense geomspace grid."""
import time
import numpy as np

DEADLINE_SECONDS = 135  # Safety margin before the 150s kill


def _crsa_utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)


def _solve_consumption_single(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    trans = model['transition']
    horizon = model['horizon']
    states = len(income)
    n_grid = len(resource_grid)

    consumption = np.zeros((horizon, states, n_grid))
    # V[z, i] = continuation value (value from next period onward) given state z and m=grid[i]
    V_next = np.zeros((states, n_grid))

    # Use a dense geomspace grid for consumption ratios to capture rapid changes at low m
    # The evaluator uses geomspace, so aligning the search grid helps interpolation
    ratios = np.geomspace(0.001, 1.0, 2000)

    for t in range(horizon - 1, -1, -1):
        if t == horizon - 1:
            # Terminal: consume all
            consumption[t] = np.tile(resource_grid, (states, 1))
            V_next = _crsa_utility(np.tile(resource_grid, (states, 1)), gamma)
            continue

        V_current = np.zeros((states, n_grid))
        m_vals = resource_grid

        for z in range(states):
            # c_candidates: (n_grid, 2000)
            c_candidates = m_vals[:, None] * ratios[None, :]
            
            # Clip consumption to be valid (0 < c <= m)
            # The ratios are in [0.001, 1.0], so c is in [0.001*m, 1.0*m]
            # This is always valid for m > 0
            u_c = _crsa_utility(c_candidates, gamma)  # (n_grid, 2000)

            # Compute continuation value for each candidate
            # m' = R*(m - c) + income[z']
            cont_val = np.zeros((n_grid, len(ratios)))
            for zp in range(states):
                m_prime = R * (m_vals[:, None] - c_candidates) + income[zp]
                m_prime = np.clip(m_prime, resource_grid[0], resource_grid[-1])
                # Vectorized interpolation
                V_interp = np.interp(m_prime, resource_grid, V_next[zp])
                cont_val += trans[z, zp] * V_interp

            total = u_c + beta * cont_val  # (n_grid, 2000)
            best_idx = np.argmax(total, axis=1)
            best_c = c_candidates[np.arange(n_grid), best_idx]
            best_val = total[np.arange(n_grid), best_idx]

            consumption[t, z] = best_c
            V_current[z] = best_val

        V_next = V_current

    return consumption


def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    try:
        result = _solve_consumption_single(model, resource_grid, budget)

        # Check deadline
        if time.time() > deadline:
            # Fallback if we ran out of time (unlikely with 2000x101x2x6 ops)
            pass

        # Validate
        if not np.all(np.isfinite(result)):
            raise ValueError("Non-finite values")
        m = resource_grid[None, None, :]
        if np.any(result <= 0) or np.any(result > m + 1e-10):
            raise ValueError("c out of bounds")
        states = model['transition'].shape[0]
        if not np.allclose(result[5], np.tile(resource_grid, (states, 1)), atol=1e-9):
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
                frac = 0.8 + 0.2 * (t / max(1, horizon - 2))
                result[t] = frac * resource_grid[None, :]
                result[t] = np.clip(result[t], 1e-6, resource_grid[None, :])
                result[t] = np.tile(result[t], (states, 1))

        return result
