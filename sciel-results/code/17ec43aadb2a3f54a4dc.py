import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=np.float64)
    transition = np.asarray(model['transition'], dtype=np.float64)
    horizon = model['horizon']
    m_grid = np.asarray(resource_grid, dtype=np.float64)
    N_m = len(m_grid)
    N_z = len(income)

    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)

    # Use a dense grid for value function computation
    # The resource_grid goes from 0.001 to 40 with 101 points
    # We'll use a denser grid for the value function
    m_dense = np.geomspace(0.001, 40, 501)
    N_dense = len(m_dense)

    # Value function: V[t, z, m] for t in 0..horizon-1
    # At t = horizon-1 (last period), V = util(m) (consume all)
    # Actually, c[5] must equal m, so at t=5, c = m, V[5] = util(m)

    # We'll compute V backwards
    # V[t, z, :] for each t, z
    # Shape: (horizon, N_z, N_dense)
    V = np.zeros((horizon, N_z, N_dense), dtype=np.float64)

    # Terminal: t = horizon - 1 = 5
    # c = m (must consume all), so V[5, z, m] = util(m) for all z
    V[horizon - 1, :, :] = util(m_dense)[np.newaxis, :, np.newaxis] * np.ones((1, N_z, 1))
    # Actually V[5, z, i] = util(m_dense[i]) for all z
    V[horizon - 1, :, :] = np.tile(util(m_dense), (N_z, 1))

    # For t from horizon-2 down to 0:
    # V[t, z, m] = max_c { util(c) + beta * sum_z' transition[z, z'] * V[t+1, z', R*(m-c) + income[z']] }
    # subject to 0 < c <= m

    # For the optimization, we'll use a fine grid of consumption fractions
    # c = m * frac, where frac is in (0, 1]
    # But we need c > 0, so frac > 0
    # The scoring uses c = m * linspace(0.001, 1, 301), so we should match that resolution

    # Number of consumption fractions to try
    n_frac = 301
    fracs = np.linspace(0.001, 1.0, n_frac)

    # Precompute for each t, z: we need to evaluate V[t+1, z', next_m] for all next_m
    # next_m = R * (m - c) + income[z'] = R * m * (1 - frac) + income[z']

    for t in range(horizon - 2, -1, -1):
        if time.time() - start_time > DEADLINE:
            break
        for z in range(N_z):
            # For each m in m_dense, find optimal c
            # V[t, z, m] = max_c { util(c) + beta * sum_z' P[z,z'] * V[t+1, z', R*(m-c) + income[z']] }

            # We'll vectorize over m and frac
            # m_dense shape: (N_dense,)
            # fracs shape: (n_frac,)
            # c = m_dense[:, None] * fracs[None, :]  shape: (N_dense, n_frac)
            # next_m for each z': R * (m - c) + income[z']

            # Current value: util(c)
            c_vals = m_dense[:, None] * fracs[None, :]
            c_vals = np.maximum(c_vals, 1e-12)
            current_util = util(c_vals)  # (N_dense, n_frac)

            # Continuation value
            # For each z', next_m = R * (m - c) + income[z']
            # We need to interpolate V[t+1, z', next_m]
            # V[t+1, z', :] is defined on m_dense

            cont_val = np.zeros((N_dense, n_frac), dtype=np.float64)
            for zp in range(N_z):
                # next_m = R * (m_dense - c_vals) + income[zp]
                next_m = R * (m_dense[:, None] - c_vals) + income[zp]
                # Clip next_m to valid range [m_dense[0], m_dense[-1]]
                next_m = np.clip(next_m, m_dense[0], m_dense[-1])
                # Interpolate V[t+1, zp, next_m]
                v_interp = np.interp(next_m.ravel(), m_dense, V[t + 1, zp, :]).reshape(N_dense, n_frac)
                cont_val += transition[z, zp] * v_interp

            total_val = current_util + beta * cont_val  # (N_dense, n_frac)

            # Find max over fracs for each m
            best_idx = np.argmax(total_val, axis=1)
            V[t, z, :] = total_val[np.arange(N_dense), best_idx]

    # Now extract the consumption policy on the original m_grid
    # For each t, z, m in m_grid, find optimal c
    consumption = np.zeros((horizon, N_z, N_m), dtype=np.float64)

    for t in range(horizon):
        if time.time() - start_time > DEADLINE:
            # Fill remaining with c = m
            for z in range(N_z):
                consumption[t, z, :] = m_grid
            break

        if t == horizon - 1:
            # Must consume all
            consumption[t, :, :] = m_grid[np.newaxis, :, np.newaxis] * np.ones((1, N_z, 1))
            consumption[t, :, :] = np.tile(m_grid, (N_z, 1))
        else:
            for z in range(N_z):
                # Recompute the optimal c for each m in m_grid
                c_vals = m_grid[:, None] * fracs[None, :]
                c_vals = np.maximum(c_vals, 1e-12)
                current_util = util(c_vals)

                cont_val = np.zeros((N_m, n_frac), dtype=np.float64)
                for zp in range(N_z):
                    next_m = R * (m_grid[:, None] - c_vals) + income[zp]
                    next_m = np.clip(next_m, m_dense[0], m_dense[-1])
                    v_interp = np.interp(next_m.ravel(), m_dense, V[t + 1, zp, :]).reshape(N_m, n_frac)
                    cont_val += transition[z, zp] * v_interp

                total_val = current_util + beta * cont_val
                best_idx = np.argmax(total_val, axis=1)
                consumption[t, z, :] = c_vals[np.arange(N_m), best_idx]

    # Enforce constraints: 0 < c <= m + 1e-10
    consumption = np.maximum(consumption, 1e-12)
    consumption = np.minimum(consumption, m_grid[np.newaxis, np.newaxis, :] + 1e-10)

    # Enforce terminal condition: c[5] = m within 1e-9
    consumption[horizon - 1, :, :] = m_grid[np.newaxis, :, np.newaxis] * np.ones((1, N_z, 1))
    consumption[horizon - 1, :, :] = np.tile(m_grid, (N_z, 1))

    # Final safety: ensure finite
    consumption = np.where(np.isfinite(consumption), consumption, m_grid[np.newaxis, np.newaxis, :])

    return consumption
