import time
import numpy as np


def _utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(c)
    return np.power(c, 1.0 - gamma) / (1.0 - gamma)


def _utility_prime(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return 1.0 / c
    return np.power(c, -gamma)


def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin below the 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = int(model['horizon'])

    m_grid = np.asarray(resource_grid, dtype=float)
    M = len(m_grid)
    Z = len(income)

    consumption = np.zeros((horizon, Z, M), dtype=float)

    # Terminal condition: c_H(m) = m
    consumption[horizon - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]

    # Fraction grid for candidate consumption: c = m * frac
    fracs = np.linspace(0.001, 1.0, int(budget))
    N = len(fracs)

    # Precompute continuation-value interpolation nodes (the resource grid).
    # For each (t, z) we need V_next(m', z') for m' = R*(m - c) + income[z'].
    # We interpolate V_next over m' using the previous period's policy to recover
    # the value, but a cleaner approach: compute the value function directly.

    # We'll maintain the value function V[t, z, m] via backward induction.
    # V[horizon-1, z, m] = u(m)  (consume everything at the end)
    V = np.zeros((horizon, Z, M), dtype=float)
    V[horizon - 1, :, :] = _utility(m_grid[np.newaxis, np.newaxis, :], gamma)

    # Work backwards from t = horizon-2 down to 0.
    for t in range(horizon - 2, -1, -1):
        # Check deadline
        if time.time() > deadline - 5.0:
            # Fallback: use a coarser fraction grid for remaining periods
            fracs = np.linspace(0.001, 1.0, max(10, N // 4))
            N = len(fracs)

        for z in range(Z):
            # c_grid[t, z, n] = m_grid * fracs[n]
            # Shape: (N, M) -> c[n, m_idx]
            c_grid = m_grid[np.newaxis, :] * fracs[:, np.newaxis]  # (N, M)

            # m' = R * (m - c) + income[z']
            # For each candidate n and grid point m_idx, and each z':
            # m_prime[n, m_idx, zp] = R * (m_grid[m_idx] - c_grid[n, m_idx]) + income[zp]
            m_prime = R * (m_grid[np.newaxis, np.newaxis, :] - c_grid[:, :, np.newaxis]) + income[np.newaxis, np.newaxis, :]
            # Shape: (N, M, Z)

            # Continuation value: sum over z' of transition[z, z'] * V[t+1, z', m']
            # Interpolate V[t+1, z', :] at m_prime
            # V_next[zp, m_idx] = V[t+1, zp, m_grid[m_idx]]
            # We need to interpolate V[t+1, zp, :] at m_prime[n, m_idx, zp]
            # Vectorize: for each zp, interpolate V[t+1, zp, :] at m_prime[:, :, zp]
            V_next = np.zeros((N, M, Z), dtype=float)
            for zp in range(Z):
                # m_prime[:, :, zp] has shape (N, M)
                # V[t+1, zp, :] has shape (M,)
                V_next[:, :, zp] = np.interp(m_prime[:, :, zp], m_grid, V[t + 1, zp, :])

            # Weighted sum over z'
            w = transition[z, :]  # shape (Z,)
            cont_val = np.sum(w[np.newaxis, np.newaxis, :] * V_next, axis=2)  # (N, M)

            # Objective: u(c) + beta * cont_val
            u_c = _utility(c_grid, gamma)  # (N, M)
            obj = u_c + beta * cont_val  # (N, M)

            # For each m_idx, find the n that maximizes obj
            best_n = np.argmax(obj, axis=0)  # (M,)
            c_opt = c_grid[best_n, np.arange(M)]  # (M,)

            # Enforce validity: 0 < c <= m + 1e-10
            c_opt = np.maximum(c_opt, 1e-12)
            c_opt = np.minimum(c_opt, m_grid + 1e-10)

            consumption[t, z, :] = c_opt

            # Update value function V[t, z, m] = obj at the optimal n
            V[t, z, :] = obj[best_n, np.arange(M)]

    # Final validity pass
    consumption = np.maximum(consumption, 1e-12)
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid + 1e-10)

    # Ensure terminal condition is exact
    consumption[horizon - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]

    # Ensure finite
    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=1e10, neginf=1e-12)

    return consumption
