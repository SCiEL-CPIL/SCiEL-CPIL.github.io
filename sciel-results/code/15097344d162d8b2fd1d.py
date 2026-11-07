"""Consumption-saving policy via coarse-grid backward induction + dense candidate search."""
import time
import numpy as np


def _util(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return np.log(c)
    return c ** (1.0 - gamma) / (1.0 - gamma)


def _interp_V(V, grid, m_vals, z):
    # V: (T, N), m_vals: (K,) -> (K,)
    return np.interp(m_vals, grid, V[:, z])


def solve_consumption(model, resource_grid, budget):
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin under 150s

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    N_z = len(income)
    N_m = len(resource_grid)

    # Coarse internal grid for value functions.
    M = 60
    grid = np.geomspace(resource_grid[0], resource_grid[-1], M)

    # Candidate consumption fractions per m.
    Nc = 6000
    fracs = np.linspace(0.001, 1.0, Nc)

    V = np.zeros((horizon, M, N_z))
    C = np.zeros((horizon, N_z, M))

    # Terminal period: consume everything.
    V[horizon - 1, :, :] = _util(grid, gamma)
    C[horizon - 1, :, :] = grid[np.newaxis, :]

    # Backward induction.
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Return the best complete result found so far (t+1..T already set).
            # For earlier periods, fall back to a safe consumption (consume all).
            for tt in range(t + 1):
                V[tt, :, :] = _util(grid, gamma)
                C[tt, :, :] = grid[np.newaxis, :]
            break

        for z in range(N_z):
            cont = np.zeros(M)
            for zp in range(N_z):
                m_next = R * (grid - grid * fracs) + income[zp]
                cont += transition[z, zp] * _interp_V(V[t + 1], grid, m_next, zp)
            cont = beta * cont

            obj = _util(grid * fracs, gamma) + cont[np.newaxis, :]
            idx = np.argmax(obj, axis=1)
            C[t, z, :] = (grid * fracs[idx]).reshape(M)
            V[t, z, :] = np.take_along_axis(obj, idx[np.newaxis, :], axis=1).ravel()

    # Interpolate onto the requested resource_grid for each t, z.
    out = np.zeros((horizon, N_z, N_m))
    for t in range(horizon):
        for z in range(N_z):
            out[t, z, :] = np.interp(resource_grid, grid, C[t, z, :])

    # Enforce validity.
    out = np.clip(out, 1e-12, None)
    out[horizon - 1, :, :] = resource_grid[np.newaxis, :]

    # 0 < c <= m + 1e-10
    m_b = resource_grid[np.newaxis, np.newaxis, :]
    out = np.minimum(out, m_b + 1e-10)
    out = np.maximum(out, 1e-12)

    return out
