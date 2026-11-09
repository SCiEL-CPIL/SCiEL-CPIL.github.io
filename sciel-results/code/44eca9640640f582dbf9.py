"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    grid = np.asarray(resource_grid, dtype=float)
    n_grid = len(grid)
    n_z = len(income)

    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1.0 - gamma) / (1.0 - gamma)

    V = np.zeros((horizon + 1, n_z, n_grid))
    C = np.zeros((horizon, n_z, n_grid))

    C[horizon - 1, :, :] = grid[np.newaxis, :, :]
    V[horizon, :, :] = 0.0

    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5.0:
            break

        for z in range(n_z):
            for i_m in range(n_grid):
                m = grid[i_m]
                if m < 1e-12:
                    C[t, z, i_m] = 1e-12
                    V[t, z, i_m] = 0.0
                    continue

                def objective(c):
                    c = max(c, 1e-12)
                    c = min(c, m + 1e-10)
                    m_next_base = R * (m - c)
                    cont_val = 0.0
                    for z_next in range(n_z):
                        prob = transition[z, z_next]
                        m_next = m_next_base + income[z_next]
                        v_next = np.interp(m_next, grid, V[t + 1, z_next, :])
                        cont_val += prob * v_next
                    return u(c) + beta * cont_val

                res = minimize_scalar(lambda c: -objective(c), bounds=(1e-12, m + 1e-10), method='bounded', options={'xatol': 1e-8})
                c_opt = res.x if res.success else m * 0.5
                c_opt = max(c_opt, 1e-12)
                c_opt = min(c_opt, m + 1e-10)

                C[t, z, i_m] = c_opt
                V[t, z, i_m] = objective(c_opt)

    C[horizon - 1, :, :] = grid[np.newaxis, :, :]
    for t in range(horizon):
        for z in range(n_z):
            C[t, z, :] = np.clip(C[t, z, :], 1e-12, grid + 1e-10)

    return C
# EVOLVE-BLOCK-END
