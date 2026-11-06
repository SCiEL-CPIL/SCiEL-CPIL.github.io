import numpy as np
from scipy.optimize import minimize
import time


def angles(n, edges, p):
    start_time = time.time()
    deadline = start_time + 110.0

    dim = 1 << n
    psi0 = np.ones(dim, dtype=complex) / np.sqrt(dim)
    states_arr = np.arange(dim)

    # Build H diagonal
    h_diag = np.zeros(dim, dtype=float)
    for u, v in edges:
        bu = (states_arr >> u) & 1
        bv = (states_arr >> v) & 1
        h_diag += 1.0 - 2.0 * np.abs(bu - bv)

    # Precompute for expected cut
    edge_zz = []
    for u, v in edges:
        bu = (states_arr >> u) & 1
        bv = (states_arr >> v) & 1
        edge_zz.append(1.0 - 2.0 * np.abs(bu - bv))
    edge_zz = np.array(edge_zz)  # shape (m, dim)

    # Precompute for X rotation
    x_rot_masks = []
    for i in range(n):
        bit_i = (states_arr >> i) & 1
        mask0 = bit_i == 0
        states0 = np.where(mask0)[0]
        partners = states0 | (1 << i)
        x_rot_masks.append((states0, partners))

    def apply_cost(psi, gamma):
        phases = np.exp(1j * gamma / 2.0 * h_diag)
        return phases * psi

    def apply_x_rotation(psi, beta):
        c = np.cos(beta)
        s = np.sin(beta)
        psi = psi.copy()
        for states0, partners in x_rot_masks:
            a0 = psi[states0]
            a1 = psi[partners]
            psi[states0] = c * a0 - 1j * s * a1
            psi[partners] = -1j * s * a0 + c * a1
        return psi

    def expected_cut(psi):
        probs = np.abs(psi)**2
        zz_vals = edge_zz @ probs  # (m,)
        return 0.5 * np.sum(1.0 - zz_vals)

    def objective(params):
        g1, b1, g2, b2 = params
        psi = apply_cost(psi0, g1)
        psi = apply_x_rotation(psi, b1)
        psi = apply_cost(psi, g2)
        psi = apply_x_rotation(psi, b2)
        return -expected_cut(psi)

    best_params = None
    best_val = -np.inf

    initial_guesses = [
        [0.5, 0.5, 0.5, 0.5],
        [0.3, 0.3, 0.3, 0.3],
        [0.7, 0.4, 0.7, 0.4],
        [0.2, 0.6, 0.2, 0.6],
        [1.0, 0.5, 1.0, 0.5],
        [0.4, 0.2, 0.4, 0.2],
        [0.6, 0.3, 0.6, 0.3],
        [0.8, 0.2, 0.8, 0.2],
        [0.45, 0.15, 0.45, 0.15],
        [0.55, 0.25, 0.55, 0.25],
    ]

    for x0 in initial_guesses:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-11})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            continue

    if best_params is None:
        return np.array([[0.35, 0.25], [0.35, 0.25]])

    if time.time() < deadline - 5:
        try:
            res2 = minimize(objective, best_params, method='L-BFGS-B',
                           options={'maxiter': 300, 'gtol': 1e-12})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x
        except Exception:
            pass

    g1, b1, g2, b2 = best_params
    return np.array([[g1, b1], [g2, b2]])
