"""Broadband antireflection coating design: DE global search + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_DEADLINE = time.time() + 140.0          # hard wall-clock safety margin
_WAVES = np.linspace(400.0, 800.0, 401)


def _reflectance_stack(labels, thicknesses, problem):
    """Exact normal-incidence reflectance (%) at _WAVES via characteristic matrices."""
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    ns = problem["substrate_index"]
    na = problem["ambient_index"]
    n = np.empty(len(labels))
    for i, lab in enumerate(labels):
        n[i] = nL if lab == "L" else nH
    d = np.asarray(thicknesses, dtype=float)
    k = 2.0 * np.pi / _WAVES
    # 2x2 characteristic matrix per layer
    c = n[:, None] * d[:, None] * k[None, :]
    M = np.empty((len(labels), 2, 2), dtype=complex)
    M[:, 0, 0] = np.cos(c)
    M[:, 1, 1] = np.cos(c)
    M[:, 0, 1] = 1j / n[:, None] * np.sin(c)
    M[:, 1, 0] = 1j * n[:, None] * np.sin(c)
    # propagate through the stack (air -> substrate)
    B = np.empty((2, 2), dtype=complex)
    B[0, 0] = 1.0
    B[1, 1] = 1.0
    for i in range(len(labels)):
        B = M[i] @ B
    r = (B[0, 0] + B[0, 1] * ns - B[1, 0] * na - B[1, 1] * na * ns) / \
        (B[0, 0] + B[0, 1] * ns + B[1, 0] * na + B[1, 1] * na * ns)
    return (np.abs(r) ** 2 * 100.0).mean()


def design(problem):
    band = problem["band_nm"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    lam0 = 0.5 * (band[0] + band[1])

    # --- fixed layer structure (labels) chosen for broadband cancellation ---
    labels = ["H"] * 2 + ["L", "H"] * 6 + ["L"] * 2   # 16 layers
    # starting thicknesses: quarter-wave at band centre
    d0 = np.array([lam0 / (4.0 * (nL if lab == "L" else nH)) for lab in labels])

    def _objective(x):
        if x.sum() > max_total:
            return 1e6 + 10.0 * (x.sum() - max_total)
        return _reflectance_stack(labels, x, problem)

    bounds = [(0.0, max_layer) for _ in labels]
    best_x = d0.copy()
    best_f = _objective(d0)

    # --- global exploration: differential evolution ---
    try:
        res = differential_evolution(
            _objective, bounds, x0=d0, seed=7,
            maxiter=25, popsize=15, tol=1e-9,
            init="sobol", polish=False,
            updating="deferred",
            workers=1,
        )
        if res.success or (res.fun is not None and res.fun < best_f):
            best_x = res.x.copy()
            best_f = float(res.fun)
    except Exception:
        pass

    # --- local polish: Nelder-Mead ---
    if time.time() < _DEADLINE - 5.0:
        try:
            res2 = minimize(
                _objective, best_x, method="Nelder-Mead",
                options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-9,
                         "adaptive": True},
            )
            if res2.fun is not None and res2.fun < best_f:
                best_x = res2.x.copy()
                best_f = float(res2.fun)
        except Exception:
            pass

    # --- final constraint clamp ---
    best_x = np.clip(best_x, 0.0, max_layer)
    if best_x.sum() > max_total:
        best_x = best_x * (max_total / best_x.sum())

    return list(labels), [float(t) for t in best_x]
# EVOLVE-BLOCK-END
