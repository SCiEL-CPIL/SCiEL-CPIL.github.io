"""Reconstructed design: GL8-start Nelder-Mead over diverse CPL cosmologies."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = _T0 + 78.0  # safety margin before 90 s kill


def _cpl_E(a, wa0, wa1):
    """E(a) = H(a)/H0 for CPL dark energy, flat, matter-only matter term."""
    return np.sqrt(a ** -3 + (1.0 + wa0 + wa1 * a) * a ** (3.0 * (1.0 + wa0 + wa1)))


def _ref_dL(z, wa0, wa1, n=40000):
    """Independent adaptive reference dL/(c/H0) via fine trapezoid on z."""
    a = 1.0 / (1.0 + z)
    return np.trapz((1.0 + z) * z / _cpl_E(a, wa0, wa1), a)


def _quad_dL(z, nodes, weights, wa0, wa1):
    a = 1.0 / (1.0 + z * nodes)
    return (1.0 + z) * z * np.sum(weights / _cpl_E(a, wa0, wa1))


def _rel_err(nodes, weights, z, wa0, wa1):
    ref = _ref_dL(z, wa0, wa1)
    q = _quad_dL(z, nodes, weights, wa0, wa1)
    return np.max(np.abs(q - ref) / ref) * 100.0


def _objective(x):
    nodes = x[:8]
    weights = x[8:]
    # penalty for constraint violations
    if np.any(weights <= 0) or np.any(np.diff(nodes) <= 0) \
            or nodes[0] <= 0 or nodes[-1] >= 1:
        return 1e6 + np.sum((weights[weights <= 0]) ** 2)
    if abs(np.sum(weights) - 1.0) > 1e-9:
        return 1e6 + (np.sum(weights) - 1.0) ** 2
    worst = 0.0
    for (wa0, wa1, z) in _COSMO:
        e = _rel_err(nodes, weights, z, wa0, wa1)
        if e > worst:
            worst = e
    return worst


def construct():
    global _COSMO
    if "_COSMO" not in globals() or _COSMO is None:
        # diverse CPL models, each at a representative redshift
        _COSMO = [
            (0.0, 0.0, 1.0), (0.0, 0.0, 5.0),
            (-1.0, 0.0, 1.0), (-1.0, 0.0, 5.0),
            (-0.5, -0.5, 2.0), (-0.5, -0.5, 8.0),
            (-1.0, 0.5, 1.5), (-1.0, 0.5, 6.0),
            (-0.8, -0.2, 3.0), (-0.8, -0.2, 7.0),
        ]
    # Gauss-Legendre 8 nodes/weights on [0,1] as starting point
    gl_x, gl_w = np.polynomial.legendre.leggauss(8)
    nodes0 = 0.5 * (gl_x + 1.0)
    weights0 = 0.5 * gl_w
    x0 = np.concatenate([nodes0, weights0])

    def _enforce(x):
        nodes = x[:8]
        weights = x[8:]
        # project nodes to strictly increasing interior via soft ordering
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        # enforce increasing by cumulative spacing
        d = np.diff(nodes)
        for i in range(len(d)):
            if d[i] <= 1e-8:
                nodes[i + 1] = nodes[i] + 1e-8
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        weights = np.abs(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        return np.concatenate([nodes, weights])

    best_x = _enforce(x0)
    best_val = _objective(best_x)

    # Nelder-Mead with adaptive maxfev; stop on deadline
    maxfev = 4000
    def _wrap(x):
        if time.time() > _DEADLINE:
            raise KeyboardInterrupt
        return _objective(_enforce(x))

    try:
        res = minimize(_wrap, _enforce(x0), method="Nelder-Mead",
                       options={"maxfev": maxfev, "xatol": 1e-10,
                                "fatol": 1e-10, "adaptive": True})
        if time.time() < _DEADLINE:
            cand = _enforce(res.x)
            v = _objective(cand)
            if v < best_val:
                best_val, best_x = v, cand
    except KeyboardInterrupt:
        pass

    nodes = best_x[:8]
    weights = best_x[8:]
    # final cleanup: strictly increasing, positive, sum 1
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    d = np.diff(nodes)
    for i in range(len(d)):
        if d[i] <= 1e-8:
            nodes[i + 1] = nodes[i] + 1e-8
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    weights = np.abs(weights)
    weights = weights / weights.sum()
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
