"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 85.0  # safety margin before 90s kill


def _E(z, w0, wa, wb):
    """H(a)/H0 for a CPL dark-energy model in a flat universe."""
    a = 1.0 / (1.0 + z)
    Edm = np.power(a, 3.0 + 3.0 * wa + 3.0 * wb * z)
    return np.sqrt(a ** -2.0 + Edm)


def _integrand(z, w0, wa, wb):
    return z / _E(z, w0, wa, wb)


def _quad(z, nodes, weights, w0, wa, wb):
    return z * np.sum(weights / _E(nodes, w0, wa, wb))


def _true(z, w0, wa, wb, n=40000):
    zz = np.linspace(0.0, z, n)
    f = _integrand(zz, w0, wa, wb)
    return z * np.trapz(f, zz)


def _relerr(z, nodes, weights, w0, wa, wb):
    q = _quad(z, nodes, weights, w0, wa, wb)
    t = _true(z, w0, wa, wb)
    return abs(q - t) / t


def _objective(v, models, zs):
    n = 8
    t = time.time()
    if t > _DEADLINE:
        return 1e8
    logw = v[:n]
    x = v[n:]
    w = np.exp(logw)
    w = w / w.sum()
    nodes = np.clip(np.tanh(x), -1.0, 1.0)
    nodes = 0.5 * (nodes + 1.0)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    if not np.all(np.diff(nodes) > 0):
        return 1e8
    maxerr = 0.0
    for (w0, wa, wb) in models:
        for zz in zs:
            e = _relerr(zz, nodes, w, w0, wa, wb)
            if e > maxerr:
                maxerr = e
            if maxerr > 1e-2:
                return maxerr
    return maxerr


def _best_result():
    n = 8
    nodes = (np.arange(n) + 0.5) / n
    weights = np.full(n, 1.0 / n)
    return {"nodes": nodes.astype(float), "weights": weights.astype(float)}


def construct():
    t0 = time.time()
    n = 8

    models = [
        (0.3, 0.0, 0.0),
        (0.3, -1.0, 0.0),
        (0.3, -0.5, 0.5),
        (0.3, -0.5, -0.5),
        (0.3, -1.0, 1.0),
        (0.7, -1.0, 0.0),
        (0.7, -0.5, 0.5),
        (0.3, -1.0, 0.5),
        (0.3, -0.9, 0.3),
        (0.3, -1.2, 0.4),
    ]
    zs = [0.2, 0.5, 1.0, 2.0, 3.0]

    # Initial guess: midpoint nodes, equal weights
    nodes0 = (np.arange(n) + 0.5) / n
    weights0 = np.full(n, 1.0 / n)
    logw0 = np.log(weights0)
    # Map nodes0 to x-space via inverse tanh
    x0 = np.arctanh(2.0 * nodes0 - 1.0)
    v0 = np.concatenate([logw0, x0])

    best_v = v0.copy()
    best_err = _objective(v0, models, zs)
    best_nodes = nodes0.copy()
    best_weights = weights0.copy()

    def _update(v, err):
        nonlocal best_v, best_err, best_nodes, best_weights
        if err < best_err:
            best_err = err
            best_v = v.copy()
            logw = v[:n]
            x = v[n:]
            w = np.exp(logw)
            w = w / w.sum()
            nodes = np.clip(np.tanh(x), -1.0, 1.0)
            nodes = 0.5 * (nodes + 1.0)
            nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
            best_nodes = nodes
            best_weights = w

    _update(v0, best_err)

    # Coordinate descent on nodes (with fixed weights) to improve
    def _opt_nodes(weights):
        if time.time() > _DEADLINE - 5:
            return nodes0
        def obj(x):
            nodes = 0.5 * (np.tanh(x) + 1.0)
            nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
            if not np.all(np.diff(nodes) > 0):
                return 1e8
            maxerr = 0.0
            for (w0, wa, wb) in models:
                for zz in zs:
                    e = _relerr(zz, nodes, weights, w0, wa, wb)
                    if e > maxerr:
                        maxerr = e
                if maxerr > 1e-2:
                    return maxerr
            return maxerr
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_err:
            _update(np.concatenate([np.log(weights), res.x]), res.fun)
        return best_nodes

    # Try a few weight distributions with optimal nodes
    weight_candidates = [
        np.full(n, 1.0 / n),
        # Simpson-like
        np.array([1.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 1.0]) / 20.0,
        # Trapezoidal
        np.array([0.5, 1, 1, 1, 1, 1, 1, 0.5]) / 6.0,
        # Gaussian-ish
        np.array([1.0, 2.5, 3.5, 3.8, 3.8, 3.5, 2.5, 1.0]) / np.sum([1.0, 2.5, 3.5, 3.8, 3.8, 3.5, 2.5, 1.0]),
    ]
    for wc in weight_candidates:
        if time.time() > _DEADLINE - 10:
            break
        wc = wc / wc.sum()
        _opt_nodes(wc)

    # Full joint optimization from best
    if time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_v, args=(models, zs),
                           method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11})
            if res.fun < best_err:
                _update(res.x, res.fun)
        except Exception:
            pass

    # Final: enforce positivity and sum-to-1
    nodes = np.clip(best_nodes, 1e-6, 1.0 - 1e-6)
    weights = np.clip(best_weights, 1e-15, None)
    weights = weights / weights.sum()

    # Ensure strictly increasing
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-7

    if time.time() > _DEADLINE - 3:
        return _best_result()

    return {"nodes": nodes.astype(float), "weights": weights.astype(float)}
# EVOLVE-BLOCK-END
