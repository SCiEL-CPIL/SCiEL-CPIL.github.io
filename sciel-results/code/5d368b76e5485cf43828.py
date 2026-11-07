import numpy as np
from scipy.optimize import minimize

N = 50

def compute_value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate([[0.0], g, [0.0]])
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    if norm_1 * norm_inf < 1e-15:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def negative_value(h):
    h = np.maximum(h, 0.0)
    return -compute_value(h)

def construct():
    best_h = None
    best_val = -np.inf

    initial_guesses = [
        np.ones(N),
        np.exp(-((np.arange(N) - 24.5) ** 2) / (2 * 10.0 ** 2)),
        np.exp(-((np.arange(N) - 24.5) ** 2) / (2 * 15.0 ** 2)),
        np.maximum(0.0, 1.0 - ((np.arange(N) - 24.5) / 25.0) ** 2),
    ]

    for x0 in initial_guesses:
        x0 = np.maximum(x0, 0.0)
        try:
            result = minimize(negative_value, x0, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            h = np.maximum(result.x, 0.0)
            val = compute_value(h)
            if val > best_val:
                best_val = val
                best_h = h
        except Exception:
            continue

    if best_h is None:
        best_h = np.ones(N)

    return list(best_h)
