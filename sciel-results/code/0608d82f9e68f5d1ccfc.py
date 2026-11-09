import numpy as np
from scipy import optimize


def _compute_area(x, M=20000):
    n = 64
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n
    a = (np.arange(n) + 1.0) / n
    L = x[:, None] + a[:, None] * y[None, :]
    L_sorted = np.sort(L, axis=0)
    gaps = np.diff(L_sorted, axis=0)
    union_len = w + np.sum(np.minimum(gaps, w[None, :]), axis=0)
    return np.mean(union_len)


def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    best_x = None
    best_area = np.inf

    rng = np.random.default_rng(42)

    for trial in range(8):
        if trial == 0:
            x0 = np.zeros(n)
        else:
            if best_x is not None:
                x0 = best_x + rng.normal(0, 0.05, n)
            else:
                x0 = rng.uniform(-0.5, 0.5, n)

        res = optimize.minimize(
            lambda x: _compute_area(x, M=2000),
            x0,
            method='Powell',
            options={'maxiter': 1000, 'ftol': 1e-9, 'xtol': 1e-7}
        )

        area_full = _compute_area(res.x, M=20000)
        if area_full < best_area:
            best_area = area_full
            best_x = res.x.copy()

    return best_x.astype(np.float64)
