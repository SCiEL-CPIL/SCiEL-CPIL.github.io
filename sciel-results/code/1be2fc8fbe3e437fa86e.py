# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 620.0  # safety margin before the 630 s wall-clock kill


def _area_from_positions(x):
    """Union area of the 64 triangles via the exact quadrature spec.

    At each height y_m = (m + 0.5)/20000 the cross-section of triangle i is the
    interval [L_i, L_i + w] with L_i = x_i + a_i*y_m and w = (1 - y_m)/64.
    With L sorted, union length = w + sum_j min(L_{j+1} - L_j, w).
    The area is the mean union length over the 20000 heights.
    """
    n = 64
    a = (np.arange(1, n + 1, dtype=np.float64)) / n
    y = (np.arange(1, 20000, dtype=np.float64)) / 20000.0  # (m + 0.5)/20000
    w = (1.0 - y) / n                                      # (20000,)
    L = x[None, :] + y[:, None] * a[None, :]              # (20000, 64)
    order = np.argsort(L, axis=1)
    Ls = np.take_along_axis(L, order, axis=1)             # (20000, 64)
    gaps = Ls[:, 1:] - Ls[:, :-1]
    union = w[:, None] + np.minimum(gaps, w[:, None]).sum(axis=1)
    return float(union.mean())


def _keich_start():
    """Classical Keich construction (valid for n a power of 2)."""
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bits = np.array(list(str(bin(i))[2:]), dtype=np.int64)
        bits = np.pad(bits, (k - len(bits), 0), "constant")
        out[i] = np.sum((1.0 - j) * bits * (2.0 ** -j) / k)
    return out


def _polish(x0, area0, budget):
    """High-precision polish: very tight tolerance, more max iterations."""
    from scipy.optimize import minimize
    x0 = np.asarray(x0, dtype=np.float64).copy()
    res = minimize(
        _area_from_positions,
        x0,
        method="L-BFGS-B",
        options={"maxiter": 4000, "ftol": 1e-18, "gtol": 1e-12,
                 "maxls": 100, "maxcor": 40},
    )
    if res.success or res.fun < area0:
        x = np.clip(res.x, -5.0, 5.0)
        a = _area_from_positions(x)
        if np.isfinite(a) and a > 0 and a < area0:
            return x, a
    return x0, area0


def kakeya_needle_2d_positions64() -> np.ndarray:
    """Return 64 base positions x_0..x_63 minimizing the union area of the
    Kakeya-needle triangles (n = 64).

    Strategy: start from the Keich construction, then run a smooth L-BFGS-B
    optimization followed by a high-precision polish pass. A deadline guard
    guarantees a valid (finite) result is always returned.
    """
    x = _keich_start()
    area = _area_from_positions(x)

    # Primary smooth optimization from the Keich start.
    try:
        from scipy.optimize import minimize
        res = minimize(
            _area_from_positions,
            x.copy(),
            method="L-BFGS-B",
            options={"maxiter": 3000, "ftol": 1e-15, "gtol": 1e-10,
                     "maxls": 100, "maxcor": 40},
        )
        if res.fun < area:
            x = np.clip(res.x, -5.0, 5.0)
            area = _area_from_positions(x)
    except Exception:
        pass

    # High-precision polish pass (tighter tolerance, more iterations).
    if time.time() < _DEADLINE - 15.0:
        try:
            x, area = _polish(x, area, 15.0)
        except Exception:
            pass

    # Final safety: ensure a finite, valid result.
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        x = _keich_start()
    return x


# EVOLVE-BLOCK-END
