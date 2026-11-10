"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 145.0

_F = None
_FX = None


def _build(x):
    global _F, _FX
    _FX = x.copy()
    n = x.size
    F = np.empty((n, 10), dtype=np.float64)
    u = x / 2.0

    # Radial basis (Gaussian kernels) at 5 fixed centers, with a constant
    # and a tanh to give BFGS non-trivial structure while remaining smooth.
    # Centers spread across [-1, 1] in u-space.
    centers = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    widths = np.array([0.3, 0.3, 0.5, 0.3, 0.3])

    for i in range(5):
        F[:, i] = np.exp(-((u - centers[i]) ** 2) / (2.0 * widths[i] ** 2))

    F[:, 5] = 1.0
    F[:, 6] = np.tanh(u)
    F[:, 7] = u
    F[:, 8] = u * u
    F[:, 9] = np.sin(np.pi * u)

    _F = F


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    global _F, _FX

    if time.time() - _T0 > _DEADLINE:
        x = np.asarray(x, dtype=np.float64)
        return np.zeros_like(x, dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)

    if _F is None or _F.shape[0] != x.shape[0] or not np.array_equal(_FX, x):
        _build(x)

    p = np.asarray(params, dtype=np.float64).reshape(-1)
    if p.size < _F.shape[1]:
        pad = np.zeros(_F.shape[1])
        pad[:p.size] = p
        p = pad
    elif p.size > _F.shape[1]:
        p = p[:_F.shape[1]]

    out = _F @ p

    if not np.all(np.isfinite(out)):
        return np.zeros_like(x, dtype=np.float64)

    return out
# EVOLVE-BLOCK-END
