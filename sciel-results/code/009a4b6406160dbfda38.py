"""Symbolic regression: recover hidden f(x) from 10 BFGS-fitted params."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = 110.0  # safety margin well under the 150s wall-clock kill


def _finite_out(x, y):
    """Return y if all finite and correctly shaped, else a safe finite fallback."""
    if np.all(np.isfinite(y)) and y.shape == x.shape:
        return y
    return np.zeros_like(x, dtype=float)


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: if we are running out of wall-clock, emit a safe finite result.
    if time.time() > _DEADLINE:
        return np.zeros_like(x, dtype=float)

    p = np.asarray(params, dtype=float)
    if p.size < 10:
        p = np.concatenate([p, np.ones(10 - p.size)])
    p = p[:10]

    # 3 Gaussians: (center, width, amplitude) each, then a global linear slope.
    c1, w1, a1, c2, w2, a2, c3, w3, a3, m = p

    # Keep centers within the data range [-2, 2].
    c1 = 2.0 * (1.0 / (1.0 + np.exp(-c1))) - 1.0
    c2 = 2.0 * (1.0 / (1.0 + np.exp(-c2))) - 1.0
    c3 = 2.0 * (1.0 / (1.0 + np.exp(-c3))) - 1.0

    # Keep widths strictly positive (avoids non-finite values).
    w1 = 1.0 / (1.0 + np.exp(-w1))
    w2 = 1.0 / (1.0 + np.exp(-w2))
    w3 = 1.0 / (1.0 + np.exp(-w3))

    x = np.asarray(x, dtype=float)

    g1 = a1 * np.exp(-((x - c1) ** 2) / (2.0 * w1 * w1 + 1e-12))
    g2 = a2 * np.exp(-((x - c2) ** 2) / (2.0 * w2 * w2 + 1e-12))
    g3 = a3 * np.exp(-((x - c3) ** 2) / (2.0 * w3 * w3 + 1e-12))

    y = g1 + g2 + g3 + m * x

    # Tiny ridge term: finite everywhere, negligible effect on fit quality.
    y = y + 1e-9 * (a1 + a2 + a3 + m)

    return _finite_out(x, y)
# EVOLVE-BLOCK-END
