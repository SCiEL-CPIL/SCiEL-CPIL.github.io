import numpy as np
from scipy.special import roots_genlaguerre
import time

def construct():
    """
    Returns a list of 12 positive double-root locations for the Laguerre combination.
    Uses generalized Laguerre zeros with large alpha for near-uniform spacing,
    scaled to large magnitude to minimize C4 upper bound.
    """
    t0 = time.time()
    deadline = 580.0

    m = 12

    # Find the alpha that gives the most uniform spacing (maximizes min gap / spread)
    best_alpha = 11
    best_min_gap_ratio = 0.0

    for alpha in [8, 10, 11, 12, 14, 16, 20, 30, 50, 100, 200]:
        if time.time() - t0 > deadline - 20:
            break
        try:
            zeros = roots_genlaguerre(m, alpha)[0]
            zeros = np.sort(zeros)
            if np.any(zeros <= 0):
                continue
            gaps = np.diff(zeros)
            min_gap = np.min(gaps)
            spread = zeros[-1] - zeros[0]
            if spread > 0:
                ratio = min_gap / spread
                if ratio > best_min_gap_ratio:
                    best_min_gap_ratio = ratio
                    best_alpha = alpha
        except Exception:
            continue

    # Get zeros with best alpha
    try:
        zeros = roots_genlaguerre(m, best_alpha)[0]
        zeros = np.sort(zeros)
    except Exception:
        zeros = np.linspace(1, 12, m)

    # Scale to large magnitude
    # Larger roots -> smaller C4 bound
    target_max = 100000.0
    scale = target_max / zeros[-1]
    scaled = zeros * scale

    return list(scaled)
