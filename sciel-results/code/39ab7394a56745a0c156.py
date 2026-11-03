"""C3-prime: signed equal-width step functions minimizing max|autoconv| / (integral)^2."""
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np


def _autocorr_circ(x):
    n = x.size
    f = np.fft.rfft(x)
    return np.fft.irfft(f * np.conj(f))[:n]


def _score(x):
    ac = _autocorr_circ(x)
    peak = np.max(np.abs(ac))
    integral = float(np.sum(x))
    return peak / (integral * integral)


def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin vs 150s wall-clock

    # Random sign initialization: peak ~ sqrt(N), integral ~ N -> value ~ 1/sqrt(N),
    # far below the all-ones baseline of 1/N.
    best = np.random.choice([-1.0, 1.0], size=N)
    best_val = _score(best)

    # Local single-flip polishing: try flipping one sign; accept if it lowers the
    # score and keeps the integral safely away from zero.
    idx = np.arange(N)
    while time.time() < deadline:
        improved = False
        for i in idx:
            if time.time() >= deadline:
                break
            x = best.copy()
            x[i] *= -1.0
            integral = float(np.sum(x))
            if abs(integral) < 8.0:      # guard against integral collapse
                continue
            val = _score(x)
            if val < best_val:
                best = x
                best_val = val
                improved = True
        if not improved:
            # Local optimum reached; perturb to escape and continue polishing.
            k = np.random.randint(1, 13)
            flips = np.random.choice(N, size=k, replace=False)
            best[flips] *= -1.0
            best_val = _score(best)
            if abs(float(np.sum(best))) < 8.0:
                # integral too small; revert perturb and restart with fresh random
                best = np.random.choice([-1.0, 1.0], size=N)
                best_val = _score(best)

    return best
# EVOLVE-BLOCK-END
