"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    dx = 1.0 / (2 * N)
    deadline = time.time() + 140

    # Start with alternating ±1 pattern
    h = np.ones(N, dtype=np.float64)
    h[1::2] = -1.0

    def peak_autocorr_nonzero(h):
        """Max |autocorr| at non-zero lags, scaled by dx."""
        hf = h.astype(np.float64)
        ac = np.correlate(hf, hf, mode='full')
        n = len(hf)
        # ac[n-1] is lag 0, ac[n:] are positive lags 1..n-1
        positive = ac[n:]
        return dx * np.max(np.abs(positive))

    best_score = peak_autocorr_nonzero(h)
    best_h = h.copy()

    rng = np.random.RandomState(42)
    iter_count = 0

    while time.time() < deadline:
        iter_count += 1

        # Flip a random position
        pos = rng.randint(N)
        old = h[pos]
        h[pos] = -old

        score = peak_autocorr_nonzero(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
        else:
            h[pos] = old

        # Occasionally try multi-flips
        if iter_count % 500 == 0:
            for _ in range(3):
                k = rng.randint(2, 6)
                positions = rng.choice(N, size=k, replace=False)
                saved = h[positions].copy()
                h[positions] *= -1
                score = peak_autocorr_nonzero(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
                else:
                    h[positions] = saved

    return best_h
# EVOLVE-BLOCK-END
