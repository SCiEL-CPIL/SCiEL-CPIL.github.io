"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _peak_autocorr(f):
    """Compute the peak absolute linear autocorrelation using FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * np.conj(F), size)[:n]
    return float(np.max(np.abs(ac)))


def construct():
    start_time = time.time()
    deadline = start_time + 145

    rng = np.random.default_rng(777)
    best_f = rng.choice([-1, 1], size=N).astype(float)
    best_val = _peak_autocorr(best_f)

    while time.time() < deadline:
        improved = False
        best_flip_idx = -1
        best_flip_val = best_val

        for i in range(N):
            if time.time() > deadline:
                break
            old_val = best_f[i]
            best_f[i] = -old_val
            val = _peak_autocorr(best_f)
            if val < best_flip_val:
                best_flip_val = val
                best_flip_idx = i
            best_f[i] = old_val

        if best_flip_idx >= 0:
            best_f[best_flip_idx] = -best_f[best_flip_idx]
            best_val = best_flip_val
            improved = True

        if not improved:
            for _ in range(10):
                if time.time() > deadline:
                    break
                idx = int(rng.integers(0, N))
                best_f[idx] = -best_f[idx]
                val = _peak_autocorr(best_f)
                if val < best_val:
                    best_val = val
                    improved = True
                else:
                    best_f[idx] = -best_f[idx]

            if not improved:
                new_f = rng.choice([-1, 1], size=N).astype(float)
                new_val = _peak_autocorr(new_f)
                if new_val < best_val:
                    best_f = new_f
                    best_val = new_val

    return best_f
# EVOLVE-BLOCK-END
