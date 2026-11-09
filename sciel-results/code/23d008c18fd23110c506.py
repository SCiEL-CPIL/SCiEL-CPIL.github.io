"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_sidelobe_max(h):
    """Compute max |R[k]| for k != 0 using FFT-based linear autocorrelation."""
    n = len(h)
    nfft = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, n=nfft)
    R = np.fft.irfft(H * np.conj(H), n=nfft)
    # Linear autocorrelation: lags 0 to n-1 are in R[0:n], lags -(n-1) to -1 are in R[nfft-n+1:nfft]
    sidelobes_pos = R[1:n]
    sidelobes_neg = R[nfft - n + 1:nfft]
    all_sidelobes = np.concatenate([sidelobes_pos, sidelobes_neg])
    return np.max(np.abs(all_sidelobes))


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Use a deterministic pseudo-random initialization to break symmetry
    # Thue-Morse sequence or simple LCG-based random bits
    rng = np.random.RandomState(42)
    h = 2 * rng.randint(0, 2, size=N).astype(np.float64) - 1  # +/-1
    best_h = h.copy()
    best_score = _autocorr_sidelobe_max(best_h)

    # Greedy local search with occasional random restarts
    # We'll do multiple passes of single-bit flips, then try 2-bit flips
    improved = True
    pass_count = 0
    max_passes = 50

    while improved and pass_count < max_passes and (time.time() - start_time) < deadline - 10:
        improved = False
        pass_count += 1
        # Shuffle the order of flips to avoid bias
        order = np.random.permutation(N)
        for i in order:
            if (time.time() - start_time) > deadline - 10:
                break
            h[i] *= -1
            new_score = _autocorr_sidelobe_max(h)
            if new_score < best_score - 1e-12:
                best_score = new_score
                best_h = h.copy()
                improved = True
            else:
                h[i] *= -1  # revert

        # If no improvement in this pass, try a random restart with a new seed
        if not improved:
            if (time.time() - start_time) < deadline - 20:
                # Random restart: flip a random subset of bits
                num_flips = rng.randint(N // 4, N // 2)
                flip_indices = rng.choice(N, size=num_flips, replace=False)
                h[flip_indices] *= -1
                new_score = _autocorr_sidelobe_max(h)
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
                    improved = True  # Continue searching from this new state
                else:
                    # Revert and try a different perturbation
                    h[flip_indices] *= -1
                    # Try smaller perturbation
                    num_flips = rng.randint(5, 20)
                    flip_indices = rng.choice(N, size=num_flips, replace=False)
                    h[flip_indices] *= -1
                    new_score = _autocorr_sidelobe_max(h)
                    if new_score < best_score:
                        best_score = new_score
                        best_h = h.copy()
                        improved = True
                    else:
                        h[flip_indices] *= -1
                        # If still no improvement, accept the current best and stop
                        break

    return best_h
# EVOLVE-BLOCK-END
