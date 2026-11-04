"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_peak(s):
    """Peak of |linear autocorrelation| of s (±1, length N), excluding lag 0."""
    n = len(s)
    if n <= 1:
        return 0.0
    size = 2 * n
    S = np.fft.rfft(s, size)
    ac = np.fft.irfft(S * np.conj(S), size)
    return float(np.max(np.abs(ac[1:n])))


def _thue_morse(n):
    """Thue-Morse ±1 sequence of length n (0->+1, 1->-1)."""
    t = np.fromiter((bin(i).count('1') & 1 for i in range(n)), dtype=np.int8, count=n)
    return (1 - 2 * t).astype(np.float64)


def construct():
    start = time.time()
    deadline = start + 100.0  # explicit deadline guard (safety margin)

    best_s = None
    best_peak = np.inf

    def update_best(s, peak):
        nonlocal best_s, best_peak
        if peak < best_peak:
            best_peak = peak
            best_s = s.copy()

    rng = np.random.default_rng(2024)

    # Candidate initial sequences: Thue-Morse + random restarts
    inits = [_thue_morse(N)]
    for _ in range(3):
        inits.append(rng.choice([-1.0, 1.0], size=N))

    for s0 in inits:
        if time.time() > deadline - 5.0:
            break
        s = s0.copy()
        peak = _autocorr_peak(s)
        update_best(s, peak)

        # Simulated annealing to minimize the peak off-peak autocorrelation
        T = 5.0
        T_min = 0.05
        for _ in range(4000):
            if time.time() > deadline - 2.0:
                break
            i = int(rng.integers(N))
            old_peak = peak
            s[i] = -s[i]
            new_peak = _autocorr_peak(s)
            if new_peak < old_peak or (rng.random() < np.exp(-(new_peak - old_peak) / T)):
                peak = new_peak
                update_best(s, peak)
            else:
                s[i] = -s[i]  # revert
            T = max(T_min, T * 0.9995)

        # Quick steepest-descent polish: try each flip, keep the best single flip
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            best_i = -1
            best_new_peak = peak
            for i in range(N):
                s[i] = -s[i]
                np_ = _autocorr_peak(s)
                if np_ < best_new_peak:
                    best_new_peak = np_
                    best_i = i
                s[i] = -s[i]
            if best_i >= 0 and best_new_peak < peak:
                s[best_i] = -s[best_i]
                peak = best_new_peak
                update_best(s, peak)
            else:
                break

    if best_s is None:
        return np.ones(N)
    return best_s
# EVOLVE-BLOCK-END
