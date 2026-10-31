"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_abs_max(x):
    c = np.correlate(x, x, mode='full')
    return np.max(np.abs(c))


def _score_of(x):
    integral = np.sum(x)
    if integral == 0:
        return np.inf
    return _autocorr_abs_max(x) / (integral * integral)


def _square_wave(p_len, n_len):
    """Construct a sequence of length p_len positive then n_len negative, padded with zeros."""
    if p_len + n_len >= N:
        # No padding
        x = np.zeros(N)
        x[:p_len] = 1.0
        x[p_len:p_len + n_len] = -1.0
        return x
    # Center the block
    start = (N - p_len - n_len) // 2
    x = np.zeros(N)
    x[start:start + p_len] = 1.0
    x[start + p_len:start + p_len + n_len] = -1.0
    return x


def construct():
    t_start = time.time()
    deadline = t_start + 120  # 120s wall-clock limit

    best_score = np.inf
    best_x = np.ones(N)

    # Try a grid of (p_len, n_len) pairs
    # p_len and n_len should be roughly balanced for low ratio
    # Search over a range of widths
    for p_len in range(1, N + 1):
        if time.time() > deadline - 5:
            break
        for n_len in range(1, N - p_len + 1):
            if p_len + n_len < 2:
                continue
            x = _square_wave(p_len, n_len)
            s = _score_of(x)
            if s < best_score:
                best_score = s
                best_x = x.copy()

    # Also try constant sequence as baseline
    s_const = _score_of(np.ones(N))
    if s_const < best_score:
        best_score = s_const
        best_x = np.ones(N)

    # If we still have time, do a finer search around the best
    if time.time() < deadline - 10:
        p_best = 0
        n_best = 0
        # Extract best p, n from best_x (not directly possible, so just do a random restart)
        rng = np.random.default_rng(42)
        best_p, best_n = 0, 0
        
        # Try to infer p and n from best_x if it's not constant
        pos = np.sum(best_x > 0)
        neg = np.sum(best_x < 0)
        if pos > 0 and neg > 0:
            best_p, best_n = int(pos), int(neg)
        else:
            # Start around N/2, N/2
            best_p, best_n = N // 2, N // 2

        # Local search around best
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            p_try = max(1, min(N - 1, best_p + rng.integers(-5, 6)))
            n_try = max(1, min(N - 1, best_n + rng.integers(-5, 6)))
            x = _square_wave(p_try, n_try)
            s = _score_of(x)
            if s < best_score:
                best_score = s
                best_x = x.copy()
                best_p, best_n = p_try, n_try

    return best_x
# EVOLVE-BLOCK-END
