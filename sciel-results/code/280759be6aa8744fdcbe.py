"""Third autocorrelation inequality — coordinated block-flip search.

Strategy:
  * Fast FFT scorer: C3[k] = sum_i h[i] h[i+k] h[i+2k] (lag k), score = max_k |C3[k]|.
  * Strong search = simulated annealing over COORDINATED block flips (size 4..64).
    Block flips escape the local minima that trap single-bit coordinate descent.
  * A fast incremental score update (only the touched lags are recomputed) lets us
    afford far more moves than a full-FFT-per-move loop, so we can afford a longer,
    more aggressive anneal.
"""
import numpy as np
import time

N = 400
KMAX = (N - 1) // 2          # 199 : largest lag with at least one valid term
DEADLINE = 112.0             # safety margin below the 120 s / 150 s limits


def _c3_lag(h, k):
    """C3[k] = sum_i h[i] h[i+k] h[i+2k] over the valid i (0..N-1-2k)."""
    m = N - 2 * k
    return np.dot(h[:m], h[k:k + m] * h[2 * k:2 * k + m])


def _compute_c3(h):
    """All C3[k], k = 0..KMAX, via two FFTs (linear convolution)."""
    h = np.asarray(h, dtype=np.float64)
    size = 2 * N + 1
    H = np.fft.rfft(h, n=size)
    HH = H * H
    C3 = np.fft.irfft(HH * H, n=size)[:N]      # linear conv of h,h,h -> lags 0..N-1
    return C3[:KMAX + 1]


def _score(h):
    """The evaluator's raw value: max_k |C3[k]| (lower is better)."""
    return float(np.max(np.abs(_compute_c3(h))))


def _init_random(h, rng):
    """Random ±1. We allow a small imbalance to avoid sum=0 rejection."""
    h[:] = np.where(rng.random(N) < 0.5, 1.0, -1.0)
    # Ensure sum is not zero. If it is, flip one element.
    s = np.sum(h)
    if s == 0:
        # Flip a random element to make sum +/- 2
        idx = int(rng.integers(0, N))
        h[idx] *= -1.0


def _random_block(rng, size):
    """Return (start, length) of a random block, size chosen from a spread."""
    # spread block sizes: small (fine) up to large (coordinated)
    r = rng.random()
    if r < 0.30:
        L = int(rng.integers(4, 9))        # 4..8
    elif r < 0.60:
        L = int(rng.integers(9, 17))       # 9..16
    elif r < 0.85:
        L = int(rng.integers(17, 33))      # 17..32
    else:
        L = int(rng.integers(33, 65))      # 33..64
    L = min(L, N)
    start = int(rng.integers(0, N - L + 1))
    return start, L


def _flip_block(h, start, L):
    """Flip a block; return the list of lags k whose C3[k] changed."""
    h[start:start + L] *= -1.0
    a, b = start, start + L
    lo = max(0, b - N)
    hi = min(KMAX, (a + N - 1) // 2)
    return list(range(lo, hi + 1))


def _hill_climb(h, C3, rng, t0, deadline):
    """Greedy Hill Climbing with random block flips. Accepts only improvements."""
    best_h = h.copy()
    best_score = float(np.max(np.abs(C3)))
    cur_score = best_score
    
    while time.time() < deadline:
        start, L = _random_block(rng, 0)
        changed = _flip_block(h, start, L)
        for k in changed:
            C3[k] = _c3_lag(h, k)
        s = float(np.max(np.abs(C3)))

        if s < cur_score:
            cur_score = s
            if s < best_score:
                best_score = s
                best_h = h.copy()
        else:
            # Reject: undo the flip and restore the lags
            h[start:start + L] *= -1.0
            for k in changed:
                C3[k] = _c3_lag(h, k)
                
    return best_h, best_score


def construct():
    """Construct a signed step function (length N, ±1) minimising max|C3|."""
    t0 = time.time()
    deadline = t0 + DEADLINE
    rng = np.random.default_rng(12345)

    best_h = np.ones(N)
    best_score = _score(best_h)

    # Use Iterated Local Search structure: Multiple restarts of Hill Climbing
    # This is structurally different from the single long SA run.
    
    # Phase 1: Try diverse seeds with short Hill Climbs
    seeds = [best_h.copy()]
    for f in (1, 2, 3, 5, 8, 13, 20, 34, 50):
        x = np.linspace(0.0, 1.0, N, endpoint=False)
        s = np.sign(np.sin(2 * np.pi * f * x))
        # Ensure non-zero sum for seed
        if np.sum(s) == 0:
            s[0] *= -1
        seeds.append(s)
    
    # Add random seeds
    for _ in range(10):
        r = np.where(rng.random(N) < 0.5, 1.0, -1.0)
        if np.sum(r) == 0:
            r[0] *= -1
        seeds.append(r)

    for seed in seeds:
        if time.time() >= deadline:
            break
        h = seed.copy().astype(np.float64)
        C3 = _compute_c3(h)
        # Short hill climb from this seed
        bh, bs = _hill_climb(h, C3, rng, t0, min(deadline, t0 + 0.5))
        if bs < best_score:
            best_score = bs
            best_h = bh.copy()

    # Phase 2: Long Hill Climb from the best found so far
    if time.time() < deadline:
        h = best_h.copy().astype(np.float64)
        C3 = _compute_c3(h)
        bh, bs = _hill_climb(h, C3, rng, t0, deadline)
        if bs < best_score:
            best_score = bs
            best_h = bh.copy()

    # Final Safety Check: Ensure sum is not zero
    out = np.sign(best_h).astype(np.float64)
    if np.sum(out) == 0:
        # Flip the first element to make sum +/- 2
        out[0] *= -1.0
        # Re-score if we flipped, though it's a tiny change
        # We don't need to re-run the full score for the return value, 
        # but the evaluator will check it. The change is minimal.
        
    if np.all(out == 0):
        out = np.ones(N)
    return out


if __name__ == "__main__":
    h = construct()
    print("Constructed (first 20):", h[:20])
    print("Score:", _score(h))
