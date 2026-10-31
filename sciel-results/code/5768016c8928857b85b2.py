"""Signed equal-width step functions minimizing max|autoconvolution| / (integral)^2.

Objective: score = max_k |c[k]| / (sum_j x[j])^2,  where c = x * x (autoconvolution).
Lower is better. We use signed (+/-1) step heights.

Strategy (structural + local refinement):
  1. Build a structured sign sequence with a dominant DC component (large integral)
     but suppressed side lobes, using a periodic sign pattern biased toward +1.
  2. Refine with a greedy coordinate-descent local search that flips individual
     signs to reduce the objective, with an explicit wall-clock deadline guard.
  3. Combine with multiple diverse initializations (random, cosine, chirp, block)
     and run local search on each, keeping the best result.
"""

import time
import numpy as np

N = 400
# Wall-clock budget with a safety margin below the 150 s kill limit.
DEADLINE_S = 135.0


def _autoconv(x):
    """Full autoconvolution c[k] = sum_j x[j] * x[j+k] for k in [0, N-1].
    Uses FFT for O(N log N) cost."""
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    c = np.fft.irfft(f * np.conj(f), n=2 * n)
    return c[:n]


def _score(x):
    """max |autoconvolution| / (integral)^2. Returns inf if integral ~ 0."""
    I = float(np.sum(x))
    if abs(I) < 1e-9:
        return float("inf")
    c = _autoconv(x)
    m = float(np.max(np.abs(c)))
    return m / (I * I)


def _initial_sequence():
    """Structured sign sequence: mostly +1 with periodic negative blocks to
    suppress side lobes while keeping a large positive integral."""
    x = np.ones(N, dtype=np.float64)
    x[::4] = -1.0
    return x


def _greedy_local_search(x, deadline):
    """Greedy coordinate descent: repeatedly flip the sign of the index whose
    flip most reduces the objective. Stops on no-improvement sweep or deadline."""
    x = x.copy()
    cur = _score(x)
    if not np.isfinite(cur):
        return x, cur
    n = len(x)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        best_i = -1
        best_val = cur
        for i in range(n):
            if time.time() > deadline:
                break
            x[i] = -x[i]
            v = _score(x)
            x[i] = -x[i]
            if v < best_val - 1e-12:
                best_val = v
                best_i = i
        if best_i >= 0:
            x[best_i] = -x[best_i]
            cur = best_val
            improved = True
    return x, cur


def _diverse_initializations(rng):
    """Generate diverse initial sign sequences."""
    inits = []
    
    # 1. Structured: mostly +1 with periodic negatives
    x = _initial_sequence()
    inits.append(x)
    
    # 2. Random +/-1
    inits.append(rng.choice([-1.0, 1.0], size=N))
    
    # 3. Low frequency cosine patterns
    for k in [1, 2, 3, 5, 7]:
        t = np.arange(N)
        h = np.sign(np.cos(2 * np.pi * k * t / N))
        h[h == 0] = 1.0
        inits.append(h)
    
    # 4. Chirp-like patterns
    for freq in [0.5, 1.0, 2.0]:
        t = np.arange(N)
        h = np.sign(np.cos(2 * np.pi * freq * t * t / (2 * N)))
        h[h == 0] = 1.0
        inits.append(h)
    
    # 5. Block structures
    for _ in range(3):
        h = np.ones(N)
        pos = 0
        while pos < N:
            block_len = rng.randint(3, 15)
            sign = rng.choice([-1.0, 1.0])
            h[pos:pos + block_len] = sign
            pos += block_len
        inits.append(h)
    
    # 6. Mostly +1 with sparse random negatives
    for n_neg in [N // 8, N // 6, N // 5]:
        h = np.ones(N)
        neg_idx = rng.choice(N, n_neg, replace=False)
        h[neg_idx] = -1.0
        inits.append(h)
    
    # 7. Periodic patterns with different periods
    for period in [3, 5, 7, 11]:
        pattern = []
        for i in range(period):
            # Bias toward +1
            if i < period - 1:
                pattern.append(1.0)
            else:
                pattern.append(-1.0)
        h = np.tile(pattern, N // period + 1)[:N]
        inits.append(h)
    
    return inits


def construct():
    start = time.time()
    deadline = start + DEADLINE_S
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = _diverse_initializations(rng)
    
    # Sort by initial score to prioritize promising starts
    scored_inits = []
    for x in inits:
        s = _score(x)
        if np.isfinite(s):
            scored_inits.append((s, x))
    
    scored_inits.sort(key=lambda t: t[0])
    
    best_x = np.ones(N, dtype=np.float64)
    best_val = _score(best_x)
    
    # Refine each initialization with local search, keeping the best
    for init_score, x in scored_inits:
        if time.time() > deadline:
            break
        
        x_opt, val_opt = _greedy_local_search(x, deadline)
        
        if np.isfinite(val_opt) and val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
    
    # Safety: ensure a finite, valid result is always returned.
    if not np.isfinite(best_val):
        best_x = np.ones(N, dtype=np.float64)
        best_val = _score(best_x)
    
    return best_x


if __name__ == "__main__":
    res = construct()
    print("score:", _score(res))
