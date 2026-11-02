"""Rewritten: FFT scoring + greedy Golomb seed + local search polish."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import combinations

_START = time.time()

def _elapsed():
    return time.time() - _START

def _score_set_fft(U, _cache={}):
    """Fast score via FFT. U is a sorted list of distinct nonneg ints containing 0."""
    U_arr = np.asarray(U, dtype=np.int64)
    n = len(U_arr)
    if n == 0:
        return 0.0
    max_u = int(U_arr[-1])
    if max_u == 0:
        return 1.0
    if n == 1:
        return 1.0

    # Sumset via convolution of indicator vector
    # size of indicator = max_u+1, conv length = 2*max_u+1
    size = max_u + 1
    # next power of 2 for FFT
    conv_len = 2 * size
    N = 1
    while N < conv_len:
        N <<= 1

    f = np.zeros(N, dtype=np.float64)
    f[U_arr] = 1.0
    F = np.fft.rfft(f)
    F2 = F * F
    conv = np.fft.irfft(F2, N)
    # conv[k] = number of pairs (a,b) with a+b=k, for k in [0, 2*max_u]
    sumset_size = int(np.sum(conv[:2 * max_u + 1] > 0.5))

    # Difference set via autocorrelation: conv of f with reversed f
    # diff count at d = sum_k f[k] f[k+d]  -> autocorrelation
    # Use FFT: autocorr = ifft( |F|^2 ) but that's circular; we need linear.
    # Better: compute diffs directly for moderate n, else use FFT on padded.
    # For difference set, |U-U| = number of distinct (a-b). Range [-max_u, max_u].
    # Use convolution of f with reversed f.
    rev = np.zeros(N, dtype=np.float64)
    rev[:size] = f[size - 1::-1]  # reverse of f[0..size-1]
    R = np.fft.rfft(rev)
    F2d = F * R
    ac = np.fft.irfft(F2d, N)
    # ac[i] for i in [0, size-1] corresponds to correlation at lag (size-1 - i)
    # lag d = (size-1) - i, so i = (size-1) - d, d in [-(size-1), size-1]
    # ac[i] = sum_k f[k] f[k+d]  (count of pairs with difference d)
    # We want distinct d where count>0, d in [-max_u, max_u]
    lo = (size - 1) - max_u
    hi = (size - 1) + max_u
    ac_seg = ac[lo:hi + 1]
    diffset_size = int(np.sum(ac_seg > 0.5))

    if sumset_size == 0:
        return 0.0
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set(U):
    return _score_set_fft(U)

def _greedy_golomb(start_n=4, max_n=20, max_val=4000, seed=0):
    """Greedy construction: start with [0], repeatedly add the value that
    maximizes incremental score, preferring Golomb (distinct diffs)."""
    rng = random.Random(seed)
    U = [0]
    s = _score_set_fft(U)
    for _ in range(start_n - 1):
        # add a small value to seed
        U.append(1 if 1 not in U else 2)
    best = U[:]
    best_s = s
    for _ in range(max_n - len(U)):
        improved = False
        # candidate new values: Golomb-optimal next + range
        max_u = max(U)
        # Golomb next: smallest value with all new diffs distinct
        existing_diffs = set()
        for i in range(len(U)):
            for j in range(len(U)):
                if i != j:
                    existing_diffs.add(U[i] - U[j])
        # try candidates from max_u+1 upward and some below
        candidates = []
        for v in range(1, max_u + 200):
            if v not in U:
                candidates.append(v)
        # add some random far values
        for _ in range(30):
            candidates.append(rng.randint(1, max_u + 300))
        best_cand = None
        best_cs = best_s
        for v in candidates:
            new_U = sorted(U + [v])
            cs = _score_set_fft(new_U)
            if cs > best_cs + 1e-9:
                best_cs = cs
                best_cand = v
        if best_cand is not None:
            U = sorted(U + [best_cand])
            best = U[:]
            best_s = best_cs
            improved = True
        else:
            break
    return best, best_s

def _local_search(U, max_iters=400, deadline_margin=20, polish=False):
    """Local search: move/add/remove one element. FFT scoring."""
    U = sorted(set(U))
    best_score = _score_set_fft(U)
    best_U = U[:]
    max_u = max(U)
    upper = max_u + 60
    tol = 1e-10 if polish else 1e-9

    for it in range(max_iters):
        if _elapsed() > (150 - deadline_margin):
            break
        improved = False
        n = len(U)
        # Move one element
        for i in range(n):
            old_val = U[i]
            if old_val == 0:
                continue
            # candidate new values
            cands = list(range(1, upper + 1))
            # add some jumps
            for _ in range(8):
                cands.append(random.randint(1, upper + 80))
            best_new = None
            best_ns = best_score
            for nv in cands:
                if nv == old_val:
                    continue
                new_U = U[:]
                new_U[i] = nv
                if len(set(new_U)) != len(new_U):
                    continue
                ns = _score_set_fft(new_U)
                if ns > best_ns + tol:
                    best_ns = ns
                    best_new = nv
            if best_new is not None:
                U = U[:]
                U[i] = best_new
                U = sorted(set(U))
                best_score = best_ns
                best_U = U[:]
                upper = max(best_U) + 60
                improved = True
                break
        # Remove an element
        if not improved and n > 3:
            best_rm = None
            best_rs = best_score
            for i in range(1, n):
                new_U = U[:i] + U[i + 1:]
                rs = _score_set_fft(new_U)
                if rs > best_rs + tol:
                    best_rs = rs
                    best_rm = i
            if best_rm is not None:
                U = U[:best_rm] + U[best_rm + 1:]
                best_score = best_rs
                best_U = U[:]
                upper = max(best_U) + 60
                improved = True
        # Add an element
        if not improved and n < 25:
            best_add = None
            best_as = best_score
            for nv in range(1, upper + 30):
                if nv in U:
                    continue
                new_U = U + [nv]
                as_ = _score_set_fft(new_U)
                if as_ > best_as + tol:
                    best_as = as_
                    best_add = nv
            if best_add is not None:
                U = U + [best_add]
                U = sorted(set(U))
                best_score = best_as
                best_U = U[:]
                upper = max(best_U) + 60
                improved = True
        if not improved:
            break
    return sorted(set(best_U)), best_score

def construct():
    deadline_margin = 25
    def _timeup():
        return _elapsed() > (150 - deadline_margin)

    best_U = [0, 1, 3]
    best_score = _score_set_fft(best_U)

    # 1. Greedy Golomb construction (high-quality seed)
    for seed in (0, 1, 7, 42):
        if _timeup():
            break
        try:
            U_g, s_g = _greedy_golomb(start_n=4, max_n=18, max_val=4000, seed=seed)
            if s_g > best_score:
                best_score = s_g
                best_U = U_g
        except Exception:
            pass

    # 2. Local search from best
    if not _timeup():
        try:
            best_U, best_score = _local_search(best_U, max_iters=500, deadline_margin=deadline_margin)
        except Exception:
            pass

    # 3. Random restarts
    random.seed(123)
    for restart in range(8):
        if _timeup():
            break
        try:
            n = random.randint(4, 12)
            start_set = [0] + sorted(random.sample(range(1, 200), n - 1))
            U_r, s_r = _local_search(start_set, max_iters=200, deadline_margin=deadline_margin)
            if s_r > best_score:
                best_score = s_r
                best_U = U_r
        except Exception:
            pass

    # 4. Final high-precision polish pass
    if not _timeup():
        try:
            best_U, best_score = _local_search(best_U, max_iters=300, deadline_margin=deadline_margin, polish=True)
        except Exception:
            pass

    return sorted(set(best_U))
# EVOLVE-BLOCK-END
