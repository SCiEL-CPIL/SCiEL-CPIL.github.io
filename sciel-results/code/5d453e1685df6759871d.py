"""
Signed equal-width step functions minimizing max|autoconvolution| / (sum(h))^2.

Strategy (structural change from the seed's closed-form constant vector):
  - Treat h as a ±1 vector of length N and search the sign space.
  - Objective:  f(h) = max_i |c_i| / (sum h)^2,  where c = full autoconvolution.
  - Fast incremental updates: flipping h[idx] changes c by a sparse O(N) delta.
  - Multi-start: Rudin-Shapiro, random, and random-with-target-sum initializations.
  - Local search: first-improvement hill climbing with the incremental delta.
  - Escape: short simulated-annealing passes interleaved between restarts.
  - Explicit wall-clock deadline guard; return best found before the limit.
"""
import time
import numpy as np

N = 400
_DEADLINE = 145.0  # safety margin below the 150 s evaluator kill


def _autoconv_max_abs(h):
    """Return the maximum absolute value of the linear autoconvolution of h."""
    c = np.convolve(h, h, mode='full')
    return float(np.max(np.abs(c)))


def _integral_sq(h):
    """Squared integral (sum of heights) of the signal."""
    s = float(np.sum(h))
    return s * s


def _evaluate(h):
    """Compute the objective: max|autoconv| / integral^2."""
    c = np.convolve(h, h, mode='full')
    max_abs = float(np.max(np.abs(c)))
    s = float(np.sum(h))
    s2 = s * s
    if s2 < 1e-12:
        return max_abs
    return max_abs / s2


def _fast_flip_delta(c, h, idx, N):
    """
    Compute the change in the full convolution array when flipping h[idx].
    Returns delta array of length 2N-1.
    """
    s = h[idx]
    delta = np.zeros(2 * N - 1)
    # For each i in [0, 2N-2]:
    #   lag = i - (N-1)
    #   term1: h[idx] * h[idx+lag] if 0 <= idx+lag < N
    #   term2: h[idx-lag] * h[idx] if 0 <= idx-lag < N
    # For i = N-1 (lag=0): no change (self-term)
    
    # Vectorized approach:
    # idx + lag in [0, N-1] => lag in [-idx, N-1-idx] => i in [N-1-idx, N-1+N-1-idx] = [N-1-idx, 2N-2-idx]
    # idx - lag in [0, N-1] => lag in [idx-N+1, idx] => i in [N-1+idx-N+1, N-1+idx] = [idx, N-1+idx]
    
    # Term 1: i from max(0, N-1-idx) to min(2N-2, 2N-2-idx)
    i1_start = max(0, N - 1 - idx)
    i1_end = min(2 * N - 2, 2 * N - 2 - idx)
    if i1_start <= i1_end:
        lags1 = np.arange(i1_start, i1_end + 1) - (N - 1)
        idxs1 = idx + lags1
        # Exclude lag=0 (i=N-1)
        mask = (lags1 != 0)
        if np.any(mask):
            delta[i1_start:i1_end + 1][mask] = -2 * s * h[idxs1[mask]]
    
    # Term 2: i from max(0, idx) to min(2N-2, N-1+idx)
    i2_start = max(0, idx)
    i2_end = min(2 * N - 2, N - 1 + idx)
    if i2_start <= i2_end:
        lags2 = np.arange(i2_start, i2_end + 1) - (N - 1)
        idxs2 = idx - lags2
        # Exclude lag=0 (i=N-1)
        mask = (lags2 != 0)
        if np.any(mask):
            delta[i2_start:i2_end + 1][mask] += -2 * s * h[idxs2[mask]]
    
    return delta


def _hill_climb_fast(h, max_iter=5000, deadline=140.0):
    """
    Local search with fast autoconv updates.
    """
    c = np.convolve(h, h, mode='full')
    current_max = float(np.max(np.abs(c)))
    current_sum = float(np.sum(h))
    current_s2 = current_sum * current_sum
    current_val = current_max / current_s2 if current_s2 > 1e-12 else current_max
    
    # Precompute indices for faster access
    indices = np.arange(N)
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try a random flip
        idx = rng.integers(0, N)
        
        # Compute delta
        delta = _fast_flip_delta(c, h, idx, N)
        c_new = c + delta
        
        new_max = float(np.max(np.abs(c_new)))
        new_sum = current_sum - 2 * h[idx]
        new_s2 = new_sum * new_sum
        new_val = new_max / new_s2 if new_s2 > 1e-12 else new_max
        
        if new_val < current_val:
            h[idx] = -h[idx]
            c = c_new
            current_max = new_max
            current_sum = new_sum
            current_s2 = new_s2
            current_val = new_val
    
    return h, current_val, c


def _simulated_annealing(h, deadline=140.0):
    """
    Simulated annealing to escape local minima.
    """
    c = np.convolve(h, h, mode='full')
    current_max = float(np.max(np.abs(c)))
    current_sum = float(np.sum(h))
    current_s2 = current_sum * current_sum
    current_val = current_max / current_s2 if current_s2 > 1e-12 else current_max
    
    rng = np.random.default_rng(123)
    temp = 0.1
    best_h = h.copy()
    best_val = current_val
    
    for it in range(20000):
        if time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        delta = _fast_flip_delta(c, h, idx, N)
        c_new = c + delta
        
        new_max = float(np.max(np.abs(c_new)))
        new_sum = current_sum - 2 * h[idx]
        new_s2 = new_sum * new_sum
        new_val = new_max / new_s2 if new_s2 > 1e-12 else new_max
        
        delta_obj = new_val - current_val
        
        if delta_obj < 0 or rng.random() < np.exp(-delta_obj / temp):
            h[idx] = -h[idx]
            c = c_new
            current_max = new_max
            current_sum = new_sum
            current_s2 = new_s2
            current_val = new_val
            
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
        
        temp *= 0.9995
    
    return best_h, best_val


def _build_rudin_shapiro(n):
    """Construct Rudin-Shapiro sequence of length n."""
    A = np.array([1.0])
    B = np.array([1.0])
    while len(A) < n:
        A = np.concatenate([A, B])
        B = np.concatenate([A, -B])
    return A[:n]


def _random_with_sum_constraint(n, target_sum, rng):
    """Generate random ±1 sequence with approximate sum = target_sum."""
    h = rng.choice([-1.0, 1.0], size=n)
    current_sum = float(np.sum(h))
    # Flip some to adjust sum
    if abs(current_sum - target_sum) > n * 0.1:
        # Flip indices to move sum toward target
        if current_sum > target_sum:
            # Need to decrease sum: flip +1 to -1
            plus_indices = np.where(h == 1.0)[0]
            for idx in plus_indices[:min(len(plus_indices), int(abs(current_sum - target_sum) / 2))]:
                h[idx] = -1.0
        else:
            minus_indices = np.where(h == -1.0)[0]
            for idx in minus_indices[:min(len(minus_indices), int(abs(target_sum - current_sum) / 2))]:
                h[idx] = 1.0
    return h


def construct():
    """
    Multi-start local search over ±1 sign vectors minimizing
    max|autoconv(h)| / (sum(h))^2, with an explicit wall-clock deadline.
    """
    t0 = time.time()
    deadline = t0 + (_DEADLINE - (time.time() - t0))  # same as t0 + _DEADLINE
    rng = np.random.default_rng(20240601)

    best_h = None
    best_val = float("inf")

    # ---- initializations ------------------------------------------------
    def _init_rs(seed):
        r = np.random.default_rng(seed)
        A = np.array([1.0]); B = np.array([1.0])
        while len(A) < N:
            A = np.concatenate([A, B])
            B = np.concatenate([A, -B])
        A = A[:N]
        # random sign flips to diversify
        flips = r.choice(N, size=int(N * 0.15), replace=False)
        A[flips] *= -1.0
        return A

    def _init_random(seed):
        r = np.random.default_rng(seed)
        return r.choice([-1.0, 1.0], size=N)

    def _init_target_sum(seed, target):
        r = np.random.default_rng(seed)
        h = r.choice([-1.0, 1.0], size=N)
        # adjust toward target sum by flipping
        for _ in range(200):
            s = float(np.sum(h))
            if abs(s - target) < 2:
                break
            if s > target:
                plus = np.where(h == 1.0)[0]
                if len(plus) == 0:
                    break
                idx = plus[r.integers(0, len(plus))]
                h[idx] = -1.0
            else:
                minus = np.where(h == -1.0)[0]
                if len(minus) == 0:
                    break
                idx = minus[r.integers(0, len(minus))]
                h[idx] = 1.0
        return h

    # ---- evaluation helpers ---------------------------------------------
    def _eval(h):
        c = np.convolve(h, h, mode="full")
        m = float(np.max(np.abs(c)))
        s = float(np.sum(h))
        s2 = s * s
        return (m / s2 if s2 > 1e-12 else m), c

    # ---- hill climbing ----------------------------------------------------
    def _hill_climb(h, max_iter, dl):
        c = np.convolve(h, h, mode="full")
        cur_max = float(np.max(np.abs(c)))
        cur_sum = float(np.sum(h))
        cur_s2 = cur_sum * cur_sum
        cur_val = cur_max / cur_s2 if cur_s2 > 1e-12 else cur_max
        local_rng = np.random.default_rng(int(cur_val * 1e6) % (2**31))

        for _ in range(max_iter):
            if time.time() > dl:
                break
            idx = int(local_rng.integers(0, N))
            s = h[idx]
            # compute delta for c
            delta = np.zeros(2 * N - 1)
            # term 1: idx + lag in [0, N-1]
            i1s = max(0, N - 1 - idx)
            i1e = min(2 * N - 2, 2 * N - 2 - idx)
            if i1s <= i1e:
                lags = np.arange(i1s, i1e + 1) - (N - 1)
                idxs = idx + lags
                mask = lags != 0
                if np.any(mask):
                    delta[i1s:i1e + 1][mask] = -2.0 * s * h[idxs[mask]]
            # term 2: idx - lag in [0, N-1]
            i2s = max(0, idx)
            i2e = min(2 * N - 2, N - 1 + idx)
            if i2s <= i2e:
                lags = np.arange(i2s, i2e + 1) - (N - 1)
                idxs = idx - lags
                mask = lags != 0
                if np.any(mask):
                    delta[i2s:i2e + 1][mask] += -2.0 * s * h[idxs[mask]]

            c_new = c + delta
            new_max = float(np.max(np.abs(c_new)))
            new_sum = cur_sum - 2.0 * s
            new_s2 = new_sum * new_sum
            new_val = new_max / new_s2 if new_s2 > 1e-12 else new_max

            if new_val < cur_val:
                h[idx] = -s
                c = c_new
                cur_max = new_max
                cur_sum = new_sum
                cur_s2 = new_s2
                cur_val = new_val
        return h, cur_val, c

    # ---- simulated annealing ---------------------------------------------
    def _anneal(h, max_iter, dl, t0_temp=0.05):
        c = np.convolve(h, h, mode="full")
        cur_max = float(np.max(np.abs(c)))
        cur_sum = float(np.sum(h))
        cur_s2 = cur_sum * cur_sum
        cur_val = cur_max / cur_s2 if cur_s2 > 1e-12 else cur_max
        best_h = h.copy()
        best_val = cur_val
        temp = t0_temp
        local_rng = np.random.default_rng(int(abs(cur_val) * 1e6) % (2**31))

        for _ in range(max_iter):
            if time.time() > dl:
                break
            idx = int(local_rng.integers(0, N))
            s = h[idx]
            delta = np.zeros(2 * N - 1)
            i1s = max(0, N - 1 - idx)
            i1e = min(2 * N - 2, 2 * N - 2 - idx)
            if i1s <= i1e:
                lags = np.arange(i1s, i1e + 1) - (N - 1)
                idxs = idx + lags
                mask = lags != 0
                if np.any(mask):
                    delta[i1s:i1e + 1][mask] = -2.0 * s * h[idxs[mask]]
            i2s = max(0, idx)
            i2e = min(2 * N - 2, N - 1 + idx)
            if i2s <= i2e:
                lags = np.arange(i2s, i2e + 1) - (N - 1)
                idxs = idx - lags
                mask = lags != 0
                if np.any(mask):
                    delta[i2s:i2e + 1][mask] += -2.0 * s * h[idxs[mask]]

            c_new = c + delta
            new_max = float(np.max(np.abs(c_new)))
            new_sum = cur_sum - 2.0 * s
            new_s2 = new_sum * new_sum
            new_val = new_max / new_s2 if new_s2 > 1e-12 else new_max
            d_obj = new_val - cur_val

            if d_obj < 0 or (temp > 1e-12 and local_rng.random() < np.exp(-d_obj / temp)):
                h[idx] = -s
                c = c_new
                cur_max = new_max
                cur_sum = new_sum
                cur_s2 = new_s2
                cur_val = new_val
                if cur_val < best_val:
                    best_val = cur_val
                    best_h = h.copy()
            temp *= 0.999
        return best_h, best_val

    # ---- main search loop -------------------------------------------------
    start_seeds = []
    # Rudin-Shapiro variants
    for s in range(8):
        start_seeds.append(("rs", s))
    # random
    for s in range(16):
        start_seeds.append(("rand", s))
    # target-sum variants (small sums to keep denominator reasonable but not trivial)
    for s in range(8):
        target = int(rng.integers(-N // 4, N // 4 + 1))
        start_seeds.append(("ts", (s, target)))

    # Shuffle for diversity
    rng.shuffle(start_seeds)

    for kind, key in start_seeds:
        if time.time() > deadline - 2.0:
            break
        if kind == "rs":
            h = _init_rs(key)
        elif kind == "rand":
            h = _init_random(key)
        else:
            s, target = key
            h = _init_target_sum(s, target)

        # Hill climb
        h, v, _ = _hill_climb(h, max_iter=3000, dl=deadline - 1.0)
        if v < best_val:
            best_val = v
            best_h = h.copy()

        # Short anneal to escape
        if time.time() < deadline - 3.0:
            h2, v2 = _anneal(h, max_iter=2000, dl=deadline - 1.0)
            if v2 < best_val:
                best_val = v2
                best_h = h2.copy()

    # Final fallback: if nothing found (shouldn't happen), return ones
    if best_h is None:
        best_h = np.ones(N)

    # Safety: ensure deadline not exceeded
    if time.time() > deadline:
        pass  # return best found

    return best_h
