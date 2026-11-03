"""
Combined approach: use the Legendre-symbol (quadratic residue) deterministic
sequence as a strong starting point (low autoconvolution sidelobes by number
theory), then refine with simulated annealing using a smooth log-sum-exp
surrogate and a final greedy multi-scale polish.

The C3 autoconvolution upper bound is governed by the peak aperiodic
autocorrelation sidelobe of the step heights.  A ±1 Legendre symbol sequence
has provably low additive energy, so it starts far below the all-ones baseline.
Simulated annealing with a smooth surrogate escapes local minima, and greedy
polishing tightens the result.
"""
import numpy as np
import time

N = 400

# Precompute FFT size for linear autocorrelation
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _autocorr(h):
    """Linear one-sided aperiodic autocorrelation: ac[k] = sum_i h[i]*h[i+k]."""
    H = np.fft.rfft(h, n=_FFT_SIZE)
    ac = np.fft.irfft(H * np.conj(H), n=_FFT_SIZE)[:N]
    return ac


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe (excluding k=0)."""
    ac = _autocorr(h)
    return float(np.max(np.abs(ac[1:])))


def _hard_score(h):
    """Raw metric: peak sidelobe (lower is better)."""
    return _peak_sidelobe(h)


def _smooth_score(h, beta=40.0):
    """
    Smooth surrogate of the peak sidelobe via log-sum-exp.
    This differentiable proxy guides the annealer more effectively than
    the hard max, especially when many sidelobes are near the peak.
    """
    ac = _autocorr(h)
    # Normalize by main lobe to make scale-invariant
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    if m == 0.0:
        return 0.0
    z = beta * (y - m)
    # log-sum-exp trick for numerical stability
    log_sum = m + float(np.log(np.sum(np.exp(z)))) / beta
    return float(np.sqrt(log_sum))


def _legendre_sequence():
    """
    Deterministic ±1 sequence based on the Legendre symbol mod p,
    where p is the smallest prime >= N.  For x in 1..N-1:
      h[x-1] = +1 if x is a quadratic residue mod p, else -1.
    h[0] = -1 (0 is not a residue).
    Pad/truncate to length N.
    """
    # Find smallest prime >= N
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(np.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    p = N
    while not is_prime(p):
        p += 1

    h = np.ones(N, dtype=np.float64)
    for x in range(1, N):
        # Legendre symbol (x/p): +1 if x is QR mod p, -1 otherwise
        # Use Euler's criterion: x^((p-1)/2) mod p == 1 or p-1
        legendre = pow(x, (p - 1) // 2, p)
        if legendre == 1:
            h[x] = 1.0
        else:
            h[x] = -1.0
    h[0] = -1.0  # 0 is not a quadratic residue
    return h


def _anneal(h_init, time_budget, seed=42):
    """
    Simulated annealing with block-flip moves, guided by the smooth
    log-sum-exp surrogate.  Returns (best_h, best_hard).
    """
    rng = np.random.default_rng(seed)
    h = h_init.copy()
    cur_hard = _hard_score(h)
    best_h = h.copy()
    best = cur_hard

    T = 0.05
    T_min = 1e-5
    decay = 0.9997
    beta = 40.0
    cur_smooth = _smooth_score(h, beta)

    block_sizes = np.array([1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64])
    start = time.time()

    while time.time() - start < time_budget:
        # Choose move type: 80% block flip, 20% single flip
        if rng.random() < 0.80:
            size = int(rng.choice(block_sizes))
            if size >= N:
                size = N - 1
            i = int(rng.integers(0, N - size + 1))
            sl = slice(i, i + size)
            old = h[sl].copy()
            h[sl] *= -1.0

            new_hard = _hard_score(h)
            new_smooth = _smooth_score(h, beta)
            dS = new_smooth - cur_smooth

            if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / max(T, 1e-9))):
                cur_hard = new_hard
                cur_smooth = new_smooth
                if new_hard < best:
                    best = new_hard
                    best_h = h.copy()
            else:
                h[sl] = old
        else:
            i = int(rng.integers(0, N))
            old = h[i]
            h[i] *= -1.0

            new_hard = _hard_score(h)
            new_smooth = _smooth_score(h, beta)
            dS = new_smooth - cur_smooth

            if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / max(T, 1e-9))):
                cur_hard = new_hard
                cur_smooth = new_smooth
                if new_hard < best:
                    best = new_hard
                    best_h = h.copy()
            else:
                h[i] = old

        T = max(T_min, T * decay)

    return best_h, best


def _greedy_polish(h, time_budget):
    """
    Greedy multi-scale local polish: repeatedly flip the single bit or small
    block that most reduces the hard peak-sidelobe score.
    """
    h = h.copy()
    cur = _hard_score(h)
    best = cur
    best_h = h.copy()
    start = time.time()

    polish_sizes = [1, 2, 4, 8, 16]

    while time.time() - start < time_budget:
        improved = False
        for size in polish_sizes:
            if time.time() - start >= time_budget:
                break
            best_i = -1
            best_c = cur
            for i in range(0, N - size + 1):
                if time.time() - start >= time_budget:
                    break
                sl = slice(i, i + size)
                old = h[sl].copy()
                h[sl] *= -1.0
                c = _hard_score(h)
                if c < best_c:
                    best_c = c
                    best_i = i
                h[sl] = old

            if best_i >= 0:
                sl = slice(best_i, best_i + size)
                h[sl] *= -1.0
                cur = best_c
                improved = True
                if cur < best:
                    best = cur
                    best_h = h.copy()

        if not improved:
            break

    return best_h, best


def construct():
    try:
        hard_deadline = 135.0  # safety margin inside 150s kill
        t0 = time.time()

        # --- Initial candidates ---
        legendre_h = _legendre_sequence()
        rng = np.random.default_rng(20250101)
        random_h = rng.choice(np.array([-1.0, 1.0]), size=N)

        candidates = [legendre_h, random_h, np.ones(N)]
        best_h = None
        best_score = 1e18
        for c in candidates:
            s = _hard_score(c)
            if s < best_score:
                best_score = s
                best_h = c.copy()

        # --- Phase 1: Anneal from the best initial (likely Legendre) ---
        remaining = hard_deadline - (time.time() - t0) - 3.0
        budget1 = min(40.0, max(remaining, 5.0))
        if budget1 > 2.0:
            h1, s1 = _anneal(best_h, budget1, seed=12345)
            if s1 < best_score:
                best_score = s1
                best_h = h1

        # --- Phase 2: Shorter restarts for diversity ---
        restart_id = 1
        while time.time() - t0 < hard_deadline - 5.0:
            sub = min(12.0, hard_deadline - (time.time() - t0) - 5.0)
            if sub < 2.0:
                break
            # Alternate between Legendre and random restarts
            if restart_id % 2 == 1:
                h_init = _legendre_sequence().copy()
                # Perturb slightly
                for _ in range(10):
                    idx = rng.integers(0, N)
                    h_init[idx] *= -1.0
                seed = 100000 + restart_id
            else:
                h_init = rng.choice(np.array([-1.0, 1.0]), size=N)
                seed = 200000 + restart_id

            h_r, s_r = _anneal(h_init, sub, seed=seed)
            if s_r < best_score:
                best_score = s_r
                best_h = h_r
            restart_id += 1

        # --- Phase 3: Final greedy polish ---
        remaining = hard_deadline - (time.time() - t0) - 1.0
        if remaining > 1.0:
            h_p, s_p = _greedy_polish(best_h, remaining)
            if s_p < best_score:
                best_score = s_p
                best_h = h_p

        return np.ascontiguousarray(best_h, dtype=np.float64)

    except Exception:
        return np.ones(N)
