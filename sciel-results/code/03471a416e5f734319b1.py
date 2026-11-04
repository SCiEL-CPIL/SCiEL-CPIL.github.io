import numpy as np
import time

N = 400
FFT_SIZE = 1
while FFT_SIZE < 2 * N:
    FFT_SIZE *= 2  # 512

def _score(h):
    """Max absolute autocorrelation for lags 1..N-1."""
    H = np.fft.rfft(h, n=FFT_SIZE)
    acf = np.fft.irfft(np.abs(H) ** 2, n=FFT_SIZE)
    return float(np.max(np.abs(acf[1:N])))

def _acf(h):
    H = np.fft.rfft(h, n=FFT_SIZE)
    return np.fft.irfft(np.abs(H) ** 2, n=FFT_SIZE)

def _thue_morse(n):
    """Thue-Morse sequence: t_k = parity of popcount(k), mapped to +/-1."""
    out = np.zeros(n)
    for k in range(n):
        # popcount parity
        x = k
        p = 0
        while x:
            p ^= (x & 1)
            x >>= 1
        out[k] = 1.0 if p == 0 else -1.0
    return out

def _thue_morse_fast(n):
    """Vectorized Thue-Morse via popcount parity using numpy."""
    ks = np.arange(n)
    # parity of popcount: xor of bits
    p = np.zeros(n, dtype=np.int32)
    for shift in range(0, 16):
        p ^= ((ks >> shift) & 1)
    return np.where(p == 0, 1.0, -1.0)

def _golomb_rudin(n):
    """Golomb-Rudin: g_0=+1, g_{k+1} = g_k, -g_k (recursion). Gives low autocorr."""
    g = np.array([1.0])
    while len(g) < n:
        g = np.concatenate([g, -g])
    return g[:n]

def construct():
    start_time = time.time()
    deadline = 142.0  # safety margin before 150s

    def elapsed():
        return time.time() - start_time

    best_h = None
    best_score = float('inf')

    # ---- Fast greedy local search using O(N) ACF update per flip ----
    # ACF(k) = sum_{i} h[i] * h[i+k]   (for k >= 0, with i+k < N)
    # Full circular ACF from FFT (length FFT_SIZE, zero-padded) gives
    #   ACF_circ(k) = sum_i h[i] h[(i+k) mod FFT_SIZE]
    # Since h is zero-padded to FFT_SIZE, ACF_circ(k) = sum_{i=0}^{N-1-k} h[i] h[i+k] for 0<=k<N,
    # and ACF_circ(FFT_SIZE - k) = sum_{i=k}^{N-1} h[i] h[i-k] = same (symmetric).
    # So the FFT-based ACF[1:N] is the linear autocorrelation. Good.
    #
    # Flipping h[j] -> -h[j]:
    #   Change in ACF_circ(k) for k in 1..N-1:
    #     terms involving h[j]: h[j]*h[j+k] (if j+k<N) and h[j-k]*h[j] (if j-k>=0)
    #     each flips sign -> change = -2*h[j]*h[j+k] - 2*h[j-k]*h[j]
    #     = -2*h[j]*(h[j+k] + h[j-k])
    #   But careful: the FFT ACF is the *circular* one over FFT_SIZE. Since h is zero-padded,
    #   for k in 1..N-1, the only nonzero contributions are from indices in [0,N-1].
    #   So the linear update holds for lags 1..N-1.
    #   Also need to consider lag 0? We don't use lag 0 in score. But flipping changes ACF[0] too;
    #   we only track ACF[1:N]. Fine.
    #   However, note: the FFT computes ACF over the *circular* convolution of h with reversed h.
    #   With zero padding to FFT_SIZE >= 2N, there's no wrap-around for lags < N. Good.
    #
    # So: new_acf[k] = old_acf[k] - 2*h[j]*(h[j+k] + h[j-k])  for k=1..N-1,
    #   where h[j+k] = 0 if j+k >= N, h[j-k] = 0 if j-k < 0.
    #   Wait: is it -2 or -4? Let's re-derive.
    #   Old term at lag k: T_old = h[j]*h[j+k]  (if valid) + h[j-k]*h[j] (if valid)
    #   New term: T_new = (-h[j])*h[j+k] + h[j-k]*(-h[j]) = -h[j]*h[j+k] - h[j-k]*h[j] = -T_old
    #   So change = T_new - T_old = -2*T_old = -2*(h[j]*h[j+k] + h[j-k]*h[j]) = -2*h[j]*(h[j+k]+h[j-k])
    #   Yes, -2.
    #
    # But the FFT ACF might have a factor? np.fft.irfft(|H|^2) gives the *circular* autocorrelation
    # without any scaling beyond the FFT convention. Let's verify: for a real signal h of length n,
    # ifft(|fft(h)|^2) gives sum_i h[i] h[(i+k) mod n]. With zero-padding to FFT_SIZE, it's the
    # linear autocorrelation. No extra factor. Good.

    def _local_search(h, time_limit):
        h = h.copy()
        acf = _acf(h)
        cur_score = float(np.max(np.abs(acf[1:N])))

        # Precompute index arrays for fast delta computation
        # For a flip at j, delta[k] = -2*h[j]*(h[j+k] + h[j-k]) for k=1..N-1
        # We'll compute this vectorized per j.

        improved = True
        while improved and elapsed() < time_limit:
            improved = False
            best_j = -1
            best_new_score = cur_score
            best_delta_vec = None

            # To speed up, we can evaluate all j in a loop but vectorize the delta per j.
            # N=400, so 400 flips per pass. Each flip: compute delta vec (O(N)) + max (O(N)).
            # That's 400 * 400 = 160k ops per pass, very fast.
            for j in range(N):
                hj = h[j]
                # h[j+k] for k=1..N-1: valid when j+k < N -> k < N-j
                # h[j-k] for k=1..N-1: valid when j-k >= 0 -> k <= j
                # Build the sum vector s[k] = h[j+k] + h[j-k] for k=1..N-1
                # Vectorized:
                #   For k in 1..N-1:
                #     left = h[j-k] if j-k >= 0 else 0
                #     right = h[j+k] if j+k < N else 0
                # We can do this with slicing.
                # right part: h[j+1 : j+N] but only up to N-1 -> h[j+1 : min(N, j+N)]
                #   but we need exactly N-1 entries for k=1..N-1.
                #   right[k-1] = h[j+k] if j+k < N else 0
                #   = h[j+1 : j+N] padded with zeros to length N-1.
                # left[k-1] = h[j-k] if j-k >= 0 else 0
                #   = h[j-1 : j-N-1 : -1] i.e., h[j-1], h[j-2], ... but reversed and padded.
                #   left[k-1] = h[j-k] = h[j-1-(k-1)] -> for k-1 in 0..N-2, index j-1-(k-1) = j-k.
                #   So left = h[j-1 : j-N-1 : -1] but that gives j-1, j-2, ..., down to max(0, j-N+1)?
                #   Actually h[j-1 : j-N-1 : -1] in Python: start=j-1, stop=j-N-1 (exclusive), step=-1.
                #   That yields indices j-1, j-2, ..., j-N (if in range). We want only valid (>=0).
                #   Simpler: build via np.where.
                k = np.arange(1, N)  # 1..N-1
                # right: h[j+k]
                idx_r = j + k
                right = np.where(idx_r < N, h[idx_r], 0.0)
                # left: h[j-k]
                idx_l = j - k
                left = np.where(idx_l >= 0, h[idx_l], 0.0)
                s = left + right
                delta = -2.0 * hj * s
                new_acf = acf[1:N] + delta
                new_score = float(np.max(np.abs(new_acf)))
                if new_score < best_new_score - 1e-12:
                    best_new_score = new_score
                    best_j = j
                    best_delta_vec = delta

            if best_j != -1:
                h[best_j] = -h[best_j]
                acf[1:N] = acf[1:N] + best_delta_vec
                cur_score = best_new_score
                improved = True

        return h, cur_score

    def _pair_local_search(h, time_limit):
        """Local search with pair flips to escape local minima."""
        h = h.copy()
        acf = _acf(h)
        cur_score = float(np.max(np.abs(acf[1:N])))

        k_arr = np.arange(1, N)

        def flip_delta(j, hj):
            idx_r = j + k_arr
            right = np.where(idx_r < N, h[idx_r], 0.0)
            idx_l = j - k_arr
            left = np.where(idx_l >= 0, h[idx_l], 0.0)
            return -2.0 * hj * (left + right)

        improved = True
        while improved and elapsed() < time_limit:
            improved = False
            best_pair = None
            best_new_score = cur_score
            best_d1 = None
            best_d2 = None

            # Try all pairs (j, l), j < l. That's ~80k pairs, each O(N) -> 32M ops, too slow per pass.
            # Instead, do a first pass of single flips; if none, try a limited set of pairs.
            # First: single flip
            single_best_j = -1
            single_best_score = cur_score
            single_best_delta = None
            for j in range(N):
                d = flip_delta(j, h[j])
                ns = float(np.max(np.abs(acf[1:N] + d)))
                if ns < single_best_score - 1e-12:
                    single_best_score = ns
                    single_best_j = j
                    single_best_delta = d

            if single_best_j != -1:
                h[single_best_j] = -h[single_best_j]
                acf[1:N] = acf[1:N] + single_best_delta
                cur_score = single_best_score
                improved = True
                continue

            # No single flip helps. Try pairs: for each j, compute delta_j. Then for l != j,
            # combined delta = delta_j + delta_l (since flipping both: change is sum of individual
            # changes? Let's check: flipping j changes ACF by d_j. Then flipping l (in the new state)
            # changes by d_l computed with h after j flip. But d_l depends on h values, which changed
            # at j. So it's not simply additive. However, if j and l don't interact (their influence
            # regions don't overlap in a way that matters), it's approximately additive. For exactness,
            # we'd need to recompute. To keep it fast, we approximate: assume d_l is computed on the
            # *original* h. This is valid if the flip at j doesn't affect the terms in d_l, i.e.,
            # if j is not in {l+k, l-k} for any k. That's rare. We'll accept the approximation.
            # Actually, let's be more careful: the change from flipping both j and l is:
            #   For each lag k, the terms involving h[j] and h[l] both flip.
            #   The combined change is not simply d_j + d_l because d_j already accounts for the
            #   flip of j, and d_l accounts for the flip of l, but when both are flipped, the cross
            #   terms h[j]*h[l] (if |j-l|=k) change sign *once*, not twice.
            #   This is getting complex. For simplicity and speed, we'll do: try pairs by actually
            #   computing the new ACF from scratch for a *small* set of candidate pairs.
            #   Candidate pairs: (j, l) where l is near j (within 20) or l = j+1, j+2, etc.
            #   This is a heuristic to escape minima.
            #
            # To keep it fast, we'll try a limited number of pairs: for each j (0..N-1), try l = j+1, j+2, ..., j+5 (mod N).
            # That's 5*N = 2000 pairs per pass. For each, compute new ACF from scratch (O(N log N)) -> 2000 * ~2000 = 4M ops. OK.
            for j in range(N):
                for offset in range(1, 6):
                    l = (j + offset) % N
                    if l == j:
                        continue
                    h[j] = -h[j]
                    h[l] = -h[l]
                    ns = _score(h)
                    h[j] = -h[j]
                    h[l] = -h[l]
                    if ns < best_new_score - 1e-12:
                        best_new_score = ns
                        best_pair = (j, l)
                        improved = True

            if best_pair is not None:
                j, l = best_pair
                h[j] = -h[j]
                h[l] = -h[l]
                acf = _acf(h)
                cur_score = best_new_score

        return h, cur_score

    # ---- Generate diverse seeds ----
    rng = np.random.default_rng(12345)
    seeds = []

    # Thue-Morse (known flat autocorrelation)
    seeds.append(_thue_morse_fast(N))
    # Thue-Morse with random phase (cyclic shift)
    for shift in range(0, N, N // 8):
        seeds.append(np.roll(_thue_morse_fast(N), shift))
    # Golomb-Rudin
    seeds.append(_golomb_rudin(N))
    seeds.append(np.roll(_golomb_rudin(N), N // 4))
    # Random
    for _ in range(15):
        seeds.append(rng.choice([-1.0, 1.0], size=N))
    # Quasi-periodic
    for _ in range(8):
        freqs = rng.uniform(0.1, 5.0, size=6)
        phases = rng.uniform(0, 2 * np.pi, size=6)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for f, p in zip(freqs, phases):
            val += np.cos(2 * np.pi * f * t + p)
        seeds.append(np.sign(val))
    # Block patterns
    for bs in [5, 10, 20, 25, 50]:
        h = np.zeros(N)
        for b in range(N // bs):
            h[b*bs:(b+1)*bs] = rng.choice([-1.0, 1.0])
        if N % bs != 0:
            h[-(N % bs):] = rng.choice([-1.0, 1.0])
        seeds.append(h)
    # Alternating with defects
    base = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    for _ in range(5):
        h = base.copy()
        defect_idx = rng.choice(N, size=N // 20, replace=False)
        h[defect_idx] = -h[defect_idx]
        seeds.append(h)

    # ---- Evaluate seeds ----
    scored = []
    for s in seeds:
        sc = _score(s)
        scored.append((sc, s))
    scored.sort(key=lambda x: x[0])

    # ---- Refine top seeds with local search ----
    # Allocate time: give more time to better seeds.
    # We have ~140s. Reserve 10s for final random restarts.
    search_budget = deadline - 10.0
    n_refine = min(8, len(scored))
    # Weight by rank: rank 0 gets 3x, rank 1 gets 2x, others 1x.
    weights = [3.0, 2.0] + [1.0] * (n_refine - 2)
    total_w = sum(weights)
    per_seed_time = (search_budget - 5.0) / total_w  # reserve 5s buffer

    for rank in range(n_refine):
        if elapsed() > deadline - 5:
            break
        sc, s = scored[rank]
        t_limit = min(deadline - 5.0, elapsed() + per_seed_time * weights[rank])
        h_ref, sc_ref = _local_search(s, t_limit)
        if sc_ref < best_score:
            best_score = sc_ref
            best_h = h_ref.copy()

    # ---- Pair local search on best found ----
    if best_h is not None and elapsed() < deadline - 10:
        t_limit = min(deadline - 5.0, elapsed() + 10.0)
        h_ref, sc_ref = _pair_local_search(best_h, t_limit)
        if sc_ref < best_score:
            best_score = sc_ref
            best_h = h_ref.copy()

    # ---- Random restarts with local search ----
    while elapsed() < deadline - 3:
        h_init = rng.choice([-1.0, 1.0], size=N)
        # Occasionally use Thue-Morse variant
        if rng.random() < 0.3:
            h_init = np.roll(_thue_morse_fast(N), rng.integers(0, N))
        t_limit = min(deadline - 2.0, elapsed() + 3.0)
        h_ref, sc_ref = _local_search(h_init, t_limit)
        if sc_ref < best_score:
            best_score = sc_ref
            best_h = h_ref.copy()

    if best_h is None:
        best_h = _thue_morse_fast(N)

    return best_h
