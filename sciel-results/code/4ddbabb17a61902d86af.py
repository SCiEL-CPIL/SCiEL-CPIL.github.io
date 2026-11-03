import numpy as np
import time

N = 400
_NFFT = 1 << (2 * N - 1).bit_length()  # next power of 2 >= 2N for fast FFT


def _autocorr_fast(h):
    # linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1
    # Use real FFT trick: autocorr of real signal = IFFT(|FFT|^2)
    F = np.fft.rfft(h, n=_NFFT)
    ac = np.fft.irfft(F * np.conj(F), n=_NFFT)[:N]
    return ac


def _autocorr(h):
    return _autocorr_fast(h)


def _hard_score(h):
    # the actual metric: peak autoconvolution concentration (lower is better)
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _smooth_score(h, beta):
    # smooth (log-sum-exp) surrogate of the peak sidelobe energy, guides SA
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    z = beta * (y - m)
    return m + float(np.log(np.sum(np.exp(z)))) / beta


def _hard_score_approx(h):
    """Faster approximate score: uses a subset of lags to estimate the peak."""
    ac = _autocorr_fast(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    # Check all lags but use a quick max
    return float(np.max(np.abs(ac)) / dc)


def _fast_single_bit_eval(h, ac, i):
    """
    Evaluate the peak sidelobe ratio after flipping h[i], given current ac.
    Returns (new_peak_ratio, new_ac) where new_ac is the full autocorrelation.
    Uses O(N) update instead of O(N log N) FFT.
    """
    new_ac = ac.copy()
    # Flipping h[i] changes ac[k] for all k:
    # For k >= 0: ac[k] = sum_j h[j]*h[j+k]
    # After flip: h'[i] = -h[i]
    # Delta ac[k] = -2*h[i]*h[i+k] for i+k < N (i.e., k < N-i)
    #              -2*h[i]*h[i-k] for i-k >= 0 (i.e., k <= i)
    # Combined: for each k, delta = -2*h[i] * (h[i+k] if i+k<N else 0) + (-2*h[i] * (h[i-k] if i-k>=0 else 0))
    # But note: ac is one-sided, k=0..N-1
    # ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k]
    # After flipping h[i]:
    #   If i <= N-1-k: term j=i is present, delta = -2*h[i]*h[i+k]
    #   If i+k < N: the term at j=i in the sum for ac[k] exists
    #   Also, for the "other side": when we consider ac[k], the term h[i-k]*h[i] appears if i-k >= 0 and i-k <= N-1-k, i.e., i >= k and i <= N-1
    #     Actually ac[k] = sum_{j} h[j]*h[j+k]. The pair (i-k, i) contributes h[i-k]*h[i] when i-k >= 0 and i-k+k = i < N, so i-k >= 0 means k <= i.
    #     So delta from this side: -2*h[i]*h[i-k] when k <= i.
    # Total delta ac[k] = -2*h[i]*(h[i+k] if i+k<N else 0) - 2*h[i]*(h[i-k] if i-k>=0 else 0)
    # But wait: when k=0, both terms give h[i]*h[i]=1, so delta ac[0] = -4*h[i]^2 = -4? That's wrong because ac[0]=sum h[j]^2=N, flipping one bit doesn't change ac[0].
    # Let me re-derive. ac[k] = sum_{j=0}^{N-1-k} h[j]*h[j+k]
    # After flip at i: h'[i] = -h[i], others same.
    # ac'[k] = sum_{j} h'[j]*h'[j+k]
    # Terms affected: those where j=i or j+k=i, i.e., j=i (if i <= N-1-k) or j=i-k (if i-k >= 0 and i-k <= N-1-k).
    # Case 1: j=i is in range [0, N-1-k], i.e., i <= N-1-k, i.e., k <= N-1-i.
    #   Original term: h[i]*h[i+k]
    #   New term: (-h[i])*h[i+k] = -h[i]*h[i+k]
    #   Delta: -2*h[i]*h[i+k]
    # Case 2: j=i-k is in range [0, N-1-k], i.e., 0 <= i-k <= N-1-k, i.e., k <= i and i <= N-1.
    #   Original term: h[i-k]*h[i]
    #   New term: h[i-k]*(-h[i]) = -h[i-k]*h[i]
    #   Delta: -2*h[i-k]*h[i]
    # Note: Case 1 and Case 2 are the same term when i = i-k, i.e., k=0. When k=0:
    #   Case 1: j=i, term h[i]*h[i], delta -2*h[i]^2 = -2
    #   Case 2: j=i, term h[i]*h[i], delta -2*h[i]^2 = -2
    #   But these are the same term! We double-counted.
    # So for k=0, delta = -2*h[i]^2 (only once), not -4.
    # For k > 0: Case 1 and Case 2 are distinct terms (different j values), so delta = -2*h[i]*h[i+k] - 2*h[i]*h[i-k] (where applicable).
    
    hi = h[i]
    # Case 1: k such that i+k < N, i.e., k < N-i
    k_max1 = N - i - 1  # k from 0 to N-i-1
    if k_max1 >= 0:
        new_ac[0:k_max1+1] -= 2.0 * hi * h[i:i+k_max1+1]
    # Case 2: k such that i-k >= 0, i.e., k <= i
    k_max2 = i  # k from 0 to i
    if k_max2 >= 0:
        new_ac[0:k_max2+1] -= 2.0 * hi * h[i-k_max2:i+1][::-1]
    # Fix double-counting at k=0: we subtracted twice, add back once
    new_ac[0] += 2.0 * hi * hi  # since h[i]*h[i] = 1
    
    dc = new_ac[0]
    if dc <= 1e-30:
        return 1e9, new_ac
    peak = float(np.max(np.abs(new_ac[1:]))) / dc
    return peak, new_ac

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    t = [1]
    while len(t) < n:
        t += [-x for x in t]
    return np.array(t[:n])

def construct():
    try:
        t0 = time.time()
        deadline = t0 + 115.0          # safety margin inside the 120s limit
        rng = np.random.default_rng(20240517)

        # ---------- diverse initializations ----------
        idx = np.arange(N)
        inits = [
            _thue_morse(N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx ** 2 / (2 * N)),
            np.cos(2 * np.pi * idx ** 2 / (N + 1)),
            np.cos(2 * np.pi * idx ** 2 / (3 * N)),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx / N),
            np.sin(2 * np.pi * idx / N),
            np.cos(2 * np.pi * 2 * idx / N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
        ]
        best_h = None
        best_hard = 1e9
        for h in inits:
            s = _hard_score(h)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()

        h = best_h.copy()
        cur_hard = best_hard
        cur_smooth = _smooth_score(h, 30.0)

        # ---------- Fast Analytic Greedy Search ----------
        # Use O(N) single-bit update instead of O(N log N) FFT per evaluation
        # This enables ~10x more iterations within the time budget
        
        max_iterations = 50000
        no_improve_count = 0
        max_no_improve = 2000
        
        for _iter in range(max_iterations):
            if time.time() > deadline - 15.0:
                break
            if no_improve_count > max_no_improve:
                break
            
            ac = _autocorr(h)
            dc = ac[0]
            if dc <= 1e-30:
                break
            
            # Find the lag with the maximum absolute sidelobe
            sidelobes = np.abs(ac[1:])
            k_star = int(np.argmax(sidelobes)) + 1
            c_star = ac[k_star]
            cur_peak_ratio = float(np.max(sidelobes)) / dc
            
            if cur_peak_ratio < best_hard - 1e-12:
                best_hard = cur_peak_ratio
                best_h = h.copy()
                no_improve_count = 0
                continue
            
            # Determine which flips would reduce |C[k_star]|
            # Delta ac[k_star] from flipping h[i]:
            #   If i + k_star < N: delta1 = -2*h[i]*h[i+k_star]
            #   If i - k_star >= 0: delta2 = -2*h[i]*h[i-k_star]
            #   (if both, sum them; if k_star=0, only one term)
            
            # For each i, compute the change in ac[k_star]
            # We want to minimize |ac[k_star] + delta|
            
            # Efficiently compute delta for all i:
            # delta[i] = -2*h[i] * (h[i+k_star] if i+k_star<N else 0) + (-2*h[i]*h[i-k_star] if i-k_star>=0 else 0)
            #           but for k_star=0, this double-counts, so delta[0] = -2*h[0]*h[0] = -2 (only one term)
            
            delta = np.zeros(N)
            # Contribution from i+k_star < N
            max_i1 = N - k_star
            if max_i1 > 0:
                delta[0:max_i1] -= 2.0 * h[0:max_i1] * h[k_star:k_star+max_i1]
            # Contribution from i-k_star >= 0
            if k_star > 0:
                max_i2 = N - k_star  # i from k_star to N-1
                delta[k_star:] -= 2.0 * h[k_star:] * h[0:max_i2]
            # For k_star=0, the above two give delta[0] = -2*h[0]*h[0] - 2*h[0]*h[0] = -4, but should be -2
            if k_star == 0:
                delta[0] = -2.0  # h[i]*h[i] = 1
            
            # New ac[k_star] = c_star + delta[i]
            new_c = c_star + delta
            # We want to minimize |new_c|, but also consider global peak
            # Quick filter: only consider i where |new_c| < |c_star|
            improved = np.abs(new_c) < np.abs(c_star) - 1e-12
            if not np.any(improved):
                no_improve_count += 1
                # Try random move
                if rng.random() < 0.05:
                    i = int(rng.integers(0, N))
                    h[i] *= -1.0
                    new_hard = _hard_score(h)
                    if new_hard < cur_hard:
                        cur_hard = new_hard
                        if new_hard < best_hard:
                            best_hard = new_hard
                            best_h = h.copy()
                            no_improve_count = 0
                    else:
                        h[i] *= -1.0
                continue
            
            # Among improved candidates, pick a few to test with full evaluation
            candidate_indices = np.where(improved)[0]
            # Sort by |new_c| to prefer those that most reduce the target peak
            candidate_indices = candidate_indices[np.argsort(np.abs(new_c[candidate_indices]))]
            
            num_tests = min(8, len(candidate_indices))
            test_indices = candidate_indices[:num_tests]
            
            best_candidate = None
            best_candidate_score = cur_peak_ratio
            
            for i in test_indices:
                # Use fast analytic evaluation
                new_ratio, _ = _fast_single_bit_eval(h, ac, int(i))
                if new_ratio < best_candidate_score:
                    best_candidate_score = new_ratio
                    best_candidate = int(i)
            
            if best_candidate is not None and best_candidate_score < cur_peak_ratio - 1e-12:
                h[best_candidate] *= -1.0
                cur_hard = best_candidate_score
                if cur_hard < best_hard:
                    best_hard = cur_hard
                    best_h = h.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                # Occasional random perturbation to escape local minima
                if rng.random() < 0.02:
                    size = int(rng.choice([1, 2, 3]))
                    start = int(rng.integers(0, N - size + 1))
                    sl = slice(start, start + size)
                    old = h[sl].copy()
                    h[sl] *= -1.0
                    new_hard = _hard_score(h)
                    if new_hard < cur_hard:
                        cur_hard = new_hard
                        if new_hard < best_hard:
                            best_hard = new_hard
                            best_h = h.copy()
                            no_improve_count = 0
                    else:
                        h[sl] = old

        # ---------- final single-bit polish with fast evaluation ----------
        h = best_h.copy()
        ac = _autocorr(h)
        dc = ac[0]
        if dc <= 1e-30:
            return best_h
        cur_peak = float(np.max(np.abs(ac[1:]))) / dc
        
        for _pass in range(500):
            if time.time() > deadline - 0.5:
                break
            any_improve = False
            for i in range(N):
                new_ratio, new_ac = _fast_single_bit_eval(h, ac, i)
                if new_ratio < cur_peak - 1e-12:
                    h[i] *= -1.0
                    ac = new_ac
                    cur_peak = new_ratio
                    if cur_peak < best_hard:
                        best_hard = cur_peak
                        best_h = h.copy()
                    any_improve = True
            if not any_improve:
                break

        return best_h
    except Exception:
        return np.ones(N)
