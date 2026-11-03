import numpy as np
import time

N = 400
_NFFT = 1 << (2 * N - 1).bit_length()


def _autocorr_fast(h):
    F = np.fft.rfft(h, n=_NFFT)
    ac = np.fft.irfft(F * np.conj(F), n=_NFFT)[:N]
    return ac


def _hard_score(h):
    ac = _autocorr_fast(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _flip_ac(h, ac, i):
    """Compute new autocorrelation after flipping h[i], in O(N)."""
    ac_new = ac.copy()
    hi = h[i]
    # For k=0: unchanged
    # For k>0: delta = -2*hi*(h[i+k] if i+k<N else 0 + h[i-k] if i-k>=0 else 0)
    k = np.arange(1, N)
    # i+k valid
    vp = (i + k) < N
    hp = np.zeros(N - 1)
    if np.any(vp):
        hp[vp] = h[i + k[vp]]
    # i-k valid
    vm = (i - k) >= 0
    hm = np.zeros(N - 1)
    if np.any(vm):
        hm[vm] = h[i - k[vm]]
    ac_new[1:] += -2.0 * hi * (hp + hm)
    return ac_new


def _block_flip_ac(h, ac, indices):
    """Compute new autocorrelation after flipping multiple bits simultaneously."""
    # For a set of flipped indices S, the change in ac[k] is:
    # delta_ac[k] = -2 * sum_{i in S, i+k<N} h[i]*h[i+k] - 2 * sum_{i in S, i-k>=0} h[i-k]*h[i]
    # But we must be careful: if both i and j are in S and j = i+k, the cross term changes differently.
    # Actually, the correct formula for flipping a set S:
    # ac_new[k] = ac_old[k] - 2*sum_{i in S} [h[i]*h[i+k] (if i+k<N) + h[i-k]*h[i] (if i-k>=0)]
    #            + 4*sum_{i<j, j-i=k, i,j in S} h[i]*h[j]  (cross terms that were both flipped)
    # Wait, let me think more carefully.
    # Original: ac[k] = sum_j h[j]*h[j+k]
    # After flipping S: ac_new[k] = sum_j h'[j]*h'[j+k]
    # where h'[j] = -h[j] if j in S, else h[j]
    # 
    # Terms that change: those where j in S or j+k in S.
    # Let A = {j : j in S, j+k < N}, B = {j : j+k in S, j >= 0} = {j : j in S-k, j >= 0}
    # 
    # For j in A \ B: term changes from h[j]*h[j+k] to -h[j]*h[j+k], delta = -2*h[j]*h[j+k]
    # For j in B \ A: term changes from h[j]*h[j+k] to h[j]*(-h[j+k]), delta = -2*h[j]*h[j+k]
    # For j in A ∩ B: term changes from h[j]*h[j+k] to (-h[j])*(-h[j+k]) = h[j]*h[j+k], delta = 0
    #
    # So: delta_ac[k] = -2 * sum_{j in A \ B} h[j]*h[j+k] - 2 * sum_{j in B \ A} h[j]*h[j+k]
    # = -2 * (sum_{j in A} h[j]*h[j+k] - sum_{j in A∩B} h[j]*h[j+k] + sum_{j in B} h[j]*h[j+k] - sum_{j in A∩B} h[j]*h[j+k])
    # = -2 * (sum_{j in A} h[j]*h[j+k] + sum_{j in B} h[j]*h[j+k] - 2*sum_{j in A∩B} h[j]*h[j+k])
    
    # For simplicity and correctness with small blocks, let's just do it directly for small blocks
    # For larger blocks, use FFT
    if len(indices) <= 8:
        ac_new = ac.copy()
        S = set(indices.tolist())
        for k in range(1, N):
            # A: j in S, j+k < N
            # B: j+k in S, j >= 0 => j in S - k
            delta = 0.0
            for j in S:
                if j + k < N:
                    # j in A
                    if j + k in S:
                        # j in A ∩ B, no change
                        pass
                    else:
                        delta += -2.0 * h[j] * h[j + k]
                if j - k >= 0 and j - k not in S:
                    # j-k in B but j not in A (since j-k not in S means j-k not in B... wait)
                    # Actually B = {m : m+k in S} = {m : m in S-k}
                    # So m = j-k is in B if j in S
                    # We already handled j in A above
                    # For m in B: m+k in S, m >= 0
                    # m = j-k where j in S and j-k >= 0
                    # If m not in S (i.e., j-k not in S), then m in B \ A
                    # delta += -2*h[m]*h[m+k] = -2*h[j-k]*h[j]
                    delta += -2.0 * h[j - k] * h[j]
            ac_new[k] += delta
        return ac_new
    else:
        # For larger blocks, just recompute via FFT
        h_new = h.copy()
        h_new[indices] *= -1.0
        return _autocorr_fast(h_new)


def _thue_morse(n):
    t = [1]
    while len(t) < n:
        t += [-x for x in t]
    return np.array(t[:n])


def construct():
    try:
        t0 = time.time()
        deadline = t0 + 140.0
        rng = np.random.default_rng(20240517)

        idx = np.arange(N)
        inits = [
            _thue_morse(N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.sign(np.cos(2 * np.pi * idx ** 2 / (2 * N))),
            np.sign(np.cos(2 * np.pi * idx ** 2 / (N + 1))),
            np.sign(np.cos(2 * np.pi * idx ** 2 / (3 * N))),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.sign(np.cos(2 * np.pi * idx / N)),
            np.sign(np.sin(2 * np.pi * idx / N)),
            np.sign(np.cos(2 * np.pi * 2 * idx / N)),
            rng.choice(np.array([-1.0, 1.0]), size=N),
        ]
        best_h = None
        best_hard = 1e9
        for h in inits:
            h = np.sign(h)
            if np.all(h == 0):
                h = np.ones(N)
            s = _hard_score(h)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()

        # Phase 1: Greedy coordinate descent
        h = best_h.copy()
        cur_hard = best_hard
        ac = _autocorr_fast(h)
        
        max_p1 = 30000
        for _iter in range(max_p1):
            if time.time() > deadline - 40.0:
                break
            
            # Find top 3 worst lags
            sidelobes = np.abs(ac[1:])
            top_k = np.argsort(sidelobes)[-3:][::-1]
            top_k = top_k + 1  # convert to actual lag indices
            
            best_candidate = None
            best_candidate_score = cur_hard
            best_candidate_ac = None
            
            for k_star in top_k:
                c_star = ac[k_star]
                max_i = N - k_star
                indices = np.arange(max_i)
                products = h[indices] * h[indices + k_star]
                
                if c_star > 0:
                    candidates = indices[products > 0]
                else:
                    candidates = indices[products < 0]
                
                if len(candidates) == 0:
                    continue
                
                # Test up to 30 candidates per lag
                num_tests = min(30, len(candidates))
                if len(candidates) > num_tests:
                    test_idx = rng.choice(candidates, num_tests, replace=False)
                else:
                    test_idx = candidates
                
                for i in test_idx:
                    i = int(i)
                    ac_new = _flip_ac(h, ac, i)
                    new_hard = float(np.max(np.abs(ac_new)) / ac_new[0])
                    if new_hard < best_candidate_score:
                        best_candidate_score = new_hard
                        best_candidate = i
                        best_candidate_ac = ac_new
            
            if best_candidate is not None and best_candidate_score < cur_hard:
                h[best_candidate] *= -1.0
                ac = best_candidate_ac
                cur_hard = best_candidate_score
                if cur_hard < best_hard:
                    best_hard = cur_hard
                    best_h = h.copy()
            else:
                # Random escape
                if rng.random() < 0.2:
                    i = int(rng.integers(0, N))
                    ac_new = _flip_ac(h, ac, i)
                    new_hard = float(np.max(np.abs(ac_new)) / ac_new[0])
                    if new_hard < cur_hard:
                        h[i] *= -1.0
                        ac = ac_new
                        cur_hard = new_hard
                        if cur_hard < best_hard:
                            best_hard = cur_hard
                            best_h = h.copy()

        # Phase 2: Simulated annealing
        h = best_h.copy()
        cur_hard = best_hard
        ac = _autocorr_fast(h)
        
        T = 0.3
        T_end = 0.005
        cooling = 0.997
        max_sa = 15000
        
        for _iter in range(max_sa):
            if time.time() > deadline - 20.0:
                break
            
            # Block size
            bs = int(2 ** rng.integers(0, 4))  # 1,2,4,8
            if bs > N:
                bs = N
            flip_idx = rng.choice(N, size=bs, replace=False)
            
            ac_new = _block_flip_ac(h, ac, flip_idx)
            new_hard = float(np.max(np.abs(ac_new)) / ac_new[0])
            
            if new_hard < cur_hard or rng.random() < np.exp(-(new_hard - cur_hard) / T):
                h[flip_idx] *= -1.0
                ac = ac_new
                cur_hard = new_hard
                if cur_hard < best_hard:
                    best_hard = cur_hard
                    best_h = h.copy()
            
            T *= cooling
            if T < T_end:
                T = T_end

        # Phase 3: Final polish - full coordinate descent with exact FFT
        h = best_h.copy()
        for _pass in range(50):
            if time.time() > deadline - 2.0:
                break
            any_improve = False
            ac = _autocorr_fast(h)
            dc = ac[0]
            if dc <= 1e-30:
                break
            cur_peak = float(np.max(np.abs(ac)) / dc)
            for i in range(N):
                ac_new = _flip_ac(h, ac, i)
                new_peak = float(np.max(np.abs(ac_new)) / ac_new[0])
                if new_peak < cur_peak - 1e-14:
                    h[i] *= -1.0
                    ac = ac_new
                    cur_peak = new_peak
                    any_improve = True
            if not any_improve:
                break

        final_hard = _hard_score(h)
        if final_hard < best_hard:
            best_hard = final_hard
            best_h = h.copy()

        return best_h
    except Exception:
        return np.ones(N)
