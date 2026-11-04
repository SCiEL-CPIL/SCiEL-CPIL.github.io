"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def autoconv(h):
        return np.convolve(h, h, mode='full')

    def score_fn(h):
        """Return the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf). Higher is better."""
        h = np.maximum(h, 0.0)
        norm2_sq = np.sum(h * h)
        if norm2_sq < 1e-30:
            return 0.0
        norm1 = np.sum(h)
        if norm1 < 1e-30:
            return 0.0
        ac = autoconv(h)
        ac_max = np.max(np.abs(ac))
        if ac_max < 1e-30:
            return 0.0
        return norm2_sq / (norm1 * ac_max)

    def _get_sidon_set(n, k, seed=42):
        """Find a subset of {0,...,n-1} of size k with distinct pairwise sums (Sidon set)."""
        rng = np.random.RandomState(seed)
        best_set = None
        best_size = 0
        
        for attempt in range(300):
            if time.time() > deadline - 15:
                break
            s = [0]
            sums = {0}
            for _ in range(k - 1):
                found = False
                for cand in range(1, n):
                    if cand in s:
                        continue
                    new_sums = set(sums)
                    ok = True
                    for idx in s:
                        ns = cand + idx
                        if ns in new_sums:
                            ok = False
                            break
                        new_sums.add(ns)
                    if ok:
                        ns = cand + cand
                        if ns in new_sums:
                            ok = False
                        else:
                            new_sums.add(ns)
                    if ok:
                        s.append(cand)
                        sums = new_sums
                        found = True
                        break
                if not found:
                    break
            if len(s) > best_size:
                best_size = len(s)
                best_set = s[:]
                if best_size >= k:
                    break
        
        if best_set is None:
            best_set = list(range(k))
        return best_set[:k]

    def _optimize_sparse_amplitudes(support, max_iter=200):
        """Optimize amplitudes on a given support to maximize the ratio."""
        m = len(support)
        if m < 2:
            return None, 0.0
        
        h = np.zeros(N)
        h[support] = 1.0 / np.sqrt(m)
        best_h = h.copy()
        best_score = score_fn(h)
        
        try:
            from scipy.optimize import minimize_scalar
        except ImportError:
            return best_h, best_score
        
        amps = np.ones(m) / np.sqrt(m)
        current_score = best_score
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(m):
                def obj_i(x_i):
                    a = amps.copy()
                    a[i] = max(x_i, 0.0)
                    h_tmp = np.zeros(N)
                    h_tmp[support] = a
                    return -score_fn(h_tmp)
                
                try:
                    res = minimize_scalar(obj_i, bounds=(0.0, 1.0), method='bounded',
                                         options={'maxiter': 30})
                    if res.success and res.fun < -current_score + 1e-12:
                        amps[i] = max(res.x, 0.0)
                        h[support] = amps
                        s = score_fn(h)
                        if s > current_score:
                            current_score = s
                            best_h = h.copy()
                            improved = True
                except Exception:
                    pass
            
            if not improved:
                break
        
        return best_h, best_score

    best_h = None
    best_score = 0.0

    # Phase 1: Try Sidon sets of various sizes
    for k in range(3, 16):
        if time.time() > deadline - 30:
            break
        
        sidon = _get_sidon_set(N, k)
        if len(sidon) < 3:
            continue
        
        h_opt, s_opt = _optimize_sparse_amplitudes(sidon)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        
        # Try perturbations
        for trial in range(3):
            if time.time() > deadline - 30:
                break
            support_pert = sidon[:]
            idx_rep = np.random.randint(0, len(support_pert))
            old_val = support_pert[idx_rep]
            new_val = None
            for offset in [-2, -1, 1, 2, -3, 3]:
                cand = old_val + offset
                if 0 <= cand < N and cand not in support_pert:
                    new_val = cand
                    break
            if new_val is not None:
                support_pert[idx_rep] = new_val
            h_opt, s_opt = _optimize_sparse_amplitudes(support_pert)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # Phase 2: Try dense Gaussian signals as fallback
    if time.time() < deadline - 15:
        x = np.linspace(-0.25, 0.25, N)
        for alpha in [20, 50, 100, 200, 500, 1000]:
            if time.time() > deadline - 10:
                break
            h = np.exp(-alpha * x * x)
            h = h / np.sqrt(np.sum(h * h))
            s = score_fn(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()

    # Phase 3: Coordinate descent polish
    if best_h is not None and time.time() < deadline - 3:
        h_curr = best_h.copy()
        curr_score = score_fn(h_curr)

        for step in [0.002, 0.001, 0.0005, 0.0002, 0.0001]:
            if time.time() > deadline - 2:
                break
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for i in range(N):
                    h_trial = h_curr.copy()
                    h_trial[i] += step
                    h_trial = np.maximum(h_trial, 0.0)
                    n2 = np.sqrt(np.sum(h_trial * h_trial))
                    if n2 > 1e-30:
                        h_trial = h_trial / n2
                    s = score_fn(h_trial)
                    if s > curr_score + 1e-14:
                        h_curr = h_trial
                        curr_score = s
                        improved = True
                        continue
                    h_trial = h_curr.copy()
                    h_trial[i] -= step
                    h_trial = np.maximum(h_trial, 0.0)
                    n2 = np.sqrt(np.sum(h_trial * h_trial))
                    if n2 > 1e-30:
                        h_trial = h_trial / n2
                    s = score_fn(h_trial)
                    if s > curr_score + 1e-14:
                        h_curr = h_trial
                        curr_score = s
                        improved = True

        if curr_score > best_score:
            best_score = curr_score
            best_h = h_curr.copy()

    if best_h is None:
        return [1.0] * N

    return [float(x) for x in best_h]

# EVOLVE-BLOCK-END
