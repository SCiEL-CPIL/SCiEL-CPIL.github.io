import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    def score(h):
        """Compute the ratio: sum(h^2) / (sum(h) * max(autoconv(h)))"""
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 0.0
        return l2_sq / (l1 * max_conv)

    def optimize_amplitudes(S, max_iter=500, tol=1e-12):
        """Optimize amplitudes for a fixed support set S using multiplicative updates."""
        k = len(S)
        if k == 0:
            return np.zeros(N), 0.0
        
        S_set = set(S)
        h = np.zeros(N)
        h[S] = 1.0 / k
        
        best_h = h.copy()
        best_s = score(h)
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Compute convolution and find peak
            conv = np.convolve(h, h, mode='full')
            max_conv = np.max(conv)
            if max_conv < 1e-15:
                break
            
            peak_idx = np.argmax(conv)
            shift = peak_idx - (N - 1)
            
            l1 = np.sum(h)
            l2_sq = np.sum(h**2)
            
            if l2_sq < 1e-15 or l1 < 1e-15:
                break
            
            # Gradient of log(ratio) = log(l2_sq) - log(l1) - log(max_conv)
            # d/dh[i] log(l2_sq) = 2*h[i]/l2_sq
            # d/dh[i] log(l1) = 1/l1
            # d/dh[i] log(max_conv) = d_peak[i] / max_conv
            # where d_peak[i] is derivative of peak conv value w.r.t. h[i]
            
            d_peak = np.zeros(N)
            for i in S:
                j1 = i + shift
                j2 = i - shift
                if 0 <= j1 < N and j1 in S_set:
                    d_peak[i] += h[j1]
                if 0 <= j2 < N and j2 in S_set:
                    d_peak[i] += h[j2]
            
            # Multiplicative update with adaptive step size
            eta = 0.3
            for i in S:
                grad_val = 2.0 * h[i] / l2_sq - 1.0 / l1 - d_peak[i] / max_conv
                h[i] *= np.exp(eta * grad_val)
            
            # Renormalize (scale invariance means this doesn't change ratio,
            # but keeps values in a reasonable range)
            h_sum = np.sum(h)
            if h_sum < 1e-15:
                break
            h = h / h_sum
            
            s = score(h)
            if s > best_s:
                best_s = s
                best_h = h.copy()
            
            if it > 20 and abs(s - best_s) < tol:
                break
        
        return best_h, best_s

    def find_sidon_set(k, seed=0, max_swaps=200):
        """Find a k-element set that minimizes the max autoconvolution multiplicity.
        Uses local search with swaps."""
        rng = np.random.RandomState(seed)
        
        # Start with a spread-out initialization
        base = np.linspace(0, N-1, k).astype(int)
        # Add some randomness
        for attempt in range(20):
            S = sorted(set(base.tolist()))
            while len(S) < k:
                S.append(rng.randint(0, N))
                S = sorted(set(S))
            
            # Compute initial max conv
            h_tmp = np.zeros(N)
            h_tmp[S] = 1.0
            conv = np.convolve(h_tmp, h_tmp, mode='full')
            best_max_conv = np.max(conv)
            
            # Local search
            for swap_iter in range(max_swaps):
                if time.time() > deadline - 5.0:
                    break
                
                improved = False
                for i in range(k):
                    if time.time() > deadline - 5.0:
                        break
                    for j in range(N):
                        if j in S:
                            continue
                        S_new = sorted(S[:i] + [j] + S[i+1:])
                        h_tmp2 = np.zeros(N)
                        h_tmp2[S_new] = 1.0
                        conv2 = np.convolve(h_tmp2, h_tmp2, mode='full')
                        mc = np.max(conv2)
                        if mc < best_max_conv:
                            best_max_conv = mc
                            S = S_new
                            improved = True
                            break
                    if improved:
                        break
                
                if not improved:
                    break
            
            if best_max_conv <= 2:
                return S
        
        # Return the best found
        return S

    def coordinate_ascent(h, max_rounds=50):
        """Coordinate ascent refinement: try adjusting each element."""
        h = h.copy()
        best_s = score(h)
        support = list(np.where(h > 1e-12)[0])
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            improved = False
            for i in support:
                if time.time() > deadline - 3.0:
                    break
                old_val = h[i]
                # Try several perturbations
                for frac in [0.5, 0.7, 0.8, 1.2, 1.5, 2.0, 0.3]:
                    h_trial = h.copy()
                    h_trial[i] = old_val * frac
                    s = score(h_trial)
                    if s > best_s + 1e-12:
                        best_s = s
                        h = h_trial
                        improved = True
                        break
            if not improved:
                break
        
        return h, best_s

    def scipy_polish(h_init, support):
        """Use scipy minimize for high-precision polish on the best solution."""
        if len(support) == 0:
            return h_init, score(h_init)
        
        k = len(support)
        
        def neg_ratio(params):
            h = np.zeros(N)
            h[support] = np.maximum(params, 0)
            s = np.sum(h)
            if s < 1e-15:
                return 0.0
            h = h / s
            return -score(h)
        
        h_init_norm = h_init.copy()
        s_init = np.sum(h_init_norm)
        if s_init > 0:
            h_init_norm = h_init_norm / s_init
        
        x0 = h_init_norm[support].copy()
        
        try:
            res = minimize(neg_ratio, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            
            h_res = np.zeros(N)
            h_res[support] = np.maximum(res.x, 0)
            s_res = np.sum(h_res)
            if s_res > 0:
                h_res = h_res / s_res
            
            s_res = score(h_res)
            if s_res > score(h_init):
                return h_res, s_res
        except:
            pass
        
        return h_init, score(h_init)

    # Main search
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Phase 1: Try structured Sidon-like supports with optimized amplitudes
    seed_sets = [
        [0, 1, 4, 10, 18, 23, 30],
        [0, 1, 4, 10, 18, 23, 31, 39],
        [0, 1, 4, 10, 18, 23, 31, 40, 47],
        [0, 1, 4, 10, 18, 23, 31, 40, 47, 49],
        [0, 2, 5, 11, 19, 25, 33, 42],
        [0, 1, 3, 7, 12, 18, 26, 35, 45],
        [0, 2, 5, 9, 14, 20, 27, 35, 44],
        [0, 1, 4, 10, 18, 23, 30, 39, 47, 49],
        [0, 2, 5, 11, 19, 25, 33, 42, 48],
        [0, 1, 3, 7, 12, 18, 26, 35, 45, 49],
    ]
    
    for S in seed_sets:
        if time.time() > deadline - 15.0:
            break
        h_opt, s = optimize_amplitudes(S)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()
    
    # Phase 2: Search for good Sidon sets of various sizes
    for k in range(3, 15):
        if time.time() > deadline - 15.0:
            break
        for seed in range(5):
            if time.time() > deadline - 15.0:
                break
            S = find_sidon_set(k, seed=seed)
            h_opt, s = optimize_amplitudes(S, max_iter=300)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
    
    # Phase 3: Random restarts with support selection and amplitude optimization
    rng = np.random.RandomState(42)
    for it in range(1500):
        if time.time() > deadline - 15.0:
            break
        
        k = rng.randint(3, 14)
        S = sorted(rng.choice(N, k, replace=False).tolist())
        
        # Quick check: only proceed if max conv multiplicity is low
        h_check = np.zeros(N)
        h_check[S] = 1.0
        conv_check = np.convolve(h_check, h_check, mode='full')
        if np.max(conv_check) > 3:
            continue
        
        h_opt, s = optimize_amplitudes(S, max_iter=150)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()
    
    # Phase 4: Coordinate ascent polish on the best solution
    if best_score > 0:
        best_h, best_score = coordinate_ascent(best_h, max_rounds=100)
        
        # Additional refinement: try adding/removing support points
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            support = list(np.where(best_h > 1e-12)[0])
            improved = False
            
            # Try removing a point
            if len(support) > 3:
                for i in support:
                    if time.time() > deadline - 5.0:
                        break
                    h_trial = best_h.copy()
                    h_trial[i] = 0
                    s = score(h_trial)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_h = h_trial
                        improved = True
                        break
            
            # Try adding a point
            if not improved and len(support) < 20:
                for j in range(N):
                    if j in support:
                        continue
                    if time.time() > deadline - 5.0:
                        break
                    h_trial = best_h.copy()
                    h_trial[j] = 0.01
                    s = score(h_trial)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_h = h_trial
                        improved = True
                        break
            
            if not improved:
                break
        
        # Final coordinate ascent
        best_h, best_score = coordinate_ascent(best_h, max_rounds=200)
        
        # Phase 5: High-precision polish with scipy
        if time.time() < deadline - 10.0:
            support = list(np.where(best_h > 1e-12)[0])
            if len(support) > 0 and len(support) < 25:
                best_h, best_score = scipy_polish(best_h, support)
                
                # Another round of coordinate ascent after scipy
                best_h, best_score = coordinate_ascent(best_h, max_rounds=100)
    
    # Ensure valid output
    if best_score <= 0:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    else:
        best_h = np.maximum(best_h, 0)
        s = np.sum(best_h)
        if s > 0:
            best_h = best_h / s
        else:
            best_h = np.zeros(N)
            best_h[0] = 1.0
    
    return best_h.tolist()
