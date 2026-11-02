import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def score(h):
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

    def score_fast(h):
        """Faster score using the fact that we can compute conv efficiently."""
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

    def optimize_amplitudes(S, max_iter=200, tol=1e-10):
        """Optimize amplitudes for a fixed support set S using iterative reweighting.
        We want to maximize l2_sq / (l1 * max_conv) subject to h_i >= 0, support = S.
        Since it's scale invariant, fix l1 = 1.
        """
        k = len(S)
        if k == 0:
            return np.zeros(N), 0.0
        h = np.zeros(N)
        h[S] = 1.0 / k
        h = h / np.sum(h)
        
        best_h = h.copy()
        best_s = score(h)
        
        for it in range(max_iter):
            if time.time() > deadline - 3.0:
                break
            
            # Compute convolution
            conv = np.convolve(h, h, mode='full')
            max_conv = np.max(conv)
            if max_conv < 1e-15:
                break
            
            # The peak index in the full convolution
            peak_idx = np.argmax(conv)
            shift = peak_idx - (N - 1)
            
            # For each i in S, compute the contribution to the peak
            # conv[peak_idx] = sum_j h[j] * h[j+shift] (with bounds)
            # d(max_conv)/dh[i] for i in S:
            # = h[i+shift] if i+shift in [0,N) and (i+shift) in S
            # + h[i-shift] if i-shift in [0,N) and (i-shift) in S
            
            d_peak = np.zeros(N)
            for i in S:
                if 0 <= i + shift < N:
                    if (i + shift) in S:
                        d_peak[i] += h[i + shift]
                if 0 <= i - shift < N:
                    if (i - shift) in S:
                        d_peak[i] += h[i - shift]
            
            # Gradient of log(ratio) w.r.t. h[i] for i in S:
            # d/dh[i] log(l2_sq) - d/dh[i] log(l1) - d/dh[i] log(max_conv)
            # = 2*h[i]/l2_sq - 1/l1 - d_peak[i]/max_conv
            l1 = np.sum(h)
            l2_sq = np.sum(h**2)
            
            grad = np.zeros(N)
            for i in S:
                grad[i] = 2.0 * h[i] / l2_sq - 1.0 / l1 - d_peak[i] / max_conv
            
            # Multiplicative update: h[i] *= exp(eta * grad[i])
            # But we need to keep h[i] >= 0 and on support S
            eta = 0.5
            for i in S:
                h[i] *= np.exp(eta * grad[i])
            
            # Renormalize
            h_sum = np.sum(h)
            if h_sum < 1e-15:
                break
            h = h / h_sum
            
            s = score(h)
            if s > best_s:
                best_s = s
                best_h = h.copy()
            
            if it > 10 and abs(s - best_s) < tol:
                break
        
        return best_h, best_s

    def find_sidon_set(k, seed=0):
        """Find a k-element Sidon-like set (minimize max autoconvolution)."""
        rng = np.random.RandomState(seed)
        best_set = None
        best_max_conv = np.inf
        
        # Try multiple random initializations
        for attempt in range(50):
            S = sorted(rng.choice(N, k, replace=False).tolist())
            # Local search: try swapping elements
            improved = True
            while improved:
                improved = False
                for i in range(k):
                    for j in range(N):
                        if j in S:
                            continue
                        S_new = sorted(S[:i] + [j] + S[i+1:])
                        h_tmp = np.zeros(N)
                        h_tmp[S_new] = 1.0
                        conv = np.convolve(h_tmp, h_tmp, mode='full')
                        mc = np.max(conv)
                        if mc < best_max_conv:
                            best_max_conv = mc
                            best_set = S_new
                            S = S_new
                            improved = True
                            break
                    if improved:
                        break
        
        if best_set is None:
            best_set = sorted(rng.choice(N, k, replace=False).tolist())
        
        return best_set

    # Try various support sizes
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Start with known good Sidon-like sets
    seed_sets = [
        [0, 1, 4, 10, 18, 23, 30],
        [0, 1, 4, 10, 18, 23, 31, 39],
        [0, 1, 4, 10, 18, 23, 31, 40, 47],
        [0, 1, 4, 10, 18, 23, 31, 40, 47, 49],
    ]
    
    for S in seed_sets:
        if time.time() > deadline - 5.0:
            break
        h_opt, s = optimize_amplitudes(S)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()
    
    # Try various sizes with greedy construction
    for k in range(3, 16):
        if time.time() > deadline - 5.0:
            break
        for seed in range(5):
            if time.time() > deadline - 5.0:
                break
            S = find_sidon_set(k, seed=seed)
            h_opt, s = optimize_amplitudes(S)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
    
    # Random restarts with optimization
    rng = np.random.RandomState(42)
    for iter in range(500):
        if time.time() > deadline - 5.0:
            break
        
        k = rng.randint(3, 14)
        S = sorted(rng.choice(N, k, replace=False).tolist())
        h_opt, s = optimize_amplitudes(S, max_iter=100)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()
    
    # Final coordinate ascent polish
    for _ in range(200):
        if time.time() > deadline - 2.0:
            break
        improved = False
        support = list(np.where(best_h > 1e-12)[0])
        for i in support:
            for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                h_trial = best_h.copy()
                h_trial[i] = max(0, h_trial[i] + delta)
                s = score(h_trial)
                if s > best_score:
                    best_score = s
                    best_h = h_trial.copy()
                    improved = True
                    break
        if not improved:
            break
    
    return list(best_h)
