import numpy as np
import time
from scipy.optimize import minimize
import warnings

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

    def neg_score(h):
        return -score(h)

    def optimize_on_simplex(h0, max_iter=500):
        """Optimize using projected gradient on the simplex (l1=1, h>=0)."""
        h = np.maximum(h0, 0).copy()
        s = np.sum(h)
        if s < 1e-10:
            return h0, 0.0
        h = h / s
        best_h = h.copy()
        best_s = score(h)
        
        for it in range(max_iter):
            if time.time() > deadline - 3.0:
                break
            
            conv = np.convolve(h, h, mode='full')
            max_conv = np.max(conv)
            if max_conv < 1e-15:
                break
            peak_idx = np.argmax(conv)
            shift = peak_idx - (N - 1)
            
            # Gradient of log(ratio) = log(l2_sq) - log(max_conv) (since l1=1)
            # d/dh[i] log(l2_sq) = 2*h[i]/l2_sq
            # d/dh[i] log(max_conv): need subgradient of max
            l2_sq = np.sum(h**2)
            
            # Subgradient of max_conv w.r.t. h[i]
            d_peak = np.zeros(N)
            for i in range(N):
                if h[i] > 0:
                    if 0 <= i + shift < N:
                        d_peak[i] += h[i + shift]
                    if 0 <= i - shift < N:
                        d_peak[i] += h[i - shift]
            
            grad = 2.0 * h / l2_sq - d_peak / max_conv
            
            # Projected gradient step on simplex
            eta = 0.1
            h_new = h + eta * grad
            # Project onto simplex: find t such that h_new - t >= 0 and sum = 1
            # Simple approach: clip and renormalize
            h_new = np.maximum(h_new, 0)
            s_new = np.sum(h_new)
            if s_new < 1e-15:
                break
            h_new = h_new / s_new
            
            s_new_val = score(h_new)
            if s_new_val > best_s:
                best_s = s_new_val
                best_h = h_new.copy()
            
            # Adaptive step size
            if s_new_val < best_s * 0.99:
                eta *= 0.5
            elif it > 50 and abs(s_new_val - best_s) < 1e-12:
                break
            
            h = h_new
        
        return best_h, best_s

    def optimize_with_scipy(h0):
        """Use scipy minimize with SLSQP on the simplex."""
        h = np.maximum(h0, 0).copy()
        s = np.sum(h)
        if s < 1e-10:
            return h0, 0.0
        h = h / s
        
        def neg_ratio(x):
            x = np.maximum(x, 0)
            s = np.sum(x)
            if s < 1e-10:
                return 1e10
            x = x / s
            return -score(x)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0, 1)] * N
        
        try:
            result = minimize(neg_ratio, h, method='SLSQP', 
                             bounds=bounds, constraints=constraints,
                             options={'maxiter': 200, 'ftol': 1e-12})
            if result.success:
                h_opt = np.maximum(result.x, 0)
                s_opt = np.sum(h_opt)
                if s_opt > 1e-10:
                    h_opt = h_opt / s_opt
                    s_val = score(h_opt)
                    return h_opt, s_val
        except:
            pass
        
        return h, score(h)

    def find_sidon_set(k, seed=0):
        """Find a k-element Sidon-like set via greedy + local search."""
        rng = np.random.RandomState(seed)
        best_set = None
        best_mc = np.inf
        
        for attempt in range(30):
            S = set()
            # Greedy: add elements that minimize max autoconvolution
            for _ in range(k):
                candidates = [j for j in range(N) if j not in S]
                if not candidates:
                    break
                best_cand = None
                best_mc_cand = np.inf
                for j in candidates:
                    S_test = S | {j}
                    h_tmp = np.zeros(N)
                    h_tmp[list(S_test)] = 1.0
                    conv = np.convolve(h_tmp, h_tmp, mode='full')
                    mc = np.max(conv)
                    if mc < best_mc_cand:
                        best_mc_cand = mc
                        best_cand = j
                if best_cand is not None:
                    S.add(best_cand)
            
            S = sorted(S)
            h_tmp = np.zeros(N)
            h_tmp[S] = 1.0
            conv = np.convolve(h_tmp, h_tmp, mode='full')
            mc = np.max(conv)
            
            # Local search: swap elements
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for i in range(len(S)):
                    for j in range(N):
                        if j in S:
                            continue
                        S_new = sorted(S[:i] + [j] + S[i+1:])
                        h_tmp = np.zeros(N)
                        h_tmp[S_new] = 1.0
                        conv = np.convolve(h_tmp, h_tmp, mode='full')
                        mc_new = np.max(conv)
                        if mc_new < mc:
                            mc = mc_new
                            S = S_new
                            improved = True
                            break
                    if improved:
                        break
            
            if mc < best_mc:
                best_mc = mc
                best_set = S
        
        return best_set if best_set else sorted(rng.choice(N, k, replace=False).tolist())

    # Strategy 1: Optimize on fixed Sidon-like supports
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Known good Sidon sets
    seed_sets = [
        [0, 1, 4, 10, 18, 23, 30],
        [0, 1, 4, 10, 18, 23, 31, 39],
        [0, 1, 4, 10, 18, 23, 31, 40, 47],
        [0, 1, 4, 10, 18, 23, 31, 40, 47, 49],
        [0, 2, 5, 11, 19, 24, 31],
        [0, 2, 5, 11, 19, 24, 31, 40],
    ]
    
    for S in seed_sets:
        if time.time() > deadline - 5.0:
            break
        h0 = np.zeros(N)
        h0[S] = 1.0 / len(S)
        
        # Try projected gradient
        h_opt, s = optimize_on_simplex(h0, max_iter=300)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()
        
        # Try scipy
        if time.time() < deadline - 10.0:
            h_opt2, s2 = optimize_with_scipy(h0)
            if s2 > best_score:
                best_score = s2
                best_h = h_opt2.copy()
    
    # Strategy 2: Find good Sidon sets and optimize
    for k in range(4, 12):
        if time.time() > deadline - 5.0:
            break
        for seed in range(3):
            if time.time() > deadline - 5.0:
                break
            S = find_sidon_set(k, seed=seed)
            h0 = np.zeros(N)
            h0[S] = 1.0 / len(S)
            h_opt, s = optimize_on_simplex(h0, max_iter=200)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
    
    # Strategy 3: Random restarts with optimization
    rng = np.random.RandomState(123)
    for iter in range(300):
        if time.time() > deadline - 5.0:
            break
        
        k = rng.randint(3, 10)
        S = sorted(rng.choice(N, k, replace=False).tolist())
        h0 = np.zeros(N)
        h0[S] = 1.0 / k
        # Add some noise to break symmetry
        h0 += rng.randn(N) * 0.01
        h0 = np.maximum(h0, 0)
        
        h_opt, s = optimize_on_simplex(h0, max_iter=150)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()
    
    # Strategy 4: Try concentrated distributions (few large values)
    for k in range(2, 8):
        if time.time() > deadline - 5.0:
            break
        for seed in range(5):
            if time.time() > deadline - 5.0:
                break
            rng2 = np.random.RandomState(seed * 100 + k)
            S = sorted(rng2.choice(N, k, replace=False).tolist())
            # Try geometric weights
            for ratio in [1.5, 2.0, 3.0]:
                h0 = np.zeros(N)
                weights = np.array([ratio ** i for i in range(k)])
                weights /= np.sum(weights)
                h0[S] = weights
                h_opt, s = optimize_on_simplex(h0, max_iter=200)
                if s > best_score:
                    best_score = s
                    best_h = h_opt.copy()
    
    # Final polish with scipy on best solution
    if time.time() < deadline - 5.0:
        h_opt, s = optimize_with_scipy(best_h)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()
    
    return list(best_h)
