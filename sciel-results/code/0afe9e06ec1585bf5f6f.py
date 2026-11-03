import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s deadline, 10s margin before 140s timeout
    
    # The ratio is scale-invariant, so we can fix L1 norm = 1 (sum h = 1)
    # Then ratio = ||h||_2^2 / ||h*h||_inf
    # = sum(h_i^2) / max_j sum_i h_i h_{j-i} (with proper indexing)
    
    # Let's define the autoconvolution properly.
    # For a sequence h_0, h_1, ..., h_{N-1} on equal intervals,
    # the autoconvolution at lag k is sum_i h_i h_{i+k} for k >= 0
    # (or symmetrically). The supremum over all lags.
    
    # Actually, the autoconvolution of h with itself:
    # (h * h)[k] = sum_i h[i] h[k-i] for k in -(N-1)..(N-1)
    # The supremum norm is max over all k.
    
    # Let's compute this carefully.
    
    def compute_ratio(h):
        """Compute ||h||_2^2 / ||h*h||_inf for nonnegative h with sum=1."""
        h = np.asarray(h, dtype=np.float64)
        # Make sure h is nonnegative and sum to 1
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-15:
            return 0.0
        h = h / s
        l2_sq = np.sum(h**2)
        # Autoconvolution
        conv = np.convolve(h, h, mode='full')
        sup_norm = np.max(conv)
        if sup_norm < 1e-15:
            return 0.0
        return l2_sq / sup_norm
    
    # Strategy: Use gradient-based optimization with L1 constraint
    # Use log-barrier or project onto simplex
    # We want to maximize f(h) = sum(h_i^2) / max_k (h*h)_k
    # subject to sum(h_i) = 1, h_i >= 0
    
    # Let's use a parameterization: h_i = exp(x_i) / sum(exp(x_j))
    # This automatically satisfies h_i > 0 and sum = 1
    # Then we can do gradient descent on x
    
    # Initialize with some starting points
    def score(x):
        h = np.exp(x)
        h = h / np.sum(h)
        return compute_ratio(h)
    
    # Generate multiple random initializations
    best_score = 0.0
    best_h = np.ones(N) / N
    
    # Start with uniform
    best_score = score(np.zeros(N))
    
    # Try some structured initializations
    # Sparse vectors
    for k in range(2, 20):
        for trial in range(20):
            idx = np.random.choice(N, size=k, replace=False)
            h_init = np.zeros(N)
            h_init[idx] = 1.0
            x_init = np.log(h_init + 1e-15)
            # Handle zeros
            x_init = np.where(h_init > 0, np.log(h_init + 1e-15), -50)
            s = score(x_init)
            if s > best_score:
                best_score = s
                best_h = h_init.copy()
    
    # Gradient-based optimization using scipy
    def neg_score(x):
        if time.time() > deadline:
            return 0.0
        return -score(x)
    
    # Try optimization from multiple starting points
    num_starts = 20
    for start_idx in range(num_starts):
        if time.time() > deadline - 10.0:
            break
        
        # Generate random starting point
        if start_idx == 0:
            x0 = np.zeros(N)
        elif start_idx == 1:
            # Sparse
            idx = np.random.choice(N, size=10, replace=False)
            x0 = np.full(N, -50.0)
            x0[idx] = 0.0
        elif start_idx == 2:
            # Sidon-like
            # Try to construct a Sidon set greedily
            sidon = [0]
            for i in range(1, N):
                can_add = True
                for j in range(len(sidon)):
                    for k in range(j+1, len(sidon)):
                        if i - sidon[j] == sidon[k] - sidon[j] and sidon[k] != i:
                            pass
                        # Check if i + sidon[j] equals sidon[k] + sidon[l] for some existing
                    # Simpler: check all pair sums
                # Use a different check
                sums = set()
                for a in range(len(sidon)):
                    for b in range(a+1, len(sidon)):
                        sums.add(sidon[a] + sidon[b])
                        sums.add(abs(sidon[a] - sidon[b]))
                ok = True
                for a in range(len(sidon)):
                    if i + sidon[a] in sums:
                        ok = False
                        break
                    if abs(i - sidon[a]) in sums and sidon[a] != 0:
                        ok = False
                        break
                if ok:
                    sidon.append(i)
            x0 = np.full(N, -50.0)
            for s in sidon:
                x0[s] = 0.0
        else:
            x0 = np.random.randn(N) * 2.0
        
        # Optimize
        try:
            res = minimize(neg_score, x0, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 500})
            if res.fun < -best_score:
                h_opt = np.exp(res.x)
                h_opt = h_opt / np.sum(h_opt)
                s = compute_ratio(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt
        except Exception:
            pass
    
    # Try a few more random restarts if time permits
    for extra in range(10):
        if time.time() > deadline - 5.0:
            break
        x0 = np.random.randn(N) * 3.0
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead', options={'maxiter': 500, 'maxfev': 1000})
            if res.fun < -best_score:
                h_opt = np.exp(res.x)
                h_opt = h_opt / np.sum(h_opt)
                s = compute_ratio(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt
        except Exception:
            pass
    
    # Ensure best_h is nonnegative and valid
    best_h = np.maximum(best_h, 0)
    s = np.sum(best_h)
    if s > 1e-15:
        best_h = best_h / s
    
    return list(best_h)
# EVOLVE-BLOCK-END
