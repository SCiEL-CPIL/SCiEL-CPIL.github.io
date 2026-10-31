import numpy as np
import time
from scipy.optimize import minimize
import itertools
import random

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for h >= 0.
    """
    h = np.asarray(h, dtype=np.float64)
    # Ensure non-negativity
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Efficient autoconvolution using FFT
    # h*h has length 2N-1
    # Zero pad to power of 2 for FFT efficiency
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    hh = np.fft.irfft(H * H, n=n_fft)
    hh_max = np.max(hh)
    
    if hh_max < 1e-300:
        return 0.0
        
    return l2_sq / (l1 * hh_max)

def optimize_weights_for_support(support):
    """
    Given a set of indices (support), optimize the weights h_i > 0 to maximize the ratio.
    We fix sum(h) = 1 due to scale invariance.
    Objective: max sum(h^2) / ||h*h||_inf
    Variables: h_i for i in support, with constraint sum(h_i) = 1, h_i >= 0.
    """
    K = len(support)
    if K == 0:
        return 0.0, np.zeros(N)
    
    # Initial guess: uniform weights on support
    h0 = np.zeros(N)
    h0[support] = 1.0 / K
    
    def objective_and_grad(x):
        # x are the weights for the support
        # Ensure non-negativity and normalize? 
        # L-BFGS-B handles bounds, but normalization is tricky.
        # Let's use the trick: maximize f(x) = sum(x^2) / ||c(x)||_inf subject to sum(x)=1, x>=0.
        # Instead of constraining sum(x)=1 explicitly in minimize (which requires SLSQP or similar),
        # we can just maximize the ratio directly. The ratio is homogeneous of degree 0.
        # So we can just maximize R(x) over x >= 0.
        # R(x) = sum(x^2) / ||c(x)||_inf
        # This is not convex. But for a fixed support, it's a small problem.
        
        # Construct full h
        h = np.zeros(N)
        h[support] = x
        
        l1 = np.sum(h)
        if l1 < 1e-300:
            return 0.0, np.zeros(K)
            
        l2_sq = np.sum(h * h)
        
        # FFT for autocorrelation
        n_fft = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h, n=n_fft)
        hh = np.fft.irfft(H * H, n=n_fft)
        hh_max = np.max(hh)
        
        if hh_max < 1e-300:
            return 0.0, np.zeros(K)
            
        val = l2_sq / (l1 * hh_max)
        
        # Gradient calculation is complex for the max term. 
        # Let's use numerical gradient or just rely on L-BFGS-B's finite diff?
        # Finite diff is slow but K is small (<= 15).
        # Actually, let's just use a simpler local search or Nelder-Mead for the weights?
        # Nelder-Mead doesn't use gradients and is robust for small K.
        
        return -val, None # We will use Nelder-Mead which doesn't need grad

    # Since K is small, Nelder-Mead is a good choice for the weight optimization.
    try:
        res = minimize(
            objective_and_grad, 
            h0[support], 
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        
        h_opt = np.zeros(N)
        h_opt[support] = np.maximum(res.x, 0)
        
        # Normalize to ensure non-zero
        s = np.sum(h_opt)
        if s > 0:
            h_opt /= s
            
        score = compute_ratio(h_opt)
        return score, h_opt
    except Exception:
        return 0.0, np.zeros(N)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = -np.inf
    
    # Strategy:
    # 1. Explore sparse supports.
    # 2. For each support, optimize the weights.
    # 3. Use a mix of random supports and structured supports (Sidon-like).
    
    # Phase 1: Random supports of varying sparsity
    # We'll try K from 3 to 20.
    # For each K, try a few random sets.
    
    # Also include the full support (K=50) but with a good initial guess?
    # The current program does full support. Let's see if sparse is better.
    
    # Pre-compute some structured supports
    structured_supports = []
    
    # Uniformly spaced
    for K in [5, 10, 15, 20, 25, 30]:
        idx = np.linspace(0, N-1, K, dtype=int)
        structured_supports.append(idx)
        
    # Random Sidon-like (greedy)
    def greedy_sidon(K, seed):
        r = np.random.default_rng(seed)
        support = [0]
        while len(support) < K:
            # Pick a random index that maximizes min distance to existing?
            # Or just random with conflict check?
            # Simple random is fine for diversity.
            candidates = [i for i in range(N) if i not in support]
            if not candidates:
                break
            # Prefer larger gaps?
            # Just pick random for now to ensure diversity
            idx = r.choice(candidates)
            support.append(idx)
        return np.array(sorted(support))
    
    for i in range(20):
        for K in [5, 10, 15]:
            structured_supports.append(greedy_sidon(K, 42 + i*K))
            
    # Add full support as a fallback
    full_support = np.arange(N)
    
    # List of supports to try
    supports_to_try = []
    
    # 1. Structured
    for s in structured_supports:
        supports_to_try.append(s)
        
    # 2. Random supports
    # We need to be careful not to spend too much time.
    # Nelder-Mead on K=15 is fast. K=50 is slow.
    # Let's limit K to 25 for random search to keep it fast.
    
    num_random = 100
    for i in range(num_random):
        K = rng.integers(3, 25)
        idx = rng.choice(N, size=K, replace=False)
        supports_to_try.append(np.sort(idx))
        
    # Shuffle
    rng.shuffle(supports_to_try)
    
    # Add full support at the end, only if time permits
    # Actually, let's try a few full support optimizations with different inits
    # using the current method (L-BFGS-B) if time allows.
    
    # Iterate through supports
    for support in supports_to_try:
        if time.time() > deadline - 10.0:
            break
            
        score, h_opt = optimize_weights_for_support(support)
        
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
            
    # If we have time, try to refine the best found solution using L-BFGS-B on full support
    # starting from the best sparse solution (zero-padded).
    if time.time() < deadline - 30.0 and best_h is not None:
        
        def neg_ratio_full(h):
            return -compute_ratio(h)
            
        # Use the best sparse solution as init
        # It's sparse, so many zeros. L-BFGS-B might struggle with non-differentiability at 0?
        # But it handles bounds.
        
        try:
            res = minimize(
                neg_ratio_full, 
                best_h, 
                method='L-BFGS-B', 
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'maxfun': 5000, 'disp': False}
            )
            
            candidate_h = np.maximum(res.x, 0)
            score = compute_ratio(candidate_h)
            if score > best_score:
                best_score = score
                best_h = candidate_h.copy()
        except Exception:
            pass
            
    # If no valid result, return uniform
    if best_h is None:
        best_h = np.ones(N) / N
        
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
