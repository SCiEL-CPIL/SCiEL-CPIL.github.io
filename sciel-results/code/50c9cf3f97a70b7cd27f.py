import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sequence of 50 nonnegative heights.
    Strategy: Local optimization of weights for small cluster supports.
    """
    t0 = time.time()
    deadline = 110.0  # Safety margin well under 120s (evaluator kills at 150s)
    
    def compute_score(h):
        """Compute the ratio: L2^2 / (L1 * sup_norm_autoconv)."""
        h = np.array(h, dtype=float)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        l2_sq = np.sum(h**2)
        # Autoconvolution (linear)
        conv = np.convolve(h, h, mode='full')
        conv_inf = np.max(conv)
        if conv_inf < 1e-15:
            return 0.0
        return l2_sq / (l1 * conv_inf)

    best_score = 0.0
    best_h = None
    
    # Candidate supports (indices of non-zero heights)
    # We focus on small clusters near the origin to maximize overlap/convolution peak
    # relative to L2 norm.
    supports = [
        [0, 1],
        [0, 2],
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 4],
        [0, 1, 2, 3, 4],
        [0, 2, 4],
        [0, 1, 2, 5],
        [0, 1, 3, 5],
        [0, 2, 4, 6],
    ]
    
    # Also try some shifted versions? The problem is on [-1/4, 1/4].
    # The indices 0..49 map to this interval.
    # Center is around index 24/25.
    # However, the current best scores come from clusters at the edge (index 0) or near it?
    # Let's check the current program: it puts weights at h[0], h[1], etc.
    # Why edge? If we put a cluster in the middle, the convolution is symmetric.
    # If we put it at the edge, the convolution is triangular-ish?
    # Actually, for non-negative h, the max of h*h is often at the center of the support.
    # Let's test supports centered around 0, and also shifted to center (24, 25).
    
    # Generate shifted supports
    shifted_supports = []
    for shift in [0, 1, 2, 24, 25]:
        for s in supports:
            new_s = [i + shift for i in s]
            if max(new_s) < N:
                shifted_supports.append(new_s)
    
    # Remove duplicates
    unique_supports = []
    seen = set()
    for s in shifted_supports:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_supports.append(s)
            
    # Optimization function for a given support
    def optimize_support(support):
        n = len(support)
        if n == 0:
            return 0.0, None
        
        # We optimize weights w_i for i in support.
        # Constraint: sum(w) = 1 is not strictly necessary for the ratio, but helps.
        # The ratio is scale invariant.
        # Let's optimize log-weights to ensure positivity.
        # Or just use bounds.
        
        def neg_ratio(log_w):
            if time.time() - t0 > deadline:
                return 1e10
            w = np.exp(log_w)
            # Normalize to sum 1 to avoid scaling issues in optimizer
            w = w / np.sum(w)
            
            h = np.zeros(N)
            for i, idx in enumerate(support):
                h[idx] = w[i]
            
            return -compute_score(h)
        
        # Initial guess: uniform weights
        log_w0 = np.zeros(n)
        
        # Bounds for log_w: -5 to 5 (weights 0.006 to 148)
        # Since we normalize, the scale of log_w doesn't matter, only relative.
        # But bounds help Nelder-Mead.
        bounds = [(-5, 5)] * n
        
        try:
            # Use Nelder-Mead for unconstrained optimization of log-weights
            res = minimize(neg_ratio, log_w0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success or res.fun < 0:
                log_w_opt = res.x
                w_opt = np.exp(log_w_opt)
                w_opt = w_opt / np.sum(w_opt)
                
                h = np.zeros(N)
                for i, idx in enumerate(support):
                    h[idx] = w_opt[i]
                
                score = compute_score(h)
                return score, h
        except Exception:
            pass
        return 0.0, None

    # Iterate over supports
    for s in unique_supports:
        if time.time() - t0 > deadline:
            break
        
        score, h = optimize_support(s)
        if score > best_score:
            best_score = score
            best_h = h
    
    # If optimization failed or timed out, fallback to a known good configuration
    if best_h is None or best_score < 1.0:
        # Fallback: 2-point cluster 2:1 at index 0,1
        best_h = np.zeros(N)
        best_h[0] = 2.0 / 3.0
        best_h[1] = 1.0 / 3.0
        best_score = compute_score(best_h)
        
    return list(best_h)
# EVOLVE-BLOCK-END
