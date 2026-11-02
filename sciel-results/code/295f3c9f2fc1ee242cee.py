"""Improved program: optimize amplitudes on a Sidon-like support to maximize the ratio."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 50 nonnegative step heights to maximize the ratio:
        (sum(h^2)) / (sum(h) * max(autoconvolution))
    where autoconvolution is computed on the discrete support.
    
    Strategy:
    1. Choose a subset of indices that form a Sidon set (B2 sequence) to minimize
       the number of pairs (i,j) with i+j = k, which controls the autoconvolution peaks.
    2. Optimize the amplitudes on this support via gradient ascent on the log-ratio.
    """
    deadline = time.time() + 140  # 140s to be safe under 150s limit
    
    # A good Sidon set (B2 sequence) of size ~12-15 within [0, 49]
    # Using the classic Sidon sequence: floor(n^2) type or known optimal
    # For 50 elements, we can pick a subset. Let's use a known B2 sequence.
    # Greedy B2 sequence: 0, 1, 3, 7, 12, 20, 30, 42, ...
    # But we want to maximize the ratio, so we need to think about what support is best.
    
    # Actually, let's think more carefully. The ratio is:
    #   R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # where h*h is the autoconvolution.
    
    # For a single non-zero element, R = 1/1 * 1/1 = 1 (since h*h has one element = h^2, 
    # ||h||_1 = h, ||h*h||_inf = h^2, so R = h^2/(h*h^2) = 1/h... wait let me recompute)
    # 
    # Let h have one element h_0. Then:
    # ||h||_2^2 = h_0^2
    # ||h||_1 = h_0
    # autoconvolution: (h*h)[k] = sum_i h[i]*h[k-i]
    # For one element at index 0: (h*h)[0] = h_0^2, all others 0.
    # ||h*h||_inf = h_0^2
    # R = h_0^2 / (h_0 * h_0^2) = h_0^2 / h_0^3 = 1/h_0
    # This goes to infinity as h_0 -> 0, but we need nonnegative and presumably the problem
    # normalizes or the ratio is scale-invariant? Let me check...
    
    # Actually wait: R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # If we scale h by c: ||ch||_2^2 = c^2||h||_2^2, ||ch||_1 = c||h||_1, ||ch*ch||_inf = c^2||h*h||_inf
    # So R(ch) = c^2||h||_2^2 / (c||h||_1 * c^2||h*h||_inf) = ||h||_2^2 / (||h||_1 * ||h*h||_inf) / c
    # So R scales as 1/c. This means we should make h as small as possible? That can't be right.
    # 
    # Let me re-read: "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
    # So it's ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # 
    # With scaling h -> c*h:
    # ||ch||_2^2 = c^2 ||h||_2^2
    # ||ch||_1 = c ||h||_1  
    # ||(ch)*(ch)||_inf = c^2 ||h*h||_inf
    # Ratio = c^2||h||_2^2 / (c||h||_1 * c^2||h*h||_inf) = ||h||_2^2/(||h||_1*||h*h||_inf) * (1/c)
    # 
    # So the ratio is NOT scale invariant - it's 1/c times the base ratio.
    # This means making h smaller increases the ratio without bound? That seems problematic.
    # 
    # Unless there's an implicit normalization like sum(h) = 1 or max(h) = 1.
    # Given the context "choose 50 nonnegative heights", perhaps there's a constraint I'm missing.
    # 
    # Looking at the seed: [1.0]*N gives score 1.0. Let me compute for h = [1,1,...,1] (N=50):
    # ||h||_2^2 = 50
    # ||h||_1 = 50
    # autoconvolution: (h*h)[k] = number of pairs (i,j) with i+j=k and 0<=i,j<50
    # For k=0: 1 pair (0,0)
    # For k=1: 2 pairs
    # ...
    # For k=49: 50 pairs (max)
    # For k=50: 49 pairs
    # ...
    # ||h*h||_inf = 50 (at k=49)
    # Ratio = 50 / (50 * 50) = 50/2500 = 1/50 = 0.02
    # 
    # But the score is 1.0 for the seed, so the "score" must be normalized differently.
    # The task says "score is the lower bound on C2" and "search score is value/seed".
    # So seed value = 1.0 means the seed achieves some baseline, and we want to beat it.
    
    # Let me reconsider. Perhaps the problem has an implicit constraint like sum(h) = 1 or ||h||_1 = 1.
    # If sum(h) = 1, then for h = [1/50]*50:
    # ||h||_2^2 = 50*(1/50)^2 = 1/50
    # ||h||_1 = 1
    # autoconvolution max = 50*(1/50)^2 = 1/50
    # Ratio = (1/50)/(1 * 1/50) = 1
    # That gives ratio 1, matching the seed score of 1.0!
    
    # So the constraint is likely sum(h) = 1 (or equivalent normalization).
    
    # Let's verify with the 2-element example from observations:
    # h = [2/3, 1/3] embedded in 50 elements (rest zero)
    # sum = 1
    # ||h||_2^2 = (4/9) + (1/9) = 5/9
    # autoconvolution: 
    #   (h*h)[0] = (2/3)^2 = 4/9
    #   (h*h)[1] = 2*(2/3)*(1/3) = 4/9
    #   (h*h)[2] = (1/3)^2 = 1/9
    # ||h*h||_inf = 4/9
    # Ratio = (5/9)/(1 * 4/9) = 5/4 = 1.25
    # Score = 1.25/1.0 = 1.25? But observation says 1.1944...
    # 
    # 1.1944 = 43/36? No. 1.1944... let me check: maybe the indices matter.
    # If h[0]=2/3, h[1]=1/3, the autoconvolution is computed on the full 50-length vector.
    # (h*h)[0] = h[0]*h[0] = 4/9
    # (h*h)[1] = h[0]*h[1] + h[1]*h[0] = 4/9
    # (h*h)[2] = h[1]*h[1] = 1/9
    # Max is 4/9. Ratio = (5/9)/(4/9) = 1.25.
    # 
    # But observed score is 1.19444... = 43/36? 43/36 = 1.19444... Yes!
    # So maybe my understanding is slightly off. Let me just implement the optimization
    # with sum(h)=1 constraint and see what we get.
    
    # Define the objective: maximize log(||h||_2^2 / (||h||_1 * ||h*h||_inf))
    # subject to sum(h) = 1, h >= 0
    
    def compute_ratio(h):
        """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
        h = np.asarray(h, dtype=np.float64)
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        # Autoconvolution
        conv = np.convolve(h, h, mode='full')
        conv_max = np.max(conv)
        if conv_max < 1e-15:
            return 0.0
        return l2_sq / (l1 * conv_max)
    
    def neg_log_ratio(h):
        """Negative log of the ratio for minimization"""
        r = compute_ratio(h)
        if r <= 0:
            return 1e10
        return -np.log(r)
    
    # Start with uniform distribution
    h0 = np.ones(N) / N
    
    # Use SLSQP with constraint sum(h) = 1, h >= 0
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
    bounds = [(0.0, 1.0)] * N
    
    best_h = h0.copy()
    best_ratio = compute_ratio(best_h)
    
    # Try multiple initializations
    initializations = [
        h0,  # uniform
    ]
    
    # Add some structured initializations
    # Sparse initializations
    for support_size in [2, 3, 4, 5]:
        for _ in range(3):
            idx = np.random.choice(N, size=support_size, replace=False)
            h_init = np.zeros(N)
            h_init[idx] = 1.0 / support_size
            initializations.append(h_init)
    
    # Sidon-like initialization
    sidon_indices = [0, 1, 3, 7, 12, 20, 30, 42]
    h_sidon = np.zeros(N)
    h_sidon[sidon_indices] = 1.0 / len(sidon_indices)
    initializations.append(h_sidon)
    
    deadline = time.time() + 140
    
    for h_init in initializations:
        if time.time() > deadline - 20:
            break
        
        try:
            result = minimize(
                neg_log_ratio,
                h_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if result.success or result.fun < 1e9:
                h_opt = np.clip(result.x, 0, None)
                # Renormalize
                h_opt_sum = np.sum(h_opt)
                if h_opt_sum > 1e-15:
                    h_opt = h_opt / h_opt_sum
                
                r = compute_ratio(h_opt)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Local polish: try small perturbations
    if time.time() < deadline - 10:
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            h_pert = best_h + np.random.randn(N) * 0.01
            h_pert = np.clip(h_pert, 0, None)
            h_pert_sum = np.sum(h_pert)
            if h_pert_sum > 1e-15:
                h_pert = h_pert / h_pert_sum
                r = compute_ratio(h_pert)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h_pert.copy()
    
    return list(best_h)
# EVOLVE-BLOCK-END
