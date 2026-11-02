import numpy as np
import time

N = 50

def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _is_sidon(S):
    """Check if set S is a Sidon set (all sums i+j for i<=j are distinct)."""
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _find_max_sidon(N):
    """Greedy search for a large Sidon set in [0, N-1]."""
    # Start with a known good seed or greedy construction
    # Greedy: add smallest possible element that maintains Sidon property
    S = []
    for i in range(N):
        S_candidate = S + [i]
        if _is_sidon(S_candidate):
            S = S_candidate
    return S

def _optimize_weights(indices):
    """
    Optimize weights for a given set of indices to maximize ratio.
    Uses simple gradient ascent or random perturbation on weights.
    """
    K = len(indices)
    # Initialize with equal weights
    w = np.ones(K) / K
    h = np.zeros(N)
    for k, idx in enumerate(indices):
        h[idx] = w[k]
    
    best_h = h.copy()
    best_r = _ratio(best_h)
    
    # Local optimization
    # We can use scipy if available, but let's stick to numpy for robustness
    # Simple coordinate ascent
    for _ in range(100):
        improved = False
        for k in range(K):
            # Try to adjust w[k]
            # Constraint: sum(w) = 1 is not strictly required for the ratio, 
            # but scaling invariance means we can fix sum(w)=1.
            # However, local steps should respect non-negativity.
            
            # Perturb w[k] slightly
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                w_trial = w.copy()
                w_trial[k] += delta
                if w_trial[k] < 0:
                    continue
                # Renormalize to keep sum 1? 
                # Actually, ratio is scale invariant. 
                # But if we change one weight, the sum changes.
                # It's better to work in a parameterization that sums to 1.
                # Or just evaluate the ratio directly.
                h_trial = np.zeros(N)
                for j, idx in enumerate(indices):
                    h_trial[idx] = w_trial[j]
                
                r = _ratio(h_trial)
                if r > best_r:
                    best_r = r
                    best_h = h_trial.copy()
                    w = w_trial.copy()
                    improved = True
        if not improved:
            break
            
    return best_h, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Find best Sidon set
    # Greedy might not be optimal. Let's try a few random restarts of greedy or backtracking.
    best_sidon = []
    
    # Try greedy
    S_greedy = _find_max_sidon(N)
    if len(S_greedy) > len(best_sidon):
        best_sidon = S_greedy
        
    # Try random greedy
    np.random.seed(42)
    for _ in range(1000):
        if time.time() > deadline - 20:
            break
        # Randomly shuffle the order of addition?
        # Or just random start?
        # Greedy from smallest is usually good.
        # Let's try a different heuristic: Randomized Greedy
        S = []
        available = list(range(N))
        np.random.shuffle(available)
        # This is not standard. Standard greedy adds smallest valid.
        # Let's stick to the standard greedy which is fast and decent.
        # The max Sidon set in 50 is likely 7 or 8.
        pass

    # The standard greedy on 0..49:
    # 0, 1, 4, 10, 18, 23, 31, 44?
    # Let's check 0,1,4,10,18,23,31,44
    # Sums:
    # 0+0=0, 0+1=1, 0+4=4, 0+10=10, 0+18=18, 0+23=23, 0+31=31, 0+44=44
    # 1+1=2, 1+4=5, 1+10=11, 1+18=19, 1+23=24, 1+31=32, 1+44=45
    # 4+4=8, 4+10=14, 4+18=22, 4+23=27, 4+31=35, 4+44=48
    # 10+10=20, 10+18=28, 10+23=33, 10+31=41, 10+44=54
    # 18+18=36, 18+23=41 (Collides with 10+31=41!) -> Not Sidon.
    
    # So we need a valid set.
    # Let's just run the greedy function which checks validity.
    
    best_h = np.zeros(N)
    best_r = 0.0
    
    # Find the largest Sidon set found by greedy
    S_best = _find_max_sidon(N)
    
    # Optimize weights for this set
    h_opt, r_opt = _optimize_weights(S_best)
    if r_opt > best_r:
        best_r = r_opt
        best_h = h_opt
    
    # Try to find larger Sidon sets with a simple backtracking search if time permits
    # But greedy is usually close to optimal for small N.
    # Max Sidon size for N=50 is 7? 
    # According to OEIS A005282, max Sidon set in {1..n} is approx sqrt(n).
    # For n=50, sqrt(50)=7.07. So 7 is likely the max, maybe 8?
    # Let's check if 8 is possible.
    # If max is 7, score is 7.
    
    # Let's also try a "random" sparse set optimization as a fallback/alternative
    # Randomly pick K indices, optimize weights.
    for K in [7, 8, 9]:
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            indices = np.random.choice(N, K, replace=False)
            h_rand, r_rand = _optimize_weights(indices)
            if r_rand > best_r:
                best_r = r_rand
                best_h = h_rand
    
    return list(best_h)
