import numpy as np
import time

N = 50
DEADLINE = 145.0  # 150s limit, 5s margin

def compute_ratio(h):
    """
    Compute the ratio: ||h||_2^2 / (||h||_1 * ||h * h||_inf)
    where * denotes linear convolution.
    """
    h = np.asarray(h)
    if np.any(h < -1e-9):
        return -1.0
    h = np.clip(h, 0, None)
    
    l1 = np.sum(h)
    if l1 < 1e-12:
        return -1.0
        
    l2_sq = np.sum(h**2)
    
    # Linear convolution
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(np.abs(h_conv))
    
    if max_conv < 1e-12:
        return -1.0
        
    return (l2_sq / (l1 * max_conv))

def is_sidon(arr):
    """Check if a set of integers is a Sidon set (B2 set)."""
    arr = sorted(set(arr))
    n = len(arr)
    sums = {}
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in sums:
                return False
            sums[s] = (i, j)
    return True

def find_sidon_set(n, max_val=200):
    """Find a Sidon set of size n using a greedy algorithm."""
    sidon = []
    for i in range(max_val):
        candidate = sidon + [i]
        if is_sidon(candidate):
            sidon.append(i)
            if len(sidon) == n:
                break
    return sidon

def optimize_sparse_weights(indices, time_limit=5.0):
    """
    Optimize weights on the given indices to maximize the ratio.
    We fix sum(weights) = 1 to handle scale invariance.
    """
    if not indices:
        return np.zeros(N)
        
    K = len(indices)
    best_val = -1.0
    best_w = np.ones(K) / K
    
    start_time = time.time()
    n_restarts = 10
    
    for i in range(n_restarts):
        if time.time() > start_time + time_limit:
            break
        if time.time() > DEADLINE - 2:
            break
            
        if i == 0:
            w0 = np.ones(K) / K
        else:
            w0 = np.random.dirichlet(np.ones(K))
            
        # Simple coordinate descent or local search
        # Since K is small (Sidon sets are small), we can do full gradient or just coordinate ascent.
        # Let's do coordinate ascent with a few iterations.
        w = w0.copy()
        for _ in range(20):
            if time.time() > DEADLINE - 2:
                break
            improved = False
            for j in range(K):
                old_w = w.copy()
                # Try adjusting w[j]
                # We need to keep sum(w)=1.
                # If we change w[j], we must adjust others or just treat it as a line search on w[j] 
                # while scaling the rest? No, simpler: just try random perturbations or grid.
                # Actually, let's just use scipy minimize on the log-weights for the sparse set.
                pass
            
            # Since coordinate ascent is tricky with the constraint, let's just rely on the global 
            # L-BFGS-B in construct() which handles the full vector, but we use this to 
            # initialize or refine specific sparse structures.
            # For now, this function is a placeholder for potential future refinement.
            # The main optimization will happen in construct().
            
        # Evaluate current w
        h = np.zeros(N)
        h[indices] = w
        val = compute_ratio(h)
        if val > best_val:
            best_val = val
            best_w = w
            
    h_final = np.zeros(N)
    h_final[indices] = best_w
    return h_final

def construct():
    start_time = time.time()
    
    # 1. Generate candidate Sidon sets
    # The theoretical max for N=50 is around sqrt(50) ~ 7.
    # Greedy Sidon sets in [0, 49] usually find size 6 or 7.
    # Let's find the largest possible Sidon set we can quickly.
    
    best_h = np.ones(N) / N
    best_val = compute_ratio(best_h)
    
    # Try to find a large Sidon set
    # Greedy approach
    sidon_indices = []
    for i in range(N):
        candidate = sidon_indices + [i]
        if is_sidon(candidate):
            sidon_indices.append(i)
            
    # If we found a decent Sidon set, use it as a starting point
    if len(sidon_indices) >= 4:
        h_sidon = np.zeros(N)
        h_sidon[sidon_indices] = 1.0 / len(sidon_indices)
        val_sidon = compute_ratio(h_sidon)
        if val_sidon > best_val:
            best_val = val_sidon
            best_h = h_sidon

    # Also try a "power of 2" style set which is a Sidon set
    # 1, 2, 4, 8, 16, 32 (size 6)
    # Indices: 0, 1, 2, 4, 8, 16, 32? No, 0+32=32, 16+16=32. Collision.
    # Standard Sidon: 1, 2, 4, 8, 16, 32 is not Sidon because 16+16=32 and 0+32=32 if 0 is included.
    # Without 0: 1, 2, 4, 8, 16, 32. Sums: 1+1=2 (collision with 2? No, 2 is an element, not a sum of distinct elements? 
    # Definition: sums of pairs (i,j) with i<=j. 
    # 1+1=2. Is 2 in the set? Yes. But the condition is about the sums being unique.
    # Sums: 1+1=2, 1+2=3, 1+4=5, 1+8=9, 1+16=17, 1+32=33
    # 2+2=4, 2+4=6, ...
    # 4+4=8. Collision! 1+7? No. 4+4=8. Is there another pair summing to 8? 1+7 (7 not in set). 
    # Wait, 8 is in the set. 4+4=8. 1+? No.
    # Let's just use the greedy one found above.
    
    # Optimization Strategy:
    # The objective R(h) = ||h||_2^2 / (||h||_1 ||h*h||_inf) is scale invariant.
    # We maximize R(h).
    # We use L-BFGS-B on the log-weights to ensure positivity.
    
    def neg_log_ratio(params):
        h = np.exp(params)
        # Normalize to L1 = 1 to stabilize optimization (scale invariance)
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 1000.0
        h_norm = h / l1
        r = compute_ratio(h_norm)
        if r <= 0:
            return 1000.0
        return -np.log(r)
        
    # Random restarts with different initializations
    n_restarts = 40
    
    for i in range(n_restarts):
        if time.time() > DEADLINE - 2:
            break
            
        if i == 0:
            # Start from best found so far
            h_start = best_h.copy()
            if np.any(h_start <= 0):
                h_start[h_start <= 0] = 1e-6
            x0 = np.log(h_start)
        elif i == 1 and len(sidon_indices) >= 4:
            # Start from Sidon set
            h_start = np.zeros(N)
            h_start[sidon_indices] = 1.0 / len(sidon_indices)
            h_start[h_start <= 0] = 1e-6
            x0 = np.log(h_start)
        else:
            # Random sparse initialization
            K = np.random.randint(3, 12)
            h_start = np.zeros(N)
            indices = np.random.choice(N, K, replace=False)
            h_start[indices] = 1.0 / K
            h_start[h_start <= 0] = 1e-6
            x0 = np.log(h_start)
            
        try:
            from scipy.optimize import minimize
            res = minimize(neg_log_ratio, x0, method='L-BFGS-B', 
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success or res.fun < 1000:
                h_opt = np.exp(res.x)
                h_opt = h_opt / np.sum(h_opt)
                val_opt = compute_ratio(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt
        except Exception:
            pass

        # Local Search: Coordinate Ascent
        # After each global step, do a few steps of coordinate ascent to polish.
        if time.time() > DEADLINE - 2:
            continue
            
        h_cur = best_h.copy()
        for _ in range(10):
            if time.time() > DEADLINE - 2:
                break
            improved = False
            for j in range(N):
                old_val = h_cur[j]
                # Try a few perturbations for h[j]
                # Keep sum(h) = 1? No, ratio is scale invariant, so we can just adjust h[j] 
                # and renormalize at the end, or just adjust h[j] and let the ratio handle it.
                # Actually, if we change h[j], the L1 and L2 and Conv all change.
                # Let's try a simple line search on h[j].
                # We want to maximize R(h).
                # Let's try scaling h[j] by factors.
                for factor in [0.5, 0.8, 1.2, 1.5, 2.0, 0.0]:
                    h_new = h_cur.copy()
                    h_new[j] = old_val * factor
                    if np.any(h_new < 0):
                        continue
                    val_new = compute_ratio(h_new)
                    if val_new > compute_ratio(h_cur) + 1e-12:
                        h_cur = h_new
                        improved = True
                        break
                
            if not improved:
                break
                
        val_cur = compute_ratio(h_cur)
        if val_cur > best_val:
            best_val = val_cur
            best_h = h_cur

    # Final check
    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) == 0:
        best_h = np.ones(N) / N
    else:
        # Normalize to sum 1? The evaluator likely doesn't care about scale, 
        # but let's keep it normalized to be safe.
        best_h = best_h / np.sum(best_h)
        
    return list(best_h)
