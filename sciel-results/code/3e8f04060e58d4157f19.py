"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

_AP_MAT_CACHE = None

def _build_ap_matrix(n=256):
    """
    Build a dense binary incidence matrix where each row corresponds to an
    arithmetic progression within [1, n] (1-indexed) and each column to a position.
    Entry (i, j) is 1 if position j+1 is in AP i.
    """
    aps = []
    for start in range(1, n + 1):
        for step in range(1, n - start + 1):
            # Generate the AP: start, start+step, ..., <= n
            # Using range is efficient
            ap_list = list(range(start, n + 1, step))
            if ap_list: # Always true for step >= 1 and start <= n
                aps.append(ap_list)
    
    m = len(aps)
    mat = np.zeros((m, n), dtype=np.int8)
    for i, ap in enumerate(aps):
        # Vectorized assignment for each AP is faster than loop if AP is long,
        # but APs are short on average. 
        # However, mat[i, np.array(ap) - 1] = 1 is clean.
        mat[i, np.array(ap, dtype=np.int32) - 1] = 1
    return mat

def _evaluate(signs, ap_mat):
    """
    Calculate the maximum absolute sum over all arithmetic progressions.
    signs: vector of length n with values -1 or 1.
    ap_mat: matrix of shape (m, n).
    """
    # signs must be float or int for matmul to work correctly with int8 mat
    # mat is int8, signs should be float64 or int32 for precision/speed
    sums = ap_mat @ signs
    return np.max(np.abs(sums))

def _sa_run(ap_mat, n, rng, time_limit, T_start, T_end, alpha):
    """
    Run a single instance of Simulated Annealing.
    Returns the best signs and cost found in this run.
    """
    # Initial solution: random signs
    current_signs = rng.choice([-1, 1], size=n).astype(np.float64)
    current_cost = _evaluate(current_signs, ap_mat)
    
    best_signs = current_signs.copy()
    best_cost = current_cost
    
    T = T_start
    start_time = time.time()
    
    # Pre-allocate delta for speed? No, we recalc full cost.
    # To speed up, we can track the sums array.
    # sums = ap_mat @ current_signs
    # If we flip index k, sums[i] changes by 2 * ap_mat[i, k] * (-1) if sign was +1?
    # Let's stick to full re-evaluation first as it's < 1ms for 32k x 256.
    # 32768 * 256 = 8.4M multiplications. NumPy does this in ~0.5-1ms.
    # 1000 iterations/sec is very low. We can do ~1000-2000 iterations/sec.
    # In 50s, we can do ~100k iterations.
    
    while (time.time() - start_time) < time_limit and T > T_end:
        # Pick a random index to flip
        idx = rng.integers(0, n)
        
        old_sign = current_signs[idx]
        current_signs[idx] = -old_sign
        
        new_cost = _evaluate(current_signs, ap_mat)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = current_signs.copy()
        else:
            delta = new_cost - current_cost
            # Accept with probability exp(-delta / T)
            # If delta is 0, exp(0)=1, so always accept.
            # If delta > 0, accept with prob < 1.
            if rng.random() < np.exp(-delta / T):
                # Accept
                pass
            else:
                # Reject: revert
                current_signs[idx] = old_sign
        
        T *= alpha
        
    return best_signs, best_cost

def construct():
    global _AP_MAT_CACHE
    if _AP_MAT_CACHE is None:
        _AP_MAT_CACHE = _build_ap_matrix(256)
    
    ap_mat = _AP_MAT_CACHE
    n = 256
    
    # Multi-restart SA
    # Total time budget
    total_start = time.time()
    max_total_time = 55.0 # Leave 5s buffer for overhead
    
    # Parameters for multi-restart
    # We want multiple diverse runs.
    # If we run 10 restarts, each gets ~5s.
    # Or we can run until time runs out, restarting when T gets too low.
    
    rng = np.random.default_rng(42)
    
    best_global_signs = None
    best_global_cost = np.inf
    
    # Strategy: Run SA with a fixed time per restart, or adaptively.
    # Let's do 5 restarts of 10s each? Or just loop while time remains.
    # A good strategy is to start with high T, cool down, and if we haven't improved in a while, restart.
    # Simpler: Fixed number of restarts.
    
    num_restarts = 5
    time_per_restart = (max_total_time - (time.time() - total_start)) / num_restarts
    
    for r in range(num_restarts):
        # Adjust time if we are running out
        elapsed = time.time() - total_start
        remaining = max_total_time - elapsed
        if remaining <= 1.0:
            break
            
        # Use slightly different params for each restart? 
        # Or just random initialization is enough for diversity.
        
        # T_start and T_end need to be tuned.
        # Cost range is roughly [-256, 256], so max diff is ~512.
        # T_start = 50.0 allows exploration.
        # T_end = 1.0 allows fine tuning.
        # Alpha should be such that T reaches T_end in the allotted time.
        # If we do ~10000 iterations in 10s (1000 iter/s), 
        # (T_end/T_start) = alpha^N => alpha = (T_end/T_start)^(1/N)
        # Let's estimate iterations.
        # 10s / 1ms per iter = 10000 iters.
        # alpha = (1/50)^(1/10000) ~ 0.9993
        
        # Let's just use a fixed alpha and let it run until time limit.
        # If T doesn't reach T_end, that's fine, we just stop.
        
        T_start = 20.0
        T_end = 0.5
        alpha = 0.999 # Cool slowly
        
        best_signs, best_cost = _sa_run(ap_mat, n, rng, time_per_restart, T_start, T_end, alpha)
        
        if best_cost < best_global_cost:
            best_global_cost = best_cost
            best_global_signs = best_signs
            
        # Check time
        if time.time() - total_start > max_total_time - 1.0:
            break
            
    if best_global_signs is None:
        # Fallback
        best_global_signs = np.ones(n)
        
    return [int(x) for x in best_global_signs]

# EVOLVE-BLOCK-END
