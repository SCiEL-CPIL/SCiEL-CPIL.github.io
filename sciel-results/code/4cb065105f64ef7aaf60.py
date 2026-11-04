"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
TARGET_SUM = N / 2.0

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_overlap(h):
    """
    Compute the maximum linear overlap C(k) = sum h[i] * (1 - h[i+k]) for k != 0.
    """
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    # c has length 2N-1.
    # c[N-1] is lag 0.
    # We want max of c[:N-1] and c[N:]
    c_nonzero = np.concatenate([c[:N-1], c[N:]])
    return float(np.max(c_nonzero))

def _project_to_feasible(h, target=TARGET_SUM, tol=1e-12):
    """
    Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = target.
    Uses bisection on a Lagrange multiplier for exact projection.
    """
    h = np.clip(h, 0.0, 1.0).copy()
    current = np.sum(h)
    
    if abs(current - target) < tol:
        return h
    
    # Bisection on lambda: find lambda such that sum(clip(h - lambda, 0, 1)) = target
    # If current > target, we need to reduce sum, so lambda should be positive (subtract)
    # If current < target, we need to increase sum, so lambda should be negative (add)
    
    # Range for lambda:
    # Min possible sum is 0 (if lambda is large positive)
    # Max possible sum is N (if lambda is large negative)
    # We can bound lambda roughly.
    lo = -1.0 - max(h)  # Ensures clip(h-lo, 0, 1) is close to 1 for all h
    hi = 1.0 + max(h)   # Ensures clip(h-hi, 0, 1) is close to 0 for all h
    
    # If current > target, we are subtracting, so lambda > 0
    # If current < target, we are adding, so lambda < 0
    if current > target:
        lo = 0.0
        hi = 1.0 + max(h)
    else:
        lo = -1.0 - max(h)
        hi = 0.0
        
    for _ in range(100):
        mid = (lo + hi) / 2.0
        h_test = np.clip(h - mid, 0.0, 1.0)
        sum_test = np.sum(h_test)
        
        if sum_test > target:
            lo = mid
        else:
            hi = mid
            
    mid = (lo + hi) / 2.0
    h_proj = np.clip(h - mid, 0.0, 1.0)
    
    # Final fine-tuning to ensure exact sum if possible
    # Since the projection might not hit the target exactly due to discrete nature of clipping,
    # we do a small adjustment on the non-boundary elements.
    diff = target - np.sum(h_proj)
    if abs(diff) > 1e-12:
        # Find indices not at 0 or 1
        mask = (h_proj > 0.0) & (h_proj < 1.0)
        if np.any(mask):
            # Distribute the remaining diff
            step = diff / np.sum(mask)
            h_proj[mask] += step
            h_proj = np.clip(h_proj, 0.0, 1.0)
            # If still off, try again
            diff2 = target - np.sum(h_proj)
            if abs(diff2) > 1e-12:
                mask2 = (h_proj > 0.0) & (h_proj < 1.0)
                if np.any(mask2):
                    step2 = diff2 / np.sum(mask2)
                    h_proj[mask2] += step2
                    h_proj = np.clip(h_proj, 0.0, 1.0)
                    
    return h_proj

def _local_search_binary(h, max_iters=50):
    """
    Greedy local search on binary sequences (or near-binary).
    Tries all swaps of 1s and 0s.
    """
    current_score = _max_overlap(h)
    
    for _ in range(max_iters):
        best_swap = None
        best_score = current_score
        best_h = h
        
        # Identify ones and zeros
        ones = np.where(h > 0.5)[0]
        zeros = np.where(h < 0.5)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        # Try all swaps
        for i in ones:
            for j in zeros:
                h_new = h.copy()
                h_new[i] = 0.0
                h_new[j] = 1.0
                score = _max_overlap(h_new)
                if score < best_score - 1e-12:
                    best_score = score
                    best_swap = (i, j)
                    best_h = h_new
        
        if best_swap is None:
            break
            
        h = best_h
        current_score = best_score
        
    return h, current_score

def _simulated_annealing_binary(h, start_temp=5.0, end_temp=0.1, 
                                max_iters=5000, deadline=None, rng=None):
    """
    Simulated annealing on binary sequences.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current = h.copy()
    current_score = _max_overlap(current)
    best = current.copy()
    best_score = current_score
    
    n_iter = 0
    while n_iter < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        
        # Temperature schedule
        progress = n_iter / max_iters
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Random swap
        ones = np.where(current > 0.5)[0]
        zeros = np.where(current < 0.5)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        i = ones[rng.integers(0, len(ones))]
        j = zeros[rng.integers(0, len(zeros))]
        
        current[i] = 0.0
        current[j] = 1.0
        
        new_score = _max_overlap(current)
        
        delta = new_score - current_score
        
        # Accept if better or with probability exp(-delta/temp)
        if delta < 0 or (temp > 0 and np.log(rng.random()) < -delta / temp):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = current.copy()
        else:
            # Reject: swap back
            current[i] = 1.0
            current[j] = 0.0
            
        n_iter += 1
    
    return best, best_score

def _continuous_refinement(h, max_iters=1000, deadline=None, rng=None):
    """
    Refine a binary solution in continuous space using small perturbations.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    current = h.copy()
    current_score = _max_overlap(current)
    best = current.copy()
    best_score = current_score
    
    n_iter = 0
    while n_iter < max_iters:
        if deadline is not None and time.time() > deadline:
            break
            
        # Random perturbation
        h_pert = current + rng.normal(0, 0.1, N)
        h_pert = _project_to_feasible(h_pert)
        
        new_score = _max_overlap(h_pert)
        
        if new_score < current_score:
            current = h_pert
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = current.copy()
                
        n_iter += 1
        
    return best, best_score

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Uses a hybrid approach:
    1. Simulated Annealing on binary sequences to find a good combinatorial structure.
    2. Local search on binary sequences to refine.
    3. Continuous refinement to fine-tune values.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    best_h_global = None
    best_score_global = float('inf')
    
    # Multiple restarts
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
            
        seed = 42 + restart
        rng_local = np.random.default_rng(seed)
        
        # Initialize with random binary sequence
        h_curr = np.zeros(N)
        indices = rng_local.choice(N, size=N//2, replace=False)
        h_curr[indices] = 1.0
        
        score_curr = _max_overlap(h_curr)
        
        if score_curr < best_score_global:
            best_score_global = score_curr
            best_h_global = h_curr.copy()
        
        # Simulated Annealing on binary sequences
        sa_deadline = min(deadline - 10.0, start_time + 5.0 + 1.0 * restart)
        h_sa, sa_score = _simulated_annealing_binary(
            h_curr, 
            start_temp=5.0, 
            end_temp=0.1, 
            max_iters=2000, 
            deadline=sa_deadline, 
            rng=rng_local
        )
        
        if sa_score < best_score_global:
            best_score_global = sa_score
            best_h_global = h_sa.copy()
            
        # Local search on binary sequences
        if time.time() < deadline - 10.0:
            h_ls, ls_score = _local_search_binary(h_sa, max_iters=20)
            
            if ls_score < best_score_global:
                best_score_global = ls_score
                best_h_global = h_ls.copy()
                
    # Continuous refinement on the best binary solution
    if time.time() < deadline - 5.0 and best_h_global is not None:
        h_cont, cont_score = _continuous_refinement(
            best_h_global, 
            max_iters=2000, 
            deadline=deadline - 2.0, 
            rng=rng
        )
        
        if cont_score < best_score_global:
            best_score_global = cont_score
            best_h_global = h_cont.copy()
            
    # Ensure exact sum
    best_h_global = _project_to_feasible(best_h_global)
    
    # Final validity check
    assert np.all(best_h_global >= 0) and np.all(best_h_global <= 1), "Values out of [0,1]"
    assert abs(np.sum(best_h_global) - TARGET_SUM) < 1e-10, f"Sum {np.sum(best_h_global)} != {TARGET_SUM}"
    
    return best_h_global

# EVOLVE-BLOCK-END
