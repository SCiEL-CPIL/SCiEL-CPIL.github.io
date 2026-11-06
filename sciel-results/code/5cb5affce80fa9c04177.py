import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Precompute all arithmetic progressions as index lists (0-based)
    # An AP: start a (0-based), step s >= 1, length k >= 1
    # indices: a, a+s, a+2s, ..., a+(k-1)s where a+(k-1)s < n
    progs = []
    for a in range(n):
        for s in range(1, n):
            idx = list(range(a, n, s))
            if len(idx) >= 1:
                progs.append(idx)
    
    Np = len(progs)
    
    # Build dense incidence matrix M (n x Np)
    # M[i, p] = 1 if position i is in progression p
    M = np.zeros((n, Np), dtype=np.float32)
    for pid, idx in enumerate(progs):
        for i in idx:
            M[i, pid] = 1.0
            
    # M.T is (Np, n). signs is (n,). sums = M.T @ signs -> (Np,)
    
    def compute_sums(signs):
        return M.T @ signs
        
    def max_abs_sum(sums):
        return np.max(np.abs(sums))
    
    # Initial random solution
    rng = np.random.RandomState(42)
    signs = rng.choice([-1, 1], size=n).astype(np.float32)
    sums = compute_sums(signs)
    best_signs = signs.copy()
    best_score = max_abs_sum(sums)
    
    # Phase 1: Identify Critical Progressions
    # We run a short, fast local search to find a decent solution,
    # then identify which progressions are "tight" (have large |sum|).
    # These will be our "critical" set for the main search.
    
    # Quick greedy local search to find a baseline
    current_signs = signs.copy()
    current_sums = sums.copy()
    current_score = best_score
    
    # Run 5000 fast iterations to stabilize
    for _ in range(5000):
        if time.time() > deadline - 70: # Keep lots of time for Phase 2
            break
        i = rng.randint(n)
        sign_i = current_signs[i]
        delta = -2.0 * sign_i * M[i, :]
        new_sums = current_sums + delta
        new_score = np.max(np.abs(new_sums))
        
        if new_score <= current_score:
            current_signs[i] = -sign_i
            current_sums = new_sums
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_signs = current_signs.copy()
    # Note: We use current_signs/current_sums as the "baseline" for critical selection,
    # even if best_signs is slightly better, because we want the critical set to be
    # relevant to the landscape we are exploring. Actually, let's use the best found so far
    # to be safe, or just the current state. Let's use current state for consistency.
    
    # Identify Critical Progressions
    # Threshold: top K progressions by absolute sum magnitude
    abs_sums = np.abs(current_sums)
    # We want a small set. Let's say top 500 or those > median.
    # Let's take top 10% of progressions by magnitude.
    k_crit = max(100, Np // 10)
    crit_indices = np.argsort(abs_sums)[-k_crit:]
    crit_mask = np.zeros(Np, dtype=bool)
    crit_mask[crit_indices] = True
    
    # Extract critical matrix M_crit (n x k_crit)
    # M_crit = M[:, crit_indices]
    # But M is (n, Np). So M[:, crit_indices] works.
    M_crit = M[:, crit_indices]
    
    # Phase 2: Targeted Local Search on Critical Set
    # We will perform Simulated Annealing or Greedy search using ONLY M_crit for evaluation.
    # This is ~10x faster.
    # We will accept a move if it improves the max_abs_sum on the CRITICAL set.
    # However, we must ensure we don't blow up the non-critical sums.
    # Strategy: 
    # 1. Do many fast steps using M_crit.
    # 2. Periodically (every 1000 steps) verify against FULL M.
    # 3. If full score improves, update best.
    # 4. If full score degrades significantly, revert or restart?
    #    Better: If a move improves critical score, accept it.
    #    Then check full score. If full score is worse than current full score by a lot, 
    #    maybe reject? 
    #    Actually, the "Critical Set" heuristic assumes that if we minimize the worst offenders,
    #    the others are likely fine or will become the new offenders.
    #    To be safe, we can use a "penalty" method or just verify frequently.
    
    # Let's use a hybrid: 
    # Evaluate new_score_crit = max(|new_sums_crit|)
    # If new_score_crit < current_score_crit:
    #    Accept.
    #    Update current_sums (full) and current_sums_crit.
    #    Check if full score improved global best.
    
    # Initialize for Phase 2
    cur_signs = current_signs.copy()
    cur_sums_full = current_sums.copy()
    cur_sums_crit = cur_sums_full[crit_indices]
    cur_score_crit = np.max(np.abs(cur_sums_crit))
    
    # We also track the full score to update global best
    cur_score_full = np.max(np.abs(cur_sums_full))
    
    # SA Parameters
    T_start = 5.0
    T_end = 0.01
    n_iters = 500000 # We can do many more because it's faster
    
    for it in range(n_iters):
        if time.time() > deadline - 2:
            break
            
        # Decay temperature
        T = T_start * (1.0 - it / float(n_iters)) + T_end
        
        i = rng.randint(n)
        sign_i = cur_signs[i]
        
        # Fast evaluation on critical set
        delta_crit = -2.0 * sign_i * M_crit[i, :]
        new_sums_crit = cur_sums_crit + delta_crit
        new_score_crit = np.max(np.abs(new_sums_crit))
        
        delta_score = new_score_crit - cur_score_crit
        
        # Acceptance criterion based on critical score
        if delta_score <= 0 or rng.rand() < np.exp(-delta_score / T):
            # Accept move
            cur_signs[i] = -sign_i
            
            # Update full sums (slower, but necessary for global tracking)
            # We can update full sums incrementally too
            delta_full = -2.0 * sign_i * M[i, :]
            cur_sums_full = cur_sums_full + delta_full
            
            # Update critical sums
            cur_sums_crit = cur_sums_crit + delta_crit
            cur_score_crit = new_score_crit
            
            # Check global best
            # To save time, we don't compute max(|cur_sums_full|) every step.
            # We can do it every 100 steps or when cur_score_crit improves significantly?
            # Let's do it every 10 steps. It's a max over 32k floats, which is fast.
            if it % 10 == 0:
                new_full_score = np.max(np.abs(cur_sums_full))
                if new_full_score < best_score:
                    best_score = new_full_score
                    best_signs = cur_signs.copy()
                    
    # Final check
    final_full_score = np.max(np.abs(cur_sums_full))
    if final_full_score < best_score:
        best_signs = cur_signs.copy()
        
    return [int(x) for x in best_signs]
