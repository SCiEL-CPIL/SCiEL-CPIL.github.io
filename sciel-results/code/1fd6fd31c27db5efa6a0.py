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
    
    # For efficient vectorized local search, precompute a sparse incidence structure
    # We need to update sums quickly.
    # Instead of iterating over progs for each position, let's keep the sums array
    # and update it based on the position flipped.
    
    # pos_to_progs[i] = list of prog_ids containing i
    pos_to_progs = [[] for _ in range(n)]
    for pid, idx in enumerate(progs):
        for i in idx:
            pos_to_progs[i].append(pid)
    
    # To make vectorization easier for "try all flips", we can precompute
    # for each position i, the list of prog indices.
    # However, the lengths of these lists vary.
    # A fully vectorized "try all 256 flips" in one go is hard with variable length lists
    # unless we use a dense matrix (256 x Np) which is 256 * ~20000 = 5M entries.
    # 5M * 8 bytes (float) = 40MB. This is feasible.
    
    # Let's build a dense incidence matrix M where M[i, p] = 1 if i in prog p, else 0.
    # Then sums = signs @ M.T (if signs is 1D and M is 2D)
    # Actually, signs is vector of length n. sums is vector of length Np.
    # sums[p] = sum_{i in p} signs[i] = dot(signs, M[:, p])
    # So sums = M.T @ signs
    
    M = np.zeros((n, Np), dtype=np.float32)
    for pid, idx in enumerate(progs):
        for i in idx:
            M[i, pid] = 1.0
            
    # M.T is (Np, n). signs is (n,). sums = M.T @ signs -> (Np,)
    
    def compute_sums(signs):
        return M.T @ signs
        
    def max_abs_sum(sums):
        return np.max(np.abs(sums))
    
    # Soft max surrogate for smooth optimization
    def soft_max_abs(sums, beta=10.0):
        # Use log-sum-exp trick for numerical stability
        abs_sums = np.abs(sums)
        max_val = np.max(abs_sums)
        if max_val == 0:
            return 0.0
        exp_sums = np.exp(beta * (abs_sums - max_val))
        return max_val + np.log(np.sum(exp_sums)) / beta
    
    # Initialize with random signs
    rng = np.random.RandomState(42)
    signs = rng.choice([-1, 1], size=n).astype(np.float32)
    sums = compute_sums(signs)
    
    best_signs = signs.copy()
    best_score = max_abs_sum(sums)
    
    # Phase 1: Smooth surrogate optimization with gradient-like steps
    # We'll use a smooth objective to guide the search, then polish with exact max
    
    T_start = 5.0
    T_end = 0.01
    beta = 10.0  # Sharpness of soft max
    
    current_signs = signs.copy()
    current_sums = sums.copy()
    current_score = best_score
    
    # Phase 1: SA with smooth surrogate (faster convergence)
    n_iters_phase1 = 50000
    
    for it in range(n_iters_phase1):
        if time.time() > deadline - 30:
            break
            
        T = T_start * (1.0 - it / float(n_iters_phase1)) + T_end
        
        # Pick random position
        i = rng.randint(n)
        
        sign_i = current_signs[i]
        delta = -2.0 * sign_i * M[i, :]
        new_sums = current_sums + delta
        
        # Use soft max for acceptance decision (smoother landscape)
        current_soft = soft_max_abs(current_sums, beta)
        new_soft = soft_max_abs(new_sums, beta)
        
        delta_soft = new_soft - current_soft
        
        if delta_soft <= 0 or rng.rand() < np.exp(-delta_soft / T):
            # Accept
            current_signs[i] = -sign_i
            current_sums = new_sums
            current_score = max_abs_sum(new_sums)
            
            if current_score < best_score:
                best_score = current_score
                best_signs = current_signs.copy()
    
    # Phase 2: Exact max-based polish (greedy local search)
    # Try all single flips, accept any improvement
    current_signs = best_signs.copy()
    current_sums = compute_sums(current_signs)
    current_score = best_score
    
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(n):
            sign_i = current_signs[i]
            delta = -2.0 * sign_i * M[i, :]
            new_sums = current_sums + delta
            new_score = max_abs_sum(new_sums)
            
            if new_score < current_score:
                current_signs[i] = -sign_i
                current_sums = new_sums
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_signs = current_signs.copy()
    
    # Phase 3: More SA iterations with exact max for fine-tuning
    current_signs = best_signs.copy()
    current_sums = compute_sums(current_signs)
    current_score = best_score
    
    T_start2 = 2.0
    T_end2 = 0.001
    n_iters_phase3 = 30000
    
    for it in range(n_iters_phase3):
        if time.time() > deadline - 2:
            break
            
        T = T_start2 * (1.0 - it / float(n_iters_phase3)) + T_end2
        
        i = rng.randint(n)
        
        sign_i = current_signs[i]
        delta = -2.0 * sign_i * M[i, :]
        new_sums = current_sums + delta
        new_score = max_abs_sum(new_sums)
        
        delta_score = new_score - current_score
        
        if delta_score <= 0 or rng.rand() < np.exp(-delta_score / T):
            current_signs[i] = -sign_i
            current_sums = new_sums
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_signs = current_signs.copy()
    
    # Final polish: greedy local search again
    current_signs = best_signs.copy()
    current_sums = compute_sums(current_signs)
    current_score = best_score
    
    improved = True
    while improved:
        if time.time() > deadline - 1:
            break
        
        improved = False
        for i in range(n):
            sign_i = current_signs[i]
            delta = -2.0 * sign_i * M[i, :]
            new_sums = current_sums + delta
            new_score = max_abs_sum(new_sums)
            
            if new_score < current_score:
                current_signs[i] = -sign_i
                current_sums = new_sums
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_signs = current_signs.copy()
    
    return [int(x) for x in best_signs]
