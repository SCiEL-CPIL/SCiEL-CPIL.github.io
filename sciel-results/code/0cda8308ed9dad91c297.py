import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
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
    
    # For efficient local search, precompute for each position i, which APs contain it
    pos_to_progs = [[] for _ in range(n)]
    for pid, idx in enumerate(progs):
        for i in idx:
            pos_to_progs[i].append(pid)
    
    def compute_sums(signs):
        sums = np.zeros(Np, dtype=np.int32)
        for pid, idx in enumerate(progs):
            s = 0
            for i in idx:
                s += signs[i]
            sums[pid] = s
        return sums
    
    def max_abs_sum(sums):
        return int(np.max(np.abs(sums)))
    
    def soft_max_abs(sums, beta=1.0):
        """Smooth surrogate for max |sum| using log-sum-exp."""
        abs_sums = np.abs(sums).astype(np.float64)
        m = np.max(abs_sums)
        if m == 0:
            return 0.0
        # log(sum(exp(beta*x))) / beta approximates max(x)
        shifted = abs_sums - m
        log_sum = np.log(np.sum(np.exp(beta * shifted))) + beta * m
        return log_sum / beta
    
    # Initialize with random signs
    rng = np.random.RandomState(42)
    signs = rng.choice([-1, 1], size=n)
    sums = compute_sums(signs)
    best_signs = signs.copy()
    best_score = max_abs_sum(sums)
    
    current_score = best_score
    
    # Local search with multiple restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            pass  # already initialized
        else:
            signs = rng.choice([-1, 1], size=n)
            sums = compute_sums(signs)
            current_score = max_abs_sum(sums)
        
        if current_score < best_score:
            best_signs = signs.copy()
            best_score = current_score
        
        # Local search: iterate through positions, flip if it reduces max abs sum
        # Use both hard max and soft surrogate to escape local optima
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                
                # Try flipping position i
                old_sign = signs[i]
                new_sign = -old_sign
                signs[i] = new_sign
                
                # Update sums for all APs containing i
                for pid in pos_to_progs[i]:
                    sums[pid] += 2 * new_sign
                
                new_score = max_abs_sum(sums)
                
                # Accept if hard score improves, or if soft score improves significantly
                # This helps escape local optima where hard score is flat
                soft_before = soft_max_abs(sums, beta=2.0)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_signs = signs.copy()
                elif new_score == current_score:
                    # Tie-breaking: accept if soft surrogate improves
                    # This allows moves that keep max same but reduce variance
                    pass  # Keep it simple for now
                else:
                    # Revert
                    signs[i] = old_sign
                    for pid in pos_to_progs[i]:
                        sums[pid] -= 2 * new_sign
    
    # Final polish: try all single flips multiple times with tighter criterion
    if time.time() < deadline - 2:
        for polish_iter in range(3):
            if time.time() > deadline - 2:
                break
            for i in range(n):
                if time.time() > deadline - 2:
                    break
                old_sign = signs[i]
                signs[i] = -old_sign
                for pid in pos_to_progs[i]:
                    sums[pid] += 2 * signs[i]
                sc = max_abs_sum(sums)
                if sc < best_score:
                    best_score = sc
                    best_signs = signs.copy()
                else:
                    signs[i] = old_sign
                    for pid in pos_to_progs[i]:
                        sums[pid] -= 2 * signs[i]
    
    return [int(x) for x in best_signs]
