import time
import itertools
import math

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    best_set = [0, 1, 3]
    best_score = -1.0
    
    def score_set(U):
        if not U or 0 not in U:
            return 0.0
        U = sorted(set(U))
        max_u = U[-1]
        if max_u == 0:
            return 1.0
            
        # Calculate |U+U|
        sums = set()
        diffs = set()
        n = len(U)
        for i in range(n):
            for j in range(i, n):
                sums.add(U[i] + U[j])
                diffs.add(U[i] - U[j])
                if i != j:
                    diffs.add(U[j] - U[i])
        
        size_sum = len(sums)
        size_diff = len(diffs)
        
        if size_sum == 0:
            return 0.0
            
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            return 1.0
            
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        val = 1.0 + math.log(ratio) / denom
        return val

    # Update best with initial guess
    best_score = score_set(best_set)
    
    # Search space
    # We search for sets of size k containing 0, with max element M.
    # We iterate M from 1 to M_MAX.
    # For each M, we iterate k from 1 to K_MAX.
    
    M_MAX = 60
    K_MAX = 8
    
    for M in range(1, M_MAX + 1):
        if time.time() > deadline:
            break
            
        # Max possible score for this M is bounded by 1 + log(2M+1)/log(2M+1) = 2
        # But practically, we can prune if best_score is very high.
        
        for k in range(1, K_MAX + 1):
            if time.time() > deadline:
                break
                
            if k == 1:
                U = [0]
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U
                continue
            
            # Choose k-1 elements from 1..M
            # Prune if combination count is too high
            # Approximate check
            if k - 1 > M:
                continue
                
            # Estimate cost
            # We just run it, but break inner loop if time exceeded
            
            for combo in itertools.combinations(range(1, M + 1), k - 1):
                if time.time() > deadline:
                    break
                    
                U = [0] + list(combo)
                # Optimization: if max(U) is same as M, and we already checked similar?
                # No, just compute.
                
                # Quick prune: 
                # Max possible ratio for this U is (2M+1) / (2k-1)
                # Max possible score is 1 + log((2M+1)/(2k-1)) / log(2M+1)
                # If this is less than best_score, skip?
                # This upper bound is loose, but might help.
                
                max_possible_ratio = (2 * M + 1) / (2 * k - 1)
                if max_possible_ratio <= 1.0:
                    continue
                    
                max_possible_score = 1.0 + math.log(max_possible_ratio) / math.log(2 * M + 1)
                if max_possible_score <= best_score:
                    continue
                
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U

    return best_set
