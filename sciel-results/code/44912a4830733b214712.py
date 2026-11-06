# EVOLVE-BLOCK-START
import time
import random

def _conflict_score(x):
    """
    Fast heuristic: count 2-element conflicts (a+b=2x) in [1,512].
    Lower is better.
    """
    count = 0
    # a ranges from max(1, 2x-512) to min(512, 2x-1)
    # a != x, b = 2x - a != x => a != x
    # a != b => a != x
    # So we just count valid a's.
    start = max(1, 2 * x - 512)
    end = min(512, 2 * x - 1)
    # Number of integers in [start, end] is end - start + 1
    # Subtract 1 if x is in that range (which it always is if x is in [1,512] and range is valid)
    # Actually, if a=x, b=x, which is not allowed (distinct elements).
    # So we subtract 1 for a=x.
    if start <= end:
        count = end - start + 1 - 1 # exclude a=x
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # 2s safety margin
    
    best_set = []
    best_len = 0
    
    # Precompute conflict scores
    conflict_scores = [0] * 513
    for x in range(1, 513):
        conflict_scores[x] = _conflict_score(x)
    
    # Sort candidates by conflict score (fewer conflicts first)
    candidates_sorted = sorted(range(1, 513), key=lambda x: conflict_scores[x])
    
    max_restarts = 1000 # Can do more restarts now
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate candidate order
        if restart == 0:
            candidates = list(candidates_sorted)
        elif restart == 1:
            candidates = list(reversed(candidates_sorted))
        else:
            # Random shuffle, biased towards low conflict
            candidates = list(range(1, 513))
            random.shuffle(candidates)
            # Simple sort with random tie-breaking might be enough, 
            # or just use random order but weighted?
            # Let's stick to simple random shuffle for speed, 
            # but sort by (conflict_score, random_noise)
            # Actually, just shuffling is often good enough for diversity.
            # Let's try sorting by conflict score but with random perturbation
            # to avoid getting stuck in the same local optimum as restart 0.
            
            # For efficiency, let's just use random shuffle for most restarts
            # The incremental DP is fast enough that we can afford many random tries.
            pass 

        current_set = []
        current_set_set = set()
        
        # Incremental DP
        # dp[k] is bitset of sums for subsets of size k from current_set
        # Max size 48, max sum 48*512 = 24576.
        # Initialize dp for empty set
        dp = [0] * 49
        dp[0] = 1
        
        n_current = 0
        
        for cand in candidates:
            if time.time() > deadline - 2.0:
                break
            
            if cand in current_set_set:
                continue
            
            # Check 1: cand is not mean of any subset of current_set
            # sum(T) = k * cand
            valid = True
            for k in range(2, n_current + 1):
                target = k * cand
                if (dp[k] >> target) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2: No a in current_set is mean of subset including cand
            # sum(T) + cand = (k+1) * a  => sum(T) = (k+1)a - cand
            # T subset of current_set, size k >= 1
            if valid:
                for a in current_set:
                    for k in range(1, n_current + 1):
                        target = (k + 1) * a - cand
                        if 0 <= target and (dp[k] >> target) & 1:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Add cand to current_set and update dp
                current_set.append(cand)
                current_set_set.add(cand)
                n_current += 1
                
                # Update dp in reverse
                # New dp[k] = old dp[k] | (old dp[k-1] << cand)
                # We need to be careful not to use the updated dp[k-1]
                # So we iterate k downwards
                for k in range(n_current, 0, -1):
                    dp[k] = dp[k] | (dp[k-1] << cand)
                
                if n_current >= 48:
                    break
        
        if len(current_set) > best_len:
            best_set = current_set
            best_len = len(current_set)
            
            if best_len >= 48:
                return best_set
    
    # Final polish: try to add more to best_set
    if best_set:
        best_set_set = set(best_set)
        n_best = len(best_set)
        
        # Rebuild dp for best_set
        dp = [0] * 49
        dp[0] = 1
        for x in best_set:
            for k in range(n_best, 0, -1):
                dp[k] = dp[k] | (dp[k-1] << x)
        
        # Try remaining numbers
        remaining = [x for x in range(1, 513) if x not in best_set_set]
        remaining.sort(key=lambda x: conflict_scores[x])
        
        for cand in remaining:
            if time.time() > deadline - 1.0:
                break
            
            # Check 1
            valid = True
            for k in range(2, n_best + 1):
                target = k * cand
                if (dp[k] >> target) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2
            if valid:
                for a in best_set:
                    for k in range(1, n_best + 1):
                        target = (k + 1) * a - cand
                        if 0 <= target and (dp[k] >> target) & 1:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                best_set.append(cand)
                n_best += 1
                for k in range(n_best, 0, -1):
                    dp[k] = dp[k] | (dp[k-1] << cand)
    
    return best_set

# EVOLVE-BLOCK-END
