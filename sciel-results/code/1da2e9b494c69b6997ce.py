import numpy as np
import random

def construct():
    n = 128
    targets = list(range(128, 257))  # 129 targets
    target_set = set(targets)
    all_candidates = list(range(0, 129))
    
    best_A = None
    best_peak = float('inf')
    
    # Run multiple greedy attempts with different random seeds to escape local optima
    for seed in range(50):
        random.seed(seed)
        
        A = set()
        # representation count for each target
        rep_count = [0] * 257  # Index 0 to 256
        
        while True:
            # Check if all targets covered
            if all(rep_count[t] > 0 for t in targets):
                break
            
            best_x = None
            best_score = None
            
            # Iterate through candidates
            # Shuffle to break ties differently
            candidates = all_candidates[:]
            random.shuffle(candidates)
            
            for x in candidates:
                if x in A:
                    continue
                
                # Simulate adding x
                new_rep = rep_count[:]
                
                # Update counts for pairs involving x
                for a in A:
                    s = x + a
                    new_rep[s] += 1
                s2 = x + x
                new_rep[s2] += 1
                
                # Calculate score
                max_count = 0
                sum_count = 0
                uncovered_count = 0
                for t in targets:
                    c = new_rep[t]
                    if c == 0:
                        uncovered_count += 1
                    else:
                        if c > max_count:
                            max_count = c
                        sum_count += c
                
                if uncovered_count > 0:
                    score = (10000 + uncovered_count, sum_count)
                else:
                    score = (max_count, sum_count)
                
                if best_score is None or score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
            
            A.add(best_x)
            # Update rep_count
            for a in A:
                s = best_x + a
                rep_count[s] += 1
            s2 = best_x + best_x
            rep_count[s2] += 1
        
        # After the loop, check if this A is valid and better
        if all(rep_count[t] > 0 for t in targets):
            peak = max(rep_count[t] for t in targets)
            if peak < best_peak:
                best_peak = peak
                best_A = A[:]
            
            # Early stopping if we reach a very good peak
            if best_peak <= 3:
                break

    if best_A is None:
        # Fallback
        return list(all_candidates[:20])
        
    return list(best_A)
