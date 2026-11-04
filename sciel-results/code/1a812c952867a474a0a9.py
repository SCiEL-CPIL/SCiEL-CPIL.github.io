import numpy as np
import random
import time

N = 50

def construct():
    # We search for a subset of indices S that maximizes |S| / N_max(S)
    # where N_max(S) is the maximum number of pairs (i, j) in S x S with i + j = constant.
    # This corresponds to a Sidon set approach for maximizing the autocorrelation inequality ratio.
    
    def calc_score(indices):
        if not indices:
            return 0.0
        k = len(indices)
        # Count sums
        sum_counts = {}
        for i in indices:
            for j in indices:
                s = i + j
                sum_counts[s] = sum_counts.get(s, 0) + 1
        
        n_max = max(sum_counts.values()) if sum_counts else 1
        return k / n_max

    best_score = 0.0
    best_indices = [0] # Fallback
    
    start_time = time.time()
    time_limit = 110.0
    
    # Try different sizes k
    # Max possible k is 50, but Sidon sets are smaller.
    # We will search for the best set of any size.
    
    # We can also just do a general local search on the bitmask of indices
    # But explicit set search is clearer.
    
    # Heuristic: Start with small sets and grow, or random sets of various sizes.
    # Given N=50, we can afford many random restarts.
    
    for attempt in range(2000):
        if time.time() - start_time > time_limit:
            break
            
        # Random size between 5 and 20 (Sidon sets are ~sqrt(N) ~ 7, but maybe larger sets with collisions are better?)
        # If k=10, N_max=2, Score=5.
        # If k=7, N_max=1, Score=7.
        # If k=14, N_max=2, Score=7.
        # So we want to find sets with low N_max.
        
        k = random.randint(1, 25)
        indices = random.sample(range(N), k)
        
        # Local search: Hill climbing
        # We try to swap elements or add/remove elements?
        # Let's stick to fixed size k first, then maybe vary k.
        # Actually, varying k is part of the search.
        
        current_score = calc_score(indices)
        
        # Try to improve by swapping
        for _ in range(50): # 50 swap attempts per restart
            if time.time() - start_time > time_limit:
                break
                
            i_idx = random.randint(0, k-1)
            old_val = indices[i_idx]
            
            # Pick a new value not in indices
            while True:
                new_val = random.randint(0, N-1)
                if new_val not in indices:
                    break
            
            indices[i_idx] = new_val
            new_score = calc_score(indices)
            
            if new_score >= current_score:
                current_score = new_score
                # Accept
            else:
                # Revert
                indices[i_idx] = old_val
        
        # Also try adding/removing elements to change k
        # If we can add an element without increasing N_max, score increases (k+1)/N_max > k/N_max
        # If we can remove an element and decrease N_max significantly, score might increase.
        
        # Try adding
        while True:
             if time.time() - start_time > time_limit:
                 break
             # Try to add a random element
             candidates = [x for x in range(N) if x not in indices]
             if not candidates:
                 break
             add_val = random.choice(candidates)
             indices.append(add_val)
             new_score = calc_score(indices)
             if new_score > current_score:
                 current_score = new_score
             else:
                 indices.pop()
                 break # Stop adding if it doesn't help immediately (greedy)
                 
        # Try removing
        while len(indices) > 1:
             if time.time() - start_time > time_limit:
                 break
             rem_idx = random.randint(0, len(indices)-1)
             old_val = indices[rem_idx]
             indices.pop(rem_idx)
             new_score = calc_score(indices)
             if new_score > current_score:
                 current_score = new_score
             else:
                 indices.insert(rem_idx, old_val)
                 break
        
        if current_score > best_score:
            best_score = current_score
            best_indices = indices[:]
            
    # Construct the final vector
    # The ratio is scale invariant. We can return 1s at best_indices.
    # However, to be safe against floating point issues in the evaluator,
    # and since the problem asks for "heights", returning 1.0 is fine.
    # If the evaluator normalizes, it doesn't matter.
    # If it doesn't, the ratio is invariant.
    
    f = np.zeros(N)
    for idx in best_indices:
        f[idx] = 1.0
        
    return list(f)
