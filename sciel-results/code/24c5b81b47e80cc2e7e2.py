"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np

def construct():
    n_max = 4096
    target_size = 80
    
    best_set = []
    
    # Run multiple trials with different strategies
    # Increase trials to find a larger set
    num_trials = 200
    
    for trial in range(num_trials):
        candidates = list(range(1, n_max + 1))
        
        # Vary the ordering strategy
        # Strategies:
        # 0: Random
        # 1: Descending
        # 2: Ascending
        # 3: Random biased to large
        # 4: Random biased to small
        # 5: Random (different seed)
        
        strategy = trial % 6
        
        if strategy == 0:
            random.seed(trial)
            random.shuffle(candidates)
        elif strategy == 1:
            candidates.sort(reverse=True)
        elif strategy == 2:
            candidates.sort()
        elif strategy == 3:
            random.seed(trial + 1000)
            # Bias towards larger numbers: sort by random * x
            # This tends to put larger numbers later if random is uniform?
            # Actually, sort by key. If key is random*x, large x has larger key range.
            # We want to pick "good" numbers.
            # Let's just use random shuffle for most, and specific orders for some.
            candidates.sort(key=lambda x: random.random() * x)
        elif strategy == 4:
            random.seed(trial + 2000)
            candidates.sort(key=lambda x: random.random() / x)
        else:
            random.seed(trial + 3000)
            random.shuffle(candidates)
        
        S = []
        existing_sums = set()
        
        # Local references for speed
        S_local = S
        sums_local = existing_sums
        
        for x in candidates:
            if len(S_local) >= target_size:
                break
            
            # Check collision incrementally
            # 1. 3x
            if 3 * x in sums_local:
                continue
            
            # 2. 2x + s
            collision = False
            for s in S_local:
                if 2 * x + s in sums_local:
                    collision = True
                    break
            if collision:
                continue
            
            # 3. x + s1 + s2
            for i in range(len(S_local)):
                si = S_local[i]
                for j in range(i, len(S_local)):
                    if x + si + S_local[j] in sums_local:
                        collision = True
                        break
                if collision:
                    break
            if collision:
                continue
            
            # No collision, add x
            S_local.append(x)
            
            # Update sums
            sums_local.add(3 * x)
            for s in S_local[:-1]:  # Exclude x itself
                sums_local.add(2 * x + s)
            
            # Add x + s1 + s2 where s1, s2 are from previous S (before x)
            # Note: S_local now contains x at the end.
            # We need sums involving x and pairs from S_local[:-1]
            prev_S = S_local[:-1]
            for i in range(len(prev_S)):
                si = prev_S[i]
                for j in range(i, len(prev_S)):
                    sums_local.add(x + si + prev_S[j])
        
        if len(S_local) > len(best_set):
            best_set = S_local[:]
        
        # Early stopping
        if len(best_set) == target_size:
            break
    
    return best_set
# EVOLVE-BLOCK-END
