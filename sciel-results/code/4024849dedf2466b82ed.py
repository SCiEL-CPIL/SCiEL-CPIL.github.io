import itertools
import numpy as np
import time
import random

D = 11

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Generate all weight-4 vectors with entries +/-1
    all_vectors = []
    indices = range(D)
    for support in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for i, s in zip(support, signs):
                v[i] = s
            all_vectors.append(tuple(v))
    
    all_vectors = np.array(all_vectors)
    
    best_set = []
    
    # Greedy set packing
    # We can try different orderings
    
    n_vecs = len(all_vectors)
    
    # To speed up, we might want to precompute supports and sign tuples?
    # But 5280 is small.
    
    # Let's represent vectors as tuples for fast hashing and comparison?
    # Or just use the array and loop.
    
    # Optimization: Precompute support masks or just use the vectors.
    # Since D=11, we can use bitmasks for supports.
    # And we need to check signs.
    
    # Let's just use the list of tuples.
    
    # Random restarts
    for restart in range(500): # Adjust based on time
        if time.time() - start_time > time_limit:
            break
            
        # Shuffle indices
        perm = np.random.permutation(n_vecs)
        
        current_set = []
        # For fast checking, we might maintain a list of selected vectors.
        # Checking a new vector against all in current_set.
        
        # To make it faster, we can group by support?
        # If supports are disjoint or intersect < 3, always OK.
        # Only conflict if intersect >= 3 and signs match.
        
        # Let's just do the naive check first, it might be fast enough.
        
        for idx in perm:
            v = all_vectors[idx]
            
            is_ok = True
            # Check against current_set
            for u in current_set:
                # Check condition
                # Count matches in positions
                # diff = v - u
                # We need dist^2 < 4 to fail.
                # dist^2 = sum((v_i - u_i)^2)
                # If v_i == u_i, term is 0.
                # If v_i != u_i and both non-zero, term is 4 (if 1 and -1) or 0 (if same).
                # If one is 0, other non-zero, term is 1.
                
                # Faster:
                # Find intersection of supports.
                # If intersection size < 3, OK.
                # Else, check if signs are all same on intersection.
                
                # Vectorized check might be slower in Python loop.
                # Let's do it manually for D=11.
                
                # Count positions where v[i] == u[i] and v[i] != 0
                matches = 0
                # We only care if matches >= 3 AND all non-zero positions in intersection are matches?
                # No, "signs match on intersection" means v[i] == u[i] for all i in intersection.
                # This implies v[i] != 0 and u[i] != 0 and v[i] == u[i].
                
                # If there are 3 such positions, it fails.
                # Are there cases where they fail with < 3 matches?
                # If intersect size is 2, max dist^2 fail?
                # If intersect size 2, and signs differ?
                # u=(1,1,0..), v=(-1,1,1,1..). Intersect {1,2}.
                # v1=-1, u1=1 (diff). v2=1, u2=1 (match).
                # Dist^2 = 4 + 0 + 1 + 1 = 6. OK.
                # So only intersect >= 3 with ALL matches fails.
                
                # So condition: count(i where v[i]==u[i] and v[i]!=0) >= 3.
                
                # Let's compute this count.
                c = 0
                for i in range(D):
                    if v[i] != 0 and v[i] == u[i]:
                        c += 1
                        if c >= 3:
                            is_ok = False
                            break
                if not is_ok:
                    break
            
            if is_ok:
                current_set.append(v)
                
        if len(current_set) > len(best_set):
            best_set = current_set
            
        # Early termination if we hit theoretical max? 
        # Max possible is 5280.
        # But we know kissing number is 512.
        # If we find > 512, we are doing great.
        
    return np.array(best_set, dtype=np.int64)
