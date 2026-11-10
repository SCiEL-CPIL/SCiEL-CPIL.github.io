import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    max_val = 256
    max_size = 60
    
    # S[k] is a bitset where bit i is 1 if sum i is achievable with cardinality k
    # Initialize S for empty set
    S = [0] * (max_size + 1)
    S[0] = 1  # Sum 0 with cardinality 0
    
    chosen = []
    
    for x in range(1, max_val + 1):
        if time.time() > deadline:
            break
            
        # Try to add x
        # new_S[k] = S[k] (subsets not containing x) | (S[k-1] << x) (subsets containing x)
        new_S = S[:]  # Copy current state
        current_len = len(chosen)
        
        # Update new_S from high k to low k to avoid using x multiple times in one step?
        # Actually, S is the state BEFORE adding x.
        # new_S[k] depends on S[k] and S[k-1].
        # So we can iterate k from 1 to current_len+1.
        
        # We need to check validity of new_S
        # To avoid allocating new_S if invalid, we can compute on the fly or just compute and check.
        # Given small sizes, computing new_S is cheap.
        
        for k in range(1, current_len + 2):
            new_S[k] = S[k] | (S[k-1] << x)
            
        # Check disjointness
        valid = True
        new_len = current_len + 1
        for k in range(1, new_len + 1):
            # Check against all j < k
            # We can optimize: if new_S[k] is 0, skip? No, sums exist.
            # Note: S[0] is 1. new_S[k] for k>=1 will have bits >= x.
            # So new_S[0] & new_S[k] is 0.
            for j in range(1, k):
                if (new_S[k] & new_S[j]) != 0:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            S = new_S
            chosen.append(x)
            # If we reached a good size, we can break early if we want, 
            # but greedy smallest first suggests continuing might allow more?
            # Actually, once we pick small numbers, we might block larger ones?
            # No, we are just adding.
            # We should continue to 256 to see if we can add more.
            
    # Ensure return type is list of ints
    return [int(x) for x in chosen]
