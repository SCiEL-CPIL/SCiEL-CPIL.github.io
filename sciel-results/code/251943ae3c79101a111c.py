"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    p = 67
    # Initialize with Paley tournament (known valid)
    squares = {x * x % p for x in range(1, p)}
    current_S = squares
    best_orientation = [0 if (j - i) % p in current_S else 1 for i in range(p) for j in range(i + 1, p)]
    
    # Local Search to find a different valid cyclic tournament or verify
    # We use a simple Hill Climbing on the set of differences S.
    # A move is swapping an element in S with an element not in S.
    
    def count_bad_sets(S):
        bad = 0
        # Optimization: Precompute out-neighborhoods for each vertex?
        # For cyclic, out-neighborhood of i is { (i+x)%p for x in S }
        # We need to check 4-subsets.
        # This is still O(n^4).
        # Given time constraints, we do few iterations.
        
        # Helper to check if a set of 4 has a dominator
        # We iterate through all 4-subsets
        for i in range(p):
            for j in range(i+1, p):
                for k in range(j+1, p):
                    for l in range(k+1, p):
                        # Check i
                        if ((j-i)%p in S) and ((k-i)%p in S) and ((l-i)%p in S):
                            continue
                        # Check j
                        if ((i-j)%p in S) and ((k-j)%p in S) and ((l-j)%p in S):
                            continue
                        # Check k
                        if ((i-k)%p in S) and ((j-k)%p in S) and ((l-k)%p in S):
                            continue
                        # Check l
                        if ((i-l)%p in S) and ((j-l)%p in S) and ((k-l)%p in S):
                            continue
                        bad += 1
        return bad

    # Since count_bad_sets is slow, we limit iterations
    start_time = time.time()
    time_limit = 100
    
    while time.time() - start_time < time_limit:
        # Pick a random swap
        s_list = list(current_S)
        not_s_list = [x for x in range(1, p) if x not in current_S]
        
        if not s_list or not not_s_list:
            break
            
        x = random.choice(s_list)
        y = random.choice(not_s_list)
        
        # Try swap
        new_S = set(current_S)
        new_S.remove(x)
        new_S.add(y)
        
        # Check cost
        # Optimization: Only check 4-subsets affected by the change?
        # The change affects pairs involving differences x and y.
        # This is complex. We do full check.
        
        cost = count_bad_sets(new_S)
        
        if cost == 0:
            # Found a new valid tournament (or same)
            current_S = new_S
            best_orientation = [0 if (j - i) % p in current_S else 1 for i in range(p) for j in range(i + 1, p)]
        # If cost > 0, reject (Hill Climbing)
        
    return {'n': p, 'orientation': best_orientation}
# EVOLVE-BLOCK-END
