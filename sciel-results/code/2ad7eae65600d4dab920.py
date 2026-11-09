"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute prime factors for 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    # prime_factors[1] remains empty
    
    def is_compatible(a, b):
        return not (prime_factors[a] & prime_factors[b])
    
    def get_sum(s):
        return sum(s)
    
    # 1. Initial Solution: Greedy by value (descending)
    # We start with 1. Then we try to add numbers from N down to 2 if compatible.
    best_set = [1]
    for a in range(N, 1, -1):
        if all(is_compatible(a, b) for b in best_set):
            best_set.append(a)
    
    best_sum = get_sum(best_set)
    
    # 2. Local Search / Hill Climbing
    # Try to replace one element with a larger one, or add a new one if possible?
    # Actually, since we started greedy descending, we likely have the "largest" possible set.
    # But greedy is not optimal. We can try to swap elements to increase sum.
    # Or, more effectively, try to add elements that were skipped if we remove something smaller?
    # A better local search: 
    # - Try adding any number not in set if compatible.
    # - Try swapping any number in set with a larger number not in set, if compatible with the rest.
    
    # We maintain a boolean array for membership for O(1) checks
    in_set = [False] * (N + 1)
    for a in best_set:
        in_set[a] = True
    
    # Precompute compatibility matrix? N=256 is small enough.
    # compat[i][j] = is_compatible(i, j)
    compat = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            if i == j:
                compat[i][j] = False
            else:
                compat[i][j] = is_compatible(i, j)
                compat[j][i] = compat[i][j]
    
    # Helper to check if x can be added to set S (given in_set and set S)
    # Optimization: iterate over S
    def can_add(x, S):
        for y in S:
            if not compat[x][y]:
                return False
        return True
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Try to add any missing number
        for x in range(N, 1, -1):
            if not in_set[x]:
                if can_add(x, best_set):
                    best_set.append(x)
                    in_set[x] = True
                    best_sum += x
                    improved = True
        
        # Pass 2: Try to swap any element in set with a larger element not in set
        # To be efficient, we iterate x (candidate to add) in descending order.
        # For each x, we look for a y in set such that y < x and x is compatible with (set - {y}).
        # If such y exists, we replace y with x.
        # We only need to find one such y for a given x to make an improvement (since x > y).
        
        # We should be careful not to swap if it doesn't improve sum. x > y ensures sum increases.
        # We iterate x from N down to 1.
        # If x is not in set:
        #   Find a y in set such that y < x and x is compatible with all z in set \ {y}.
        #   If found, replace y with x.
        
        # To make this faster, we can precompute for each x, which y's it is incompatible with.
        # If x is incompatible with only a small number of elements in the set, it's a good candidate.
        
        for x in range(N, 1, -1):
            if in_set[x]:
                continue
            
            # Check if x is compatible with all elements in best_set
            # If yes, we would have added it in Pass 1. So it must be incompatible with at least one.
            # We need to find a y in best_set to remove such that x becomes compatible with the rest.
            
            # Let's collect the elements in best_set that are incompatible with x.
            incompatible_in_set = [y for y in best_set if not compat[x][y]]
            
            # If incompatible_in_set is empty, Pass 1 should have caught it. 
            # But maybe Pass 1 failed due to order? No, Pass 1 checks all.
            # So incompatible_in_set is non-empty.
            
            # We can only remove ONE element y. So we must remove ALL incompatible elements?
            # No, the constraint is that x must be coprime with ALL remaining elements.
            # So if x is incompatible with y1 and y2, we must remove both y1 and y2.
            # But we are only doing 1-for-1 swaps.
            # So this simple swap only works if x is incompatible with exactly ONE element y in the set.
            # If it is incompatible with more than one, we can't fix it with a single swap.
            
            if len(incompatible_in_set) == 1:
                y = incompatible_in_set[0]
                if x > y:
                    # Swap y with x
                    best_set.remove(y)
                    in_set[y] = False
                    best_set.append(x)
                    in_set[x] = True
                    best_sum += (x - y)
                    improved = True
            elif len(incompatible_in_set) == 0:
                # Should not happen if Pass 1 worked, but just in case
                best_set.append(x)
                in_set[x] = True
                best_sum += x
                improved = True
    
    # Final verification to ensure validity (safety net)
    # Check distinctness
    if len(best_set) != len(set(best_set)):
        best_set = list(set(best_set))
    
    # Check pairwise coprime
    valid_set = []
    for a in sorted(best_set, reverse=True):
        if all(math.gcd(a, b) == 1 for b in valid_set):
            valid_set.append(a)
    
    # Ensure 1 is in the set? The task says "Integer one is allowed once". It doesn't say it MUST be.
    # But maximizing sum usually includes 1 because it's coprime to everything and adds 1.
    # If 1 is not in valid_set, we should try to add it.
    if 1 not in valid_set:
        # 1 is coprime to everything, so we can always add it if not present
        valid_set.append(1)
    
    # Convert to list of ints
    result = [int(x) for x in valid_set]
    return result
# EVOLVE-BLOCK-END
