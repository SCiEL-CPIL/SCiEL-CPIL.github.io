import math
import random
import time

def construct():
    N = 512
    # Precompute LCMs or use gcd
    def is_valid_pair(a, b):
        return (a * b) // math.gcd(a, b) <= N

    # Initial candidate: The set derived analytically
    # {1, 2, 4, 8, 16, 32, 64, 3, 6, 12, 24, 48, 96, 192, 384}
    initial_set = [1, 2, 4, 8, 16, 32, 64, 3, 6, 12, 24, 48, 96, 192, 384]
    
    # Verify initial set
    for i in range(len(initial_set)):
        for j in range(i+1, len(initial_set)):
            if not is_valid_pair(initial_set[i], initial_set[j]):
                # Fallback to divisors of 512 if initial is invalid
                initial_set = list(range(1, N+1))
                # Filter to divisors
                initial_set = [x for x in initial_set if N % x == 0]
                break

    best_set = initial_set.copy()
    best_size = len(best_set)
    
    # Local Search / Hill Climbing
    # Try to add elements
    # Try to swap elements
    
    # To be efficient, we can just return the best known valid set.
    # The problem asks to maximize cardinality.
    # The set of size 15 is valid.
    # Is there a larger one?
    # Let's try to construct a larger one programmatically if time permits, 
    # but returning the valid 15 is better than 10.
    
    # Simple improvement: Try adding all possible elements to the set
    # and keep the largest valid subset found via greedy addition.
    
    def greedy_addition(start_set):
        S = set(start_set)
        # Sort candidates by "potential" (e.g., number of conflicts)
        # Or just try all
        candidates = [x for x in range(1, N+1) if x not in S]
        # Heuristic: prefer smaller numbers? Or numbers that divide N?
        # Let's just try to add any that fit.
        changed = True
        while changed:
            changed = False
            for x in candidates:
                if x in S:
                    continue
                # Check if x can be added
                can_add = True
                for y in S:
                    if not is_valid_pair(x, y):
                        can_add = False
                        break
                if can_add:
                    S.add(x)
                    changed = True
        return S

    # Run greedy addition from initial set
    current_set = greedy_addition(initial_set)
    
    if len(current_set) > best_size:
        best_set = list(current_set)
        best_size = len(best_set)

    # Also try starting from divisors of 512
    divisors = [x for x in range(1, N+1) if N % x == 0]
    current_set2 = greedy_addition(divisors)
    if len(current_set2) > best_size:
        best_set = list(current_set2)
        best_size = len(best_set)

    return best_set
