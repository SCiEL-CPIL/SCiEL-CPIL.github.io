"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 256
CANDIDATES = list(range(2, N + 1))

def _is_valid_set(S):
    """Check if a set S satisfies the condition: for all distinct a,b,c in S, a % (b*c) != 0."""
    # Sort to help with early termination if we were checking products, but here we check all triples
    # Condition: NOT EXISTS distinct a,b,c in S such that a divides b*c
    # Equivalent: For all distinct a,b,c in S, (b*c) % a != 0
    # Note: If a > b*c, then a cannot divide b*c (since b*c < a and a > 0).
    # So we only need to check pairs (b,c) such that b*c < a.
    # Actually, since a,b,c are distinct, if a is the largest, b*c could be larger or smaller.
    # If b*c >= a, it's possible a divides b*c.
    # If b*c < a, a cannot divide b*c (unless b*c=0, which is not possible here).
    
    # Optimization: 
    # For a fixed a, we need to ensure there is no pair (b,c) from S\{a} such that b*c is a multiple of a.
    # Let M_a = {k*a | k>=1}. We need to ensure no two distinct elements from S\{a} multiply to an element in M_a.
    
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        # Check all pairs (b, c) from the remaining elements
        # To speed up, we can just iterate
        for j in range(n):
            if j == i: continue
            b = S_list[j]
            for k in range(n):
                if k == i or k == j: continue
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _get_conflicts(S):
    """Return a set of tuples (i, j, k) representing indices in S_list where conflict exists."""
    # This is expensive, so we only use it for verification or specific checks.
    pass

def _greedy_construct():
    """Greedy construction: add numbers if they don't create conflicts with existing pairs."""
    S = []
    # Sort candidates in descending order? Or ascending?
    # Larger numbers are less likely to divide products of smaller numbers?
    # If a is large, b*c must be a multiple of a. Since b,c < a (if we pick small first), b*c < a^2.
    # Let's try adding numbers in descending order.
    for a in reversed(CANDIDATES):
        # Check if adding a creates a conflict
        # Conflict exists if there are b, c in S such that (b*c) % a == 0 OR (a*b) % c == 0 OR (a*c) % b == 0
        # Wait, the condition is for ALL distinct a,b,c in the final set.
        # If we add a to S, we must check:
        # 1. For all b,c in S: (b*c) % a != 0
        # 2. For all b in S: (a*b) % c != 0 for all c in S, c!=b. 
        #    Actually, simpler: check all triples involving the new element a.
        
        valid = True
        for i in range(len(S)):
            b = S[i]
            # Check if a divides b*c for any c in S, c != b
            # And if b divides a*c for any c in S, c != a (c is from S)
            # And if c divides a*b for any c in S, c != a, c != b
            
            # Let's just check all pairs (b,c) in S
            for j in range(len(S)):
                if i == j: continue
                c = S[j]
                # Triples: (a, b, c), (b, a, c), (c, a, b)
                # 1. a | b*c ?
                if (b * c) % a == 0:
                    valid = False
                    break
                # 2. b | a*c ?
                if (a * c) % b == 0:
                    valid = False
                    break
                # 3. c | a*b ?
                if (a * b) % c == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.append(a)
    return S

def _local_search(initial_set, time_limit=45.0):
    """
    Hill climbing: try to swap an element out and another in, or just add if possible?
    Since we start with a maximal independent set (greedy), we can't just add.
    We must swap: remove x, add y.
    Or remove two, add three? That's complex.
    Let's stick to 1-for-1 swaps first.
    """
    start_time = time.time()
    S = set(initial_set)
    best_S = S.copy()
    best_size = len(S)
    
    # Precompute conflicts for current set? 
    # It's hard to update incrementally for swaps easily without checking.
    
    # To make this efficient, we need a fast validity check for a candidate set.
    # Given N=256, and set size ~10-20, checking all triples is O(K^3).
    # K~15 -> 3375 checks. Each check is modular arithmetic. Very fast.
    # We can afford many swaps.
    
    def is_valid_set_fast(S_set):
        S_list = list(S_set)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                if j == i: continue
                b = S_list[j]
                for k in range(n):
                    if k == i or k == j: continue
                    c = S_list[k]
                    if (b * c) % a == 0:
                        return False
        return True

    # Greedy initial set is likely maximal (no single addition possible).
    # Let's verify maximality and try swaps.
    
    # Optimization: Randomize the order of candidates for removal/addition to escape local optima?
    # Or use multiple restarts.
    
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        # Generate a random initial set or perturb the current best?
        # Let's restart with a different greedy strategy or random start.
        # Random start: Pick K random numbers, filter to valid set.
        # But random is bad. Let's use the greedy set but with random tie-breaking or order.
        
        # Strategy: Randomized Greedy
        candidates = CANDIDATES[:]
        random.shuffle(candidates)
        current_S = []
        for a in candidates:
            # Check if a can be added to current_S
            valid = True
            for i in range(len(current_S)):
                b = current_S[i]
                for j in range(len(current_S)):
                    if i == j: continue
                    c = current_S[j]
                    if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                current_S.append(a)
        
        if len(current_S) > best_size:
            best_S = set(current_S)
            best_size = len(current_S)
        
        # Try local search on current_S
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try all swaps
            # To speed up, only try swaps that look promising?
            # Or just iterate.
            # Number of elements K ~ 15. Candidates ~ 255.
            # Swaps: 15 * 255 = 3825.
            # Each swap check: O(K^3) ~ 3000 ops.
            # Total per pass: ~10M ops. Might be slow in Python.
            # We need to optimize the check.
            
            # Optimized check for swap:
            # Remove x, Add y.
            # New set S' = (S \ {x}) U {y}.
            # We only need to check triples involving y, and triples that might have been "safe" but are now "unsafe" because x is gone?
            # No, removing x only removes constraints. Adding y adds constraints.
            # So we only need to check if y is compatible with all pairs in (S \ {x}).
            
            S_list = list(S)
            for idx, x in enumerate(S_list):
                S_minus_x = [v for i, v in enumerate(S_list) if i != idx]
                # Check if any y in CANDIDATES \ S can be added to S_minus_x
                # We can break early if we find one.
                # To be efficient, we should check y's that are "likely" good.
                # Or just iterate y's.
                
                # Optimization: Pre-check. 
                # If y is small, it's more likely to divide products.
                # If y is large, less likely.
                
                # Let's just iterate y in random order to find a valid one quickly.
                y_candidates = [y for y in CANDIDATES if y not in S]
                random.shuffle(y_candidates)
                
                found_swap = False
                for y in y_candidates:
                    # Check if y is compatible with S_minus_x
                    # Condition: For all distinct b,c in S_minus_x, (b*c) % y != 0
                    # AND For all distinct b in S_minus_x, (y*b) % c != 0 for all c in S_minus_x, c!=b
                    # Basically, check all triples in (S_minus_x U {y}) that contain y.
                    
                    valid = True
                    # Pairs (b,c) in S_minus_x
                    for i in range(len(S_minus_x)):
                        b = S_minus_x[i]
                        for j in range(len(S_minus_x)):
                            if i == j: continue
                            c = S_minus_x[j]
                            # Triples: (y,b,c), (b,y,c), (c,y,b)
                            if (b * c) % y == 0:
                                valid = False
                                break
                            if (y * c) % b == 0:
                                valid = False
                                break
                            if (y * b) % c == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Perform swap
                        S.remove(x)
                        S.add(y)
                        improved = True
                        found_swap = True
                        if len(S) > best_size:
                            best_S = S.copy()
                            best_size = len(S)
                        break # Restart outer loop? Or continue? 
                             # Continuing might find more swaps. But the set changed.
                             # Let's break the y-loop and restart the x-loop (improved=True handles this)
                             # Actually, breaking the x-loop is safer to re-evaluate.
                
                if found_swap:
                    break # Restart the while improved loop
    
    return list(best_S)

def construct():
    start_time = time.time()
    deadline = 55.0 # Safety margin
    
    # 1. Get a baseline greedy set
    base_set = _greedy_construct()
    best_result = base_set
    
    # 2. Local search
    if time.time() - start_time < deadline:
        # Run local search with a few restarts or long run
        # The _local_search function above does randomized restarts internally
        result = _local_search(base_set, time_limit=deadline - (time.time() - start_time))
        if len(result) > len(best_result):
            best_result = result
    
    # Final verification just in case
    if _is_valid_set(best_result):
        return best_result
    else:
        # Fallback to primes if something went wrong
        from math import isqrt
        return [a for a in range(2, N+1) if all(a%d for d in range(2, isqrt(a)+1))]

# EVOLVE-BLOCK-END
