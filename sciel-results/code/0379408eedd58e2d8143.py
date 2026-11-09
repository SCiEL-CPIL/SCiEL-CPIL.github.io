"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_valid(S):
    """Check if set S has no three distinct elements a,b,c with all pair sums in S."""
    S_set = set(S)
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i+1, n):
            b = S[j]
            ab = a + b
            if ab not in S_set:
                continue
            for k in range(j+1, n):
                c = S[k]
                if a + c in S_set and b + c in S_set:
                    return False
    return True

def construct():
    deadline = time.time() + 80  # safety margin before 90s limit
    n = 128
    
    # Start with the seed: all numbers from 65 to 128
    base = set(range(65, n+1))
    best = list(base)
    
    # Try to add elements from 1 to 64 greedily
    # For each candidate, check if adding it preserves validity
    candidates = list(range(1, 65))
    
    # Sort by some heuristic - try to add elements that are "safe"
    # An element x is safe to add if no two existing elements a,b have a+b=x or 
    # if x+a and x+b are not both in S for any a,b in S
    current = set(base)
    
    for x in candidates:
        if time.time() > deadline - 5:
            break
        # Check if adding x to current maintains validity
        new_set = current | {x}
        if _is_valid(list(new_set)):
            current = new_set
    
    # Try multiple orderings to get better results
    import random
    best_size = len(current)
    best_set = set(current)
    
    # Try several random orderings
    rng = random.Random(42)
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        cur = set(base)
        order = list(range(1, 65))
        rng.shuffle(order)
        for x in order:
            if time.time() > deadline - 5:
                break
            new_set = cur | {x}
            if _is_valid(list(new_set)):
                cur = new_set
        if len(cur) > best_size:
            best_size = len(cur)
            best_set = set(cur)
    
    # Also try removing some large elements and adding more small ones
    # Try different base ranges
    for start in range(55, 66):
        if time.time() > deadline - 5:
            break
        base2 = set(range(start, n+1))
        cur = set(base2)
        for x in range(1, start):
            if time.time() > deadline - 5:
                break
            new_set = cur | {x}
            if _is_valid(list(new_set)):
                cur = new_set
        if len(cur) > best_size:
            best_size = len(cur)
            best_set = set(cur)
    
    result = sorted(best_set)
    # Verify final result
    assert _is_valid(result), "Final result is invalid!"
    # Ensure all are Python ints
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
