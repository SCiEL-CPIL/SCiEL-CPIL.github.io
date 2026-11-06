import itertools
import random

def is_sidon(subset):
    """
    Check if a subset is a Sidon set.
    A Sidon set requires all unordered pair sums (including diagonals a+a) to be unique.
    """
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """
    Find the size of the largest Sidon subset of S.
    S is a list of distinct integers.
    """
    n = len(S)
    # Check from largest possible size down to 1
    for k in range(n, 0, -1):
        for subset in itertools.combinations(S, k):
            if is_sidon(subset):
                return k
    return 0

def construct():
    # We want to minimize the largest Sidon subset size.
    # The minimum possible is 3 (since any 4 distinct integers contain a Sidon triple).
    # We try to find a set of 12 integers where the largest Sidon subset has size 3.
    
    best_set = None
    best_size = 12
    
    # Helper to evaluate a set
    def evaluate(S):
        return max_sidon_size(S)
    
    # Strategy 1: Start with a dense arithmetic progression
    # range(12) is a good starting point as it has many sum collisions.
    initial_sets = [
        list(range(12)),
        list(range(0, 24, 2)), # Even numbers, still dense
        list(range(0, 36, 3)), # Multiples of 3
    ]
    
    # Strategy 2: Random restarts with dense initialization
    rng = random.Random(42)
    
    for _ in range(20):
        # Generate a random dense set
        # Pick 12 distinct integers from a small range [0, 30]
        # This increases the probability of sum collisions
        candidates = rng.sample(range(0, 30), 12)
        S = sorted(candidates)
        
        current_size = evaluate(S)
        
        if current_size < best_size:
            best_size = current_size
            best_set = S.copy()
            if best_size == 3:
                break
        
        # Local search from this starting point
        # Try to reduce the size further
        for _ in range(500):
            if best_size == 3:
                break
                
            idx = rng.randint(0, 11)
            old_val = S[idx]
            
            # Try to replace with a value that creates more collisions
            # Just pick a random value in [0, 30]
            new_val = rng.randint(0, 30)
            
            # Ensure distinctness
            if new_val in S and new_val != old_val:
                continue
            
            S[idx] = new_val
            new_size = evaluate(S)
            
            if new_size < best_size:
                best_size = new_size
                best_set = S.copy()
                if best_size == 3:
                    break
            else:
                # Rollback
                S[idx] = old_val

    # If we still haven't found size 3, try a more targeted approach
    # But usually, dense sets in small ranges will yield 3.
    
    if best_size > 3:
        # Fallback: Return the best we found
        pass
    else:
        # We found a set with max Sidon size 3
        pass

    return best_set
