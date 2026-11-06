import itertools
import random
import time

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
    """
    Return a set of 12 distinct integers in [0, 128] that minimizes the largest
    Sidon subset size. We aim for a set where the largest Sidon subset has size 3.
    """
    best_set = None
    best_size = 12
    rng = random.Random(12345)
    
    # Start with some heuristic initial sets
    initial_candidates = [
        list(range(12)),
        list(range(0, 24, 2)),
        list(range(0, 36, 3)),
        list(range(0, 48, 4)),
    ]
    
    # Evaluate initial candidates
    for S in initial_candidates:
        sz = max_sidon_size(S)
        if sz < best_size:
            best_size = sz
            best_set = S.copy()
            if best_size == 3:
                return best_set
    
    # Local search / random restarts to find a set with max Sidon size 3
    for restart in range(500):
        # Generate a random dense set from a small range to encourage collisions
        max_val = rng.choice([20, 25, 30, 35, 40])
        S = sorted(rng.sample(range(0, max_val + 1), 12))
        
        current_size = max_sidon_size(S)
        
        if current_size < best_size:
            best_size = current_size
            best_set = S.copy()
            if best_size == 3:
                return best_set
        
        # Local search: try to improve the set
        for _ in range(200):
            if best_size == 3:
                break
            
            idx = rng.randint(0, 11)
            old_val = S[idx]
            new_val = rng.randint(0, 40)
            
            if new_val in S and new_val != old_val:
                continue
            
            S[idx] = new_val
            S.sort()
            new_size = max_sidon_size(S)
            
            if new_size < best_size:
                best_size = new_size
                best_set = S.copy()
                if best_size == 3:
                    return best_set
            else:
                # Rollback
                S[idx] = old_val
                S.sort()
    
    return best_set
