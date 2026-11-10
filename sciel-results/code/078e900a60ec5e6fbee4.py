import random
import time

def is_sidon(subset):
    """Check if a subset of integers is a Sidon set (all pair sums unique)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """
    Estimate the maximum size of a Sidon subset in S.
    We only care if it's 3, 4, or 5+.
    """
    S = list(S)
    # Check if there is a Sidon subset of size 5
    # This is expensive if done exhaustively, but S is small (12).
    # C(12, 5) = 792.
    # For each, check is_sidon.
    
    # Heuristic: If we find one, return 5.
    # If not, check for 4.
    # If not, return 3 (or 2, but 3 is likely min for distinct ints > 0? 
    # Actually {0,1,2} is not Sidon. {0,1,3} is Sidon.
    # So min is 2 if we can avoid 3-APs? No, {0,1,3} is Sidon.
    # So if we have any 3 elements that are not a 3-AP, they are Sidon.
    # So K >= 3 unless the set has size < 3 or is very specific.
    # With 12 distinct ints, K is at least 3.
    
    import itertools
    
    # Check for size 5
    for combo in itertools.combinations(S, 5):
        if is_sidon(combo):
            return 5
            
    # Check for size 4
    for combo in itertools.combinations(S, 4):
        if is_sidon(combo):
            return 4
            
    return 3

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit
    
    # Seed set
    seed_set = list(range(12))
    seed_k = max_sidon_size(seed_set)
    
    best_set = seed_set
    best_k = seed_k
    
    # If seed_k is 3, we can't do better (min is 3 for 12 distinct ints? 
    # Actually, can we have K=2? 
    # K=2 means no 3 Sidon subset. 
    # No 3 Sidon subset means every 3-subset has a collision. 
    # Collision in 3 means 3-AP. 
    # So set must have no 3-AP. 
    # But we established that if no 3-AP, then every 3-subset IS Sidon. 
    # Contradiction. 
    # So K is always >= 3 for any set of size >= 3.
    # So min possible K is 3.
    
    if best_k == 3:
        return best_set

    # Search for a set with K < best_k
    # Target: 3 (if best_k is 4 or 5)
    
    # Strategy: Random search in small range [0, 30]
    # Sets in small ranges have more collisions.
    
    for _ in range(10000):
        if time.time() > deadline:
            break
            
        # Generate random set of 12 distinct ints in [0, 30]
        # Using a smaller range increases collision probability
        candidate = random.sample(range(30), 12)
        
        k = max_sidon_size(candidate)
        
        if k < best_k:
            best_k = k
            best_set = candidate
            if best_k == 3:
                return best_set
                
    # If we didn't find 3, try local search around best_set
    # Swap elements to increase collisions
    
    current_set = best_set[:]
    for _ in range(5000):
        if time.time() > deadline:
            break
            
        # Pick a random index
        i = random.randint(0, 11)
        # Pick a new value
        new_val = random.randint(0, 30)
        
        # Ensure distinct
        if new_val in current_set and new_val != current_set[i]:
            continue
            
        old_val = current_set[i]
        current_set[i] = new_val
        
        k = max_sidon_size(current_set)
        
        if k < best_k:
            best_k = k
            best_set = current_set[:]
            if best_k == 3:
                return best_set
        else:
            # Revert if not better (greedy)
            current_set[i] = old_val
            
    return best_set
