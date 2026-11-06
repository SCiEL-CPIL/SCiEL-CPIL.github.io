import random
import itertools
import time

def is_sidon_subset(subset):
    """
    Check if a given subset (list of ints) is a Sidon set.
    All unordered pair sums a_i + a_j for i <= j must be unique.
    """
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_largest_sidon_subset(S):
    """
    Find the largest Sidon subset in S.
    Returns (size, subset)
    """
    # Check sizes from largest to smallest
    # Max possible size is len(S)
    # For 12 elements, checking 12, 11, ... is expensive if done naively.
    # But we only need to find *if* a subset of size k exists.
    
    # Optimization: If we just want to know the max size, we can check k=12, 11, etc.
    # However, combinatorial explosion is real.
    # For k=5, C(12,5) = 792. Very fast.
    # For k=6, C(12,6) = 924. Fast.
    # For k=7, C(12,7) = 792. Fast.
    # For k=8, C(12,8) = 495. Fast.
    # For k=9, C(12,9) = 220.
    # For k=10, C(12,10) = 66.
    # For k=11, C(12,11) = 12.
    # For k=12, C(12,12) = 1.
    
    # So we can just check all subsets of size k from 12 down to 1.
    # Total checks: ~3000. Each check is O(k^2). Very fast.
    
    for k in range(len(S), 0, -1):
        for subset in itertools.combinations(S, k):
            if is_sidon_subset(list(subset)):
                return k, list(subset)
    return 0, []

def construct():
    start_time = time.time()
    
    # Start with a dense set, likely to have small Sidon subsets
    # range(12) is a good candidate
    S = list(range(12))
    
    # Ensure distinct and in range (already true for range(12))
    
    # We want to minimize the max Sidon subset size.
    # Current best is whatever S has.
    current_max_size, _ = find_largest_sidon_subset(S)
    
    # If current_max_size is already small (e.g., 3 or 4), we might be done.
    # But let's try to improve.
    
    best_S = S[:]
    best_max_size = current_max_size
    
    # If we already have a max size of 3, that's likely optimal for 12 elements in [0,128]?
    # Actually, we can probably get 3.
    
    for restart in range(200):
        if time.time() - start_time > 55:
            break
            
        # Initialize S
        if restart == 0:
            S = list(range(12))
        else:
            # Randomize to escape local minima
            S = [random.randint(0, 128) for _ in range(12)]
            # Ensure distinct
            attempts = 0
            while len(set(S)) < 12 and attempts < 100:
                idx = random.randint(0, 11)
                S[idx] = random.randint(0, 128)
                attempts += 1
            if len(set(S)) < 12:
                continue

        current_max_size, _ = find_largest_sidon_subset(S)
        
        if current_max_size < best_max_size:
            best_max_size = current_max_size
            best_S = S[:]
            
        # If we found a set with max size <= 3, we can stop early as it's hard to beat
        if best_max_size <= 3:
            break
            
        # Local search to reduce max Sidon size
        for step in range(200):
            if time.time() - start_time > 55:
                break
                
            current_max_size, current_subset = find_largest_sidon_subset(S)
            
            # If we have no Sidon subsets (size 0 or 1), we are done
            if current_max_size <= 3:
                break
                
            # We have a Sidon subset of size >= 4.
            # We want to break it.
            # Pick a random element from the subset to move.
            # To be effective, we should try to move an element such that the new set
            # has a smaller max Sidon size.
            
            moved = False
            # Try to break the specific subset found
            # Shuffle indices of elements in S that are in current_subset
            indices_in_subset = [i for i, x in enumerate(S) if x in current_subset]
            random.shuffle(indices_in_subset)
            
            for idx in indices_in_subset:
                old_val = S[idx]
                
                # Try several new values
                # Prefer values that are close to existing ones to create collisions?
                # Or random?
                for _ in range(20):
                    new_val = random.randint(0, 128)
                    if new_val == old_val or new_val in S:
                        continue
                    
                    S[idx] = new_val
                    new_max_size, _ = find_largest_sidon_subset(S)
                    
                    if new_max_size < current_max_size:
                        # Improvement!
                        current_max_size = new_max_size
                        moved = True
                        break
                    else:
                        # Revert
                        S[idx] = old_val
                
                if moved:
                    break
            
            if not moved:
                # Stuck. Randomize one element completely
                idx = random.randint(0, 11)
                S[idx] = random.randint(0, 128)
                # Ensure distinct
                while S[idx] in S[:idx] or S[idx] in S[idx+1:]:
                    S[idx] = random.randint(0, 128)
                
                current_max_size, _ = find_largest_sidon_subset(S)
                if current_max_size < best_max_size:
                    best_max_size = current_max_size
                    best_S = S[:]
                    
    return best_S
