import time
import random
import itertools
import numpy as np

def is_sidon(subset):
    """Check if a subset of integers has all unique pair sums (including diagonals)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(A):
    """
    Compute the maximum size of a Sidon subset in A.
    A is a list of distinct integers.
    """
    n = len(A)
    # Check from largest possible size down to 1
    for k in range(n, 0, -1):
        for combo in itertools.combinations(A, k):
            if is_sidon(combo):
                return k
    return 0

def count_collision_pairs(A):
    """
    Count the number of duplicate unordered pair sums (i <= j) in A.
    More collisions = harder to find large Sidon subsets.
    """
    n = len(A)
    sums = []
    for i in range(n):
        for j in range(i, n):
            sums.append(A[i] + A[j])
    # Number of duplicates = total pairs - unique sums
    return len(sums) - len(set(sums))

def has_sidon_subset(A, k):
    """
    Check if A has ANY Sidon subset of size k.
    """
    n = len(A)
    for combo in itertools.combinations(A, k):
        if is_sidon(combo):
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Theoretical minimum: for 12 elements, we need at least 12*13/2 = 78 unique sums
    # Max sum is 256, min sum is 0, so we have 257 possible sums.
    # A Sidon set of size k needs k(k+1)/2 unique sums.
    # For k=12: 78 sums needed. 257 available, so it's possible in principle.
    # But we want to MINIMIZE the max Sidon subset size.
    
    # Strategy: Start with a set that has many sum collisions.
    # Arithmetic progressions have many collisions.
    # e.g., {0, 1, 2, ..., 11} has many collisions.
    # Let's start with a dense AP and then perturb to maximize collisions while keeping size 12.
    
    # Initial candidate: dense AP
    best_A = list(range(12))
    best_val = max_sidon_size(best_A)
    best_collisions = count_collision_pairs(best_A)
    
    # If we can get max_sidon_size down to, say, 4 or 5, that's great.
    # The theoretical lower bound: 
    # If max Sidon subset is k, then no k+1 subset is Sidon.
    # We want to minimize k.
    
    # Local search: swap elements to increase collision count
    # and decrease max Sidon size.
    
    current_A = list(best_A)
    
    # Multi-start with different initial patterns
    initial_patterns = [
        list(range(12)),  # dense AP
        list(range(0, 24, 2)),  # sparse AP
        list(range(0, 25, 2))[:12],
        [0, 1, 2, 4, 8, 16, 32, 64, 96, 120, 124, 128],  # powers of 2 + extras
        [0, 1, 3, 7, 15, 31, 63, 95, 119, 127, 128, 126],  # 2^k - 1 pattern
    ]
    
    for pattern in initial_patterns:
        if time.time() > deadline:
            break
        current_A = list(pattern)
        # Ensure distinct and in range
        if len(set(current_A)) != 12:
            continue
        if any(x < 0 or x > 128 for x in current_A):
            continue
        
        val = max_sidon_size(current_A)
        col = count_collision_pairs(current_A)
        
        if val < best_val or (val == best_val and col > best_collisions):
            best_A = list(current_A)
            best_val = val
            best_collisions = col
    
    # Now do local search from best_A
    current_A = list(best_A)
    no_improve = 0
    
    while time.time() < deadline:
        # Pick a random index to modify
        idx = random.randint(0, 11)
        old_val = current_A[idx]
        
        # Try to find a new value that increases collisions or decreases max Sidon size
        # Strategy: pick a value that creates a sum collision with existing pairs
        # For a new value v, the new sums are v + A[i] for i != idx, and v + v
        # We want these to collide with existing sums.
        
        # Compute existing sums (excluding pairs involving idx)
        other_indices = [i for i in range(12) if i != idx]
        existing_sums = set()
        for i in range(12):
            for j in range(i, 12):
                if i == idx or j == idx:
                    continue
                existing_sums.add(current_A[i] + current_A[j])
        
        # Also include sums that involve idx with old_val
        for i in other_indices:
            existing_sums.add(old_val + current_A[i])
        existing_sums.add(2 * old_val)
        
        # Try candidate values
        best_candidate = None
        best_candidate_score = -1
        
        # Generate candidates: values that would create collisions
        candidates = set()
        
        # For each existing sum s, and each i != idx, 
        # if we set v such that v + current_A[i] = s, then v = s - current_A[i]
        for s in list(existing_sums)[:50]:  # Limit to avoid too many
            for i in other_indices:
                v = s - current_A[i]
                if 0 <= v <= 128:
                    candidates.add(v)
        
        # Also try v such that 2*v = s
        for s in list(existing_sums)[:50]:
            if s % 2 == 0:
                v = s // 2
                if 0 <= v <= 128:
                    candidates.add(v)
        
        # Add some random candidates
        for _ in range(20):
            candidates.add(random.randint(0, 128))
        
        # Add values near old_val
        for delta in range(-5, 6):
            v = old_val + delta
            if 0 <= v <= 128:
                candidates.add(v)
        
        for v in candidates:
            if v == old_val:
                continue
            if v in current_A:
                continue
            
            new_A = list(current_A)
            new_A[idx] = v
            
            # Evaluate
            new_val = max_sidon_size(new_A)
            new_col = count_collision_pairs(new_A)
            
            # Score: prefer lower max_sidon_size, then higher collisions
            score = -new_val * 1000 + new_col
            
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = v
                best_candidate_val = new_val
                best_candidate_col = new_col
        
        if best_candidate is not None:
            current_A[idx] = best_candidate
            
            if best_candidate_val < best_val or (best_candidate_val == best_val and best_candidate_col > best_collisions):
                best_A = list(current_A)
                best_val = best_candidate_val
                best_collisions = best_candidate_col
                no_improve = 0
            else:
                no_improve += 1
            
            # If we've found a very good solution, do a final polish
            if best_val <= 4 and no_improve > 100:
                break
        else:
            no_improve += 1
        
        # Periodic restart if stuck
        if no_improve > 200:
            # Try a random restart
            current_A = list(best_A)
            for _ in range(3):
                idx = random.randint(0, 11)
                current_A[idx] = random.randint(0, 128)
                while current_A.count(current_A[idx]) > 1:
                    current_A[idx] = random.randint(0, 128)
            no_improve = 0
    
    return best_A
