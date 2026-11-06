"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_sidon(subset):
    """Check if a set of integers is Sidon (all pair sums i<=j distinct)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        ai = subset[i]
        for j in range(i, n):
            sm = ai + subset[j]
            if sm in sums:
                return False
            sums.add(sm)
    return True

def _count_sidon_subsets_size_k(arr, k):
    """Count the number of k-element Sidon subsets in arr."""
    n = len(arr)
    if k > n:
        return 0
    count = 0
    for combo in combinations(arr, k):
        if _is_sidon(combo):
            count += 1
    return count

def _has_sidon_subset_of_size_k(arr, k):
    """Check if arr contains a Sidon subset of size k. Returns True if found."""
    n = len(arr)
    if k > n:
        return False
    for combo in combinations(arr, k):
        if _is_sidon(combo):
            return True
    return False

def _max_sidon_size_fast(arr):
    """Find the largest Sidon subset size of arr. Optimized for small n."""
    n = len(arr)
    for k in range(n, 2, -1):
        if _has_sidon_subset_of_size_k(arr, k):
            return k
    if _has_sidon_subset_of_size_k(arr, 3):
        return 3
    return 2

def _count_4_sidon(arr):
    """Count 4-element Sidon subsets. Fast check for improvement."""
    return _count_sidon_subsets_size_k(arr, 4)

def construct():
    deadline = time.time() + 55.0  # Safety margin for 90s limit
    
    random.seed(42)
    
    # Best known set and its 4-Sidon count (we want this to be 0)
    best_set = None
    best_4_count = float('inf')
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Random sets
    for _ in range(200):
        candidates.append(random.sample(range(129), 12))
    
    # 2. Structured sets designed to create sum collisions
    # Sets with many arithmetic progressions or symmetric patterns
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    candidates.append([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14])
    candidates.append([0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13])
    candidates.append([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    candidates.append([0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16])
    candidates.append([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33])
    candidates.append([0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16])
    candidates.append([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24])
    
    # 3. Sets based on modular patterns
    for mod in [5, 6, 7, 8]:
        base = list(range(mod)) * 2
        if len(base) >= 12:
            candidates.append(base[:12])
    
    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        
        count4 = _count_4_sidon(cand)
        if count4 == 0:
            # Check if it has a 3-Sidon subset
            if _has_sidon_subset_of_size_k(cand, 3):
                return cand
        elif count4 < best_4_count:
            best_4_count = count4
            best_set = cand[:]
    
    # If we have a good candidate, do local search to reduce 4-Sidon count
    if best_set is None:
        best_set = random.sample(range(129), 12)
        best_4_count = _count_4_sidon(best_set)
    
    current = best_set[:]
    current_4_count = best_4_count
    
    # Local search: try to reduce the number of 4-Sidon subsets
    no_improve_counter = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_counter < max_no_improve:
        improved = False
        
        # Try multiple random moves per iteration
        for _ in range(50):
            if time.time() > deadline:
                break
            
            i = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val in current:
                continue
            
            old_val = current[i]
            current[i] = new_val
            
            new_count = _count_4_sidon(current)
            
            if new_count < current_4_count:
                # Accept the move
                current_4_count = new_count
                improved = True
                
                if new_count == 0:
                    # Check if it has a 3-Sidon subset
                    if _has_sidon_subset_of_size_k(current, 3):
                        return current
                elif new_count < best_4_count:
                    best_4_count = new_count
                    best_set = current[:]
            else:
                # Revert
                current[i] = old_val
        
        if not improved:
            no_improve_counter += 1
            # Occasional random restart to escape local minima
            if no_improve_counter % 50 == 0:
                current = random.sample(range(129), 12)
                current_4_count = _count_4_sidon(current)
                if current_4_count < best_4_count:
                    best_4_count = current_4_count
                    best_set = current[:]
        else:
            no_improve_counter = 0
    
    # Return the best set found
    return best_set
# EVOLVE-BLOCK-END
