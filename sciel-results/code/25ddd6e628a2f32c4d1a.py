import random
import itertools
import time

def is_sidon(subset):
    """Check if a tuple of integers is a Sidon set (including diagonals)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        si = subset[i]
        for j in range(i, n):
            s = si + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """Find the size of the largest Sidon subset of S."""
    n = len(S)
    for k in range(n, 0, -1):
        for subset in itertools.combinations(S, k):
            if is_sidon(subset):
                return k
    return 0

def has_ap3(triple):
    """Check if a 3-element tuple (sorted) forms an arithmetic progression."""
    a, b, c = triple
    return a + c == 2 * b

def count_3ap_free_4_subsets(S):
    """
    Count the number of 4-element subsets of S that do NOT contain any 3-term AP.
    If this count is 0, then every 4-subset contains an AP, 
    which implies no 4-subset is Sidon (since APs cause sum collisions).
    Thus, the max Sidon size would be at most 3.
    """
    count = 0
    S = list(S)
    for quad in itertools.combinations(S, 4):
        quad_sorted = sorted(quad)
        # Check all 4 triples in the quad
        is_3ap_free = True
        if has_ap3(quad_sorted[0:3]):
            is_3ap_free = False
        elif has_ap3(quad_sorted[0:2] + quad_sorted[3:4]): # 0,1,3
            is_3ap_free = False
        elif has_ap3(quad_sorted[0:1] + quad_sorted[2:4]): # 0,2,3
            is_3ap_free = False
        elif has_ap3(quad_sorted[1:4]): # 1,2,3
            is_3ap_free = False
            
        if is_3ap_free:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial candidates
    candidates = [
        list(range(12)),
        list(range(0, 128, 10)), # 0,10,...110 (12 elements)
        list(range(1, 128, 10)),
        [0, 1, 2, 4, 8, 16, 32, 64, 7, 15, 31, 63], # Mix
    ]
    
    # Evaluate initial candidates
    best_set = None
    best_bad_count = float('inf')
    best_sidon_size = float('inf')
    
    for c in candidates:
        if len(c) != 12:
            continue
        # Ensure distinct and in range
        if len(set(c)) != 12:
            continue
        if min(c) < 0 or max(c) > 128:
            continue
            
        bad_count = count_3ap_free_4_subsets(c)
        if bad_count < best_bad_count:
            best_bad_count = bad_count
            best_set = c[:]
            # Calculate actual sidon size for tracking, but use bad_count for search
            # Only calc sidon size if bad_count is low to save time
            if bad_count == 0:
                best_sidon_size = 3 # Assumed, will verify later or just return
            else:
                best_sidon_size = max_sidon_size(c)
                
    # If we already found a set with 0 bad 4-sets, we likely have max sidon size 3.
    # We can try to verify or just return.
    # But let's do a quick local search to see if we can improve or confirm.
    
    current = best_set[:]
    current_bad_count = best_bad_count
    
    # Local search
    for step in range(50000): # Increased steps due to faster objective
        if time.time() > deadline:
            break
            
        # If we have 0 bad sets, we are at the theoretical minimum (3).
        # We can break early if we are confident.
        if current_bad_count == 0:
            # Verify max sidon size is indeed 3 (not 2, impossible, or 4 due to error in logic?)
            # Logic: If every 4-set has an AP, no 4-set is Sidon.
            # Is it possible a 4-set has an AP but is still Sidon?
            # AP {a, b, c} in 4-set S. a+c=2b.
            # Sums: a+a, a+b, a+c, b+b, b+c, c+c, ...
            # a+c = b+b. Collision.
            # So any set with AP is NOT Sidon.
            # So if every 4-set has AP, no 4-set is Sidon.
            # Max Sidon size <= 3.
            # Since 3-sets that are not APs are Sidon, and we have 12 elements, 
            # it's highly likely there is a non-AP 3-set.
            # So Max Sidon Size = 3.
            # We can return immediately.
            # Just to be safe, let's run max_sidon_size one last time.
            actual_size = max_sidon_size(current)
            return current

        # Pick a random index
        idx = random.randint(0, 11)
        old_val = current[idx]
        
        # Try to replace with a value that creates more APs (reduces bad_count)
        # Heuristic: Pick a value that is close to existing ones or forms APs with pairs
        # Simple random first
        new_val = random.randint(0, 128)
        if new_val in current:
            continue
            
        current[idx] = new_val
        new_bad_count = count_3ap_free_4_subsets(current)
        
        if new_bad_count < current_bad_count:
            current_bad_count = new_bad_count
            if current_bad_count < best_bad_count:
                best_bad_count = current_bad_count
                best_set = current[:]
                # Update best_sidon_size only if we think it's better
                # If best_bad_count is 0, we don't need to check sidon size repeatedly
                if best_bad_count > 0:
                    best_sidon_size = max_sidon_size(best_set)
        else:
            # Revert
            current[idx] = old_val
            
    # If we didn't find 0, return the best set found
    # Re-verify its sidon size
    actual_size = max_sidon_size(best_set)
    return best_set
