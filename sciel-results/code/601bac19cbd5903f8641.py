"""Direct constructive approach: exponential sequence guarantees distinct triple sums."""
import time
import random
import sys
import numpy as np


def _all_triple_sums(lst):
    """Compute all unordered triple sums for a sorted list."""
    n = len(lst)
    sums = []
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.append(lst[i] + lst[j] + lst[k])
    return sums


def _compute_all_sums_array(lst):
    """Compute all triple sums as a numpy array for fast set construction."""
    n = len(lst)
    arr = np.array(lst, dtype=np.int64)
    sums = []
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.append(int(arr[i]) + int(arr[j]) + int(arr[k]))
    return np.array(sums, dtype=np.int64)


def _is_valid_triples(lst):
    """Check if all unordered triple sums are distinct."""
    if not lst:
        return True
    sums = _all_triple_sums(lst)
    return len(sums) == len(set(sums))


def _new_triple_sums(existing, x):
    """Compute triple sums involving x with existing sorted elements."""
    n = len(existing)
    sums = [3 * x]
    for i in range(n):
        sums.append(2 * x + existing[i])
        sums.append(x + 2 * existing[i])
    for i in range(n):
        for j in range(i + 1, n):
            sums.append(x + existing[i] + existing[j])
    return sums


def _check_and_add(existing, x, existing_sums):
    """
    Check if x can be added to existing without triple sum collision.
    Returns True if added, False otherwise.
    """
    n = len(existing)
    # Generate new triple sums involving x
    new_sums = [3 * x]
    for i in range(n):
        new_sums.append(2 * x + existing[i])
        new_sums.append(x + 2 * existing[i])
    for i in range(n):
        for j in range(i + 1, n):
            new_sums.append(x + existing[i] + existing[j])
    
    # Check for collisions
    for s in new_sums:
        if s in existing_sums:
            return False
    
    # No collision, add x
    existing_sums.update(new_sums)
    existing.append(x)
    existing.sort()
    return True


def _exponential_base(b, max_val):
    """
    Build a set from powers of base b: {b, b^2, b^3, ...}.
    For base b >= 2, this gives distinct sums if b is large enough.
    Actually, for powers of 3: {3, 9, 27, 81, ...} the ternary representation 
    gives unique sums for multisets of size up to 2 (B_2 sets).
    For B_3[1], we need a stronger base. Use base b where 2*b > sum of all smaller.
    A safe choice: a_i such that 3*a_i > sum(a_0..a_{i-1}) * 3.
    Simplest: use a_i = b^i with b large enough. For 3-term sums, b >= 4 works 
    if we think in base b with digits 0..3.
    Actually the cleanest: a_i = 4^i ensures uniqueness for sums of at most 3 elements
    since max coefficient is 3 < 4.
    """
    result = []
    val = 1
    while val <= max_val:
        result.append(val)
        val *= 4
    return result


def _construct_b3_set():
    """
    Direct construction using powers of 4 as seed, then greedy extension.
    """
    # Start with powers of 4 for guaranteed validity
    base_set = _exponential_base(4, 4096)
    
    # Try to extend using greedy
    existing = list(base_set)
    existing_sums = set(_all_triple_sums(existing))
    existing_set = set(existing)
    
    for x in range(1, 4097):
        if x in existing_set:
            continue
        if _check_and_add(existing, x, existing_sums):
            existing_set.add(x)
    
    return existing


def construct():
    """
    Find a large B_3[1] set in [1, 4096] where all unordered triple sums
    (allowing repetitions) are distinct.
    """
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    best = []
    best_size = 0
    
    # Strategy 1: Exponential base construction + greedy extension
    cand = _construct_b3_set()
    if _is_valid_triples(cand):
        best = list(cand)
        best_size = len(best)
    
    # Strategy 2: Pure greedy from 1 (fast incremental)
    existing = []
    existing_sums = set()
    for x in range(1, 4097):
        _check_and_add(existing, x, existing_sums)
    if len(existing) > best_size and _is_valid_triples(existing):
        best = list(existing)
        best_size = len(best)
    
    # Strategy 3: Randomized greedy with varying step patterns
    rng = random.Random(12345)
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        selected = []
        existing_sums = set()
        step_base = rng.randint(1, 8)
        
        x = rng.randint(1, min(50, 4096))
        while x <= 4096:
            _check_and_add(selected, x, existing_sums)
            x += step_base + rng.randint(0, 5)
        
        if len(selected) > best_size:
            best = list(selected)
            best_size = len(best)
    
    # Strategy 4: Greedy with random perturbations
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        selected = []
        existing_sums = set()
        start = rng.randint(1, 100)
        min_step = rng.randint(2, 10)
        
        x = start
        while x <= 4096:
            _check_and_add(selected, x, existing_sums)
            x += min_step + rng.randint(0, min_step)
        
        if len(selected) > best_size:
            best = list(selected)
            best_size = len(best)
    
    # Strategy 5: Try to improve best by swapping elements
    if len(best) > 0 and time.time() < deadline:
        current = list(best)
        current_set = set(current)
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            existing_sums = set(_all_triple_sums(current))
            n = len(current)
            
            # Try adding any element
            added = False
            for x in range(1, 4097):
                if x in current_set:
                    continue
                if _check_and_add(current, x, existing_sums):
                    current_set.add(x)
                    added = True
                    break
            
            if not added and n > 0:
                # Remove a random element, try to find replacement
                idx = rng.randint(0, n - 1)
                removed = current.pop(idx)
                current_set.discard(removed)
                current.sort()
                existing_sums = set(_all_triple_sums(current))
                for x in range(1, 4097):
                    if x in current_set:
                        continue
                    if _check_and_add(current, x, existing_sums):
                        current_set.add(x)
                        break
                
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(best)
    
    # Final validation
    if not _is_valid_triples(best):
        # Fallback to exponential construction
        best = _construct_b3_set()
    
    return best


# Test with known small cases
if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}")
    print(f"Set: {result[:20]}...")
    print(f"Valid: {_is_valid_triples(result)}")
