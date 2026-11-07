"""Search for large sets with divisibility-incomparable subset sums."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _check_divisibility(sums_set):
    """Check no two distinct sums in the set divide one another.
    Returns True if valid (no divisibility), False otherwise."""
    s = sorted(sums_set)
    n = len(s)
    for i in range(n):
        a = s[i]
        # Check if any larger sum is a multiple of a
        # Since s is sorted, s[j] >= a for j > i
        # We need s[j] % a == 0 for some j > i
        for j in range(i + 1, n):
            if s[j] % a == 0:
                return False
    return True


def _validate_set(vals):
    """Full validation: all subset sums distinct and no two divide."""
    n = len(vals)
    if n == 0:
        return True
    
    # Generate all subset sums and check for duplicates
    sum_count = {}
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += vals[i]
        sum_count[s] = sum_count.get(s, 0) + 1
    
    # If any sum appears more than once, invalid
    for s, c in sum_count.items():
        if c > 1:
            return False
    
    # Now check divisibility among distinct sums
    sums_set = set(sum_count.keys())
    return _check_divisibility(sums_set)


def _validate_set_fast(vals):
    """Faster validation using incremental subset sum generation with early exit."""
    n = len(vals)
    if n == 0:
        return True
    
    # Generate all subset sums incrementally, checking for duplicates
    current_sums = set()
    for v in vals:
        new_sums = set()
        for s in current_sums:
            ns = s + v
            if ns in current_sums or ns in new_sums:
                return False
            new_sums.add(ns)
        if v in current_sums or v in new_sums:
            return False
        new_sums.add(v)
        current_sums |= new_sums
    
    # Check divisibility
    return _check_divisibility(current_sums)


def _try_add(val, current_sums):
    """Try adding val to a set with current subset sums.
    Returns (new_sums, valid) where valid indicates if the addition is legal."""
    new_sums = set()
    for s in current_sums:
        ns = s + val
        if ns in current_sums or ns in new_sums:
            return None, False
        new_sums.add(ns)
    if val in current_sums or val in new_sums:
        return None, False
    new_sums.add(val)
    all_sums = current_sums | new_sums
    if not _check_divisibility(all_sums):
        return None, False
    return all_sums, True


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    
    random.seed(42)
    
    # Strategy: Use a combination of approaches
    # 1. Greedy construction with multiple random orderings
    # 2. Local search to improve
    # 3. Try different candidate pools
    
    best_set = []
    best_size = 0
    
    # Candidate numbers: 1..256
    # Focus on larger numbers to reduce divisibility conflicts
    # If all numbers are > 128, subset sums are > 128, reducing chance of small dividing large
    
    # Approach 1: Greedy with random orderings on full range
    candidates_full = list(range(1, 257))
    
    for trial in range(100):
        if time.time() > deadline:
            break
        
        random.shuffle(candidates_full)
        current = []
        current_sums = set()
        
        for v in candidates_full:
            if time.time() > deadline:
                break
            
            new_sums, valid = _try_add(v, current_sums)
            if valid:
                current.append(v)
                current_sums = new_sums
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
        
        if best_size >= 30:
            break
    
    # Approach 2: Greedy focusing on larger numbers
    candidates_large = list(range(129, 257))
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        random.shuffle(candidates_large)
        current = []
        current_sums = set()
        
        for v in candidates_large:
            if time.time() > deadline:
                break
            
            new_sums, valid = _try_add(v, current_sums)
            if valid:
                current.append(v)
                current_sums = new_sums
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
        
        if best_size >= 30:
            break
    
    # Approach 3: Local search - try to swap elements to improve
    if best_size > 0 and time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            for i in range(len(best_set)):
                if time.time() > deadline - 10.0:
                    break
                for v in range(1, 257):
                    if v in best_set:
                        continue
                    new_set = best_set[:i] + [v] + best_set[i+1:]
                    if _validate_set_fast(new_set):
                        # Check if this actually improves (it's same size, but might enable further growth)
                        best_set = new_set
                        improved = True
                        break
                if improved:
                    break
        
        # Try to add more elements after local search
        if time.time() < deadline - 5.0:
            current_sums = set()
            # Rebuild sums
            for v in best_set:
                new_sums = set()
                for s in current_sums:
                    new_sums.add(s + v)
                new_sums.add(v)
                current_sums |= new_sums
            
            candidates = [v for v in range(1, 257) if v not in best_set]
            random.shuffle(candidates)
            
            for v in candidates:
                if time.time() > deadline - 5.0:
                    break
                new_sums, valid = _try_add(v, current_sums)
                if valid:
                    best_set.append(v)
                    current_sums = new_sums
                    best_size = len(best_set)
                    if best_size >= 30:
                        break
    
    # Ensure we return a valid set
    if not best_set:
        best_set = [31, 32, 34]
    
    # Final validation
    if not _validate_set(best_set):
        # Fallback to a known valid set
        best_set = [31, 32, 34]
    
    return list(best_set)

# EVOLVE-BLOCK-END
