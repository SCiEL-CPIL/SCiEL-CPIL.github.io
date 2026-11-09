"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

START_TIME = time.time()
DEADLINE = 55.0  # safety margin before 90s kill

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def check_validity_bitset(A):
    """
    Check if set A satisfies: no sum is achieved by subsets of different cardinalities.
    Uses bitset (Python int as bit array) for speed.
    """
    n = len(A)
    if n == 0:
        return True
    
    max_sum = sum(A)
    # sums_by_k[k] is a bitset where bit s is set if sum s is achievable with exactly k elements
    sums_by_k = [0] * (n + 1)
    sums_by_k[0] = 1  # bit 0 set: sum 0 with 0 elements
    
    for a in A:
        for k in range(n - 1, -1, -1):
            if sums_by_k[k]:
                # Shift left by a to add a to each achievable sum
                sums_by_k[k + 1] |= (sums_by_k[k] << a)
    
    # Check pairwise disjointness: no sum appears in two different cardinalities
    seen = 0
    for k in range(n + 1):
        if sums_by_k[k] & seen:
            return False
        seen |= sums_by_k[k]
    return True

def greedy_construct():
    """Greedy: try to add elements in a smart order."""
    A = []
    used = set()
    
    for a in range(1, 257):
        if a in used:
            continue
        if is_done():
            break
        A.append(a)
        used.add(a)
        if not check_validity_bitset(A):
            A.pop()
            used.discard(a)
    
    return A

def greedy_construct_descending():
    """Greedy with descending order."""
    A = []
    used = set()
    
    for a in range(256, 0, -1):
        if a in used:
            continue
        if is_done():
            break
        A.append(a)
        used.add(a)
        if not check_validity_bitset(A):
            A.pop()
            used.discard(a)
    
    return A

def local_search_improve(A):
    """Try to improve A by adding elements or swapping."""
    A = list(A)
    used = set(A)
    improved = True
    
    while improved and not is_done():
        improved = False
        
        # First try to add a new element
        for add_val in range(1, 257):
            if add_val in used:
                continue
            A.append(add_val)
            used.add(add_val)
            if check_validity_bitset(A):
                improved = True
                break
            else:
                A.pop()
                used.discard(add_val)
        
        if improved:
            continue
        
        # Try swapping
        for i in range(len(A)):
            if is_done():
                break
            for new_val in range(1, 257):
                if new_val in used:
                    continue
                old_val = A[i]
                A[i] = new_val
                used.discard(old_val)
                used.add(new_val)
                
                if check_validity_bitset(A):
                    improved = True
                    break
                else:
                    A[i] = old_val
                    used.discard(new_val)
                    used.add(old_val)
            if improved:
                break
    
    return A

def random_greedy():
    """Randomized greedy: shuffle candidates and greedily add."""
    A = []
    used = set()
    candidates = list(range(1, 257))
    random.shuffle(candidates)
    
    for a in candidates:
        if a in used:
            continue
        if is_done():
            break
        A.append(a)
        used.add(a)
        if not check_validity_bitset(A):
            A.pop()
            used.discard(a)
    
    return A

def construct():
    """Main entry point: try multiple strategies and return the best valid set."""
    best = []
    
    # Strategy 1: Greedy ascending
    A1 = greedy_construct()
    if len(A1) > len(best):
        best = A1
    
    if is_done():
        return best
    
    # Strategy 2: Greedy descending
    A2 = greedy_construct_descending()
    if len(A2) > len(best):
        best = A2
    
    if is_done():
        return best
    
    # Strategy 3: Local search improvement
    A3 = local_search_improve(best)
    if len(A3) > len(best):
        best = A3
    
    if is_done():
        return best
    
    # Strategy 4: Multiple random greedy attempts
    for _ in range(20):
        if is_done():
            break
        A4 = random_greedy()
        if len(A4) > len(best):
            best = A4
            # Try to improve the new best
            A5 = local_search_improve(best)
            if len(A5) > len(best):
                best = A5
    
    # Ensure we return a valid result
    if not check_validity_bitset(best):
        # Fallback to a known valid small set
        best = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Ensure all are Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
