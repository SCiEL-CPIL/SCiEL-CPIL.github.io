"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, combinations_with_replacement

def _get_sums(aset):
    """Get all pair sums (including a+a) from a set."""
    n = len(aset)
    sums = []
    for i in range(n):
        for j in range(i, n):
            sums.append(aset[i] + aset[j])
    return sums

def _get_diffs(aset):
    """Get all nonzero differences from a set."""
    n = len(aset)
    diffs = set()
    for i in range(n):
        for j in range(i+1, n):
            diffs.add(aset[j] - aset[i])
    return diffs

def _is_valid_sidon(aset):
    """Check if aset is a valid Sidon set (unique pair sums including a+a)."""
    sums = _get_sums(aset)
    return len(sums) == len(set(sums))

def _try_add(x, aset, sums_set, diffs_set, other_diffs):
    """Try to add x to aset. Returns (new_sums_set, new_diffs_set, success)."""
    n = len(aset)
    # Check new sums: x+a for all a in aset, and x+x
    new_sums = set()
    for a in aset:
        new_sums.add(x + a)
    new_sums.add(x + x)
    
    # Check uniqueness with existing sums
    if sums_set & new_sums:
        return None, None, False
    if len(new_sums) != len(new_sums):  # always true for set, but check internal uniqueness
        return None, None, False
    
    # Check new differences: x-a and a-x for all a in aset
    new_diffs = set()
    for a in aset:
        new_diffs.add(x - a)
        new_diffs.add(a - x)
    
    # Check disjointness with other set's differences
    if new_diffs & other_diffs:
        return None, None, False
    
    # Check new diffs don't conflict with existing diffs
    if new_diffs & diffs_set:
        return None, None, False
    
    return sums_set | new_sums, diffs_set | new_diffs, True

def _greedy_build(order, time_limit):
    """Greedy build using given order of candidates."""
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    for x in order:
        if time.time() > time_limit:
            break
        # Try adding to A
        new_sums_A, new_diffs_A, ok_A = _try_add(x, A, sums_A, diffs_A, diffs_B)
        if ok_A:
            A.append(x)
            A.sort()
            sums_A = new_sums_A
            diffs_A = new_diffs_A
            continue
        # Try adding to B
        new_sums_B, new_diffs_B, ok_B = _try_add(x, B, sums_B, diffs_B, diffs_A)
        if ok_B:
            B.append(x)
            B.sort()
            sums_B = new_sums_B
            diffs_B = new_diffs_B
            continue
        # Skip x
    
    return A, B

def _local_improve(A, B, time_limit):
    """Try to improve by swapping elements or adding/removing."""
    A = list(A)
    B = list(B)
    best_size = len(A) + len(B)
    
    while time.time() < time_limit:
        improved = False
        
        # Try adding elements not in A or B
        used = set(A) | set(B)
        for x in range(1, 257):
            if x in used:
                continue
            # Try adding to A
            new_A = A + [x]
            if _is_valid_sidon(new_A):
                new_diffs_A = _get_diffs(new_A)
                if not (new_diffs_A & _get_diffs(B)):
                    A = new_A
                    A.sort()
                    best_size = len(A) + len(B)
                    improved = True
                    break
            # Try adding to B
            new_B = B + [x]
            if _is_valid_sidon(new_B):
                new_diffs_B = _get_diffs(new_B)
                if not (new_diffs_B & _get_diffs(A)):
                    B = new_B
                    B.sort()
                    best_size = len(A) + len(B)
                    improved = True
                    break
        
        if not improved:
            # Try swapping: remove an element from A and try to add a different one
            for i in range(len(A)):
                if time.time() > time_limit:
                    return A, B
                a = A[i]
                A_without = A[:i] + A[i+1:]
                if not _is_valid_sidon(A_without):
                    continue
                diffs_A_without = _get_diffs(A_without)
                # Try to find a better element to replace a
                for x in range(1, 257):
                    if x in A_without or x in B:
                        continue
                    new_A = A_without + [x]
                    if _is_valid_sidon(new_A):
                        new_diffs_A = _get_diffs(new_A)
                        if not (new_diffs_A & _get_diffs(B)):
                            A = new_A
                            A.sort()
                            best_size = len(A) + len(B)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Try swapping in B
                for i in range(len(B)):
                    if time.time() > time_limit:
                        return A, B
                    b = B[i]
                    B_without = B[:i] + B[i+1:]
                    if not _is_valid_sidon(B_without):
                        continue
                    diffs_B_without = _get_diffs(B_without)
                    for x in range(1, 257):
                        if x in A or x in B_without:
                            continue
                        new_B = B_without + [x]
                        if _is_valid_sidon(new_B):
                            new_diffs_B = _get_diffs(new_B)
                            if not (new_diffs_B & _get_diffs(A)):
                                B = new_B
                                B.sort()
                                best_size = len(A) + len(B)
                                improved = True
                                break
                    if improved:
                        break
        
        if not improved:
            break
    
    return A, B

def construct():
    start_time = time.time()
    time_limit = 80.0  # Safety margin for 90s limit
    
    best_A = []
    best_B = []
    best_size = 0
    
    # Multiple greedy strategies with different orderings
    strategies = []
    
    # Strategy 1: Natural order
    strategies.append(list(range(1, 257)))
    
    # Strategy 2: Reverse order
    strategies.append(list(range(256, 0, -1)))
    
    # Strategy 3: Random orderings
    rng = random.Random(42)
    for _ in range(10):
        order = list(range(1, 257))
        rng.shuffle(order)
        strategies.append(order)
    
    # Strategy 4: Alternating small/large
    small = list(range(1, 129))
    large = list(range(129, 257))
    alt = []
    for i in range(128):
        alt.append(small[i])
        alt.append(large[i])
    strategies.append(alt)
    
    # Strategy 5: Prime-like spacing
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251]
    strategies.append(primes + [x for x in range(1, 257) if x not in primes])
    
    for order in strategies:
        if time.time() > time_limit - 10:
            break
        A, B = _greedy_build(order, time_limit - 5)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
    
    # Local improvement on the best solution
    if best_size > 0 and time.time() < time_limit - 2:
        A, B = _local_improve(best_A, best_B, time_limit - 1)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
    
    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    # Verify and trim if needed
    if not _is_valid_sidon(best_A):
        # Fallback: rebuild minimally
        best_A = [1]
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
