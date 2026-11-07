"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

def _exact_valid_check(elements):
    """
    Exact check: no two subsets of different cardinalities have the same sum.
    Uses cardinality-indexed subset-sum bitsets.
    Returns True if valid, False otherwise.
    """
    n = len(elements)
    if n == 0:
        return True
    
    # For each cardinality k, compute the set of achievable sums
    # sums[k] is a set of integers (or we use a boolean array)
    # Max possible sum: sum of all elements
    max_sum = sum(elements)
    
    # sums[k][s] = True if there exists a subset of size k with sum s
    # Use list of sets for sparsity, or boolean lists
    # For n up to ~40 and values up to 256, max_sum up to ~10240
    # Use sets for efficiency
    
    sums_by_card = [set() for _ in range(n + 1)]
    sums_by_card[0].add(0)
    
    for x in elements:
        # Process in reverse to avoid using x multiple times
        for k in range(n - 1, -1, -1):
            if sums_by_card[k]:
                new_sums = set()
                for s in sums_by_card[k]:
                    ns = s + x
                    if ns <= max_sum:
                        new_sums.add(ns)
                sums_by_card[k + 1] |= new_sums
    
    # Check: for any k1 != k2, sums_by_card[k1] and sums_by_card[k2] are disjoint
    for k1 in range(n + 1):
        for k2 in range(k1 + 1, n + 1):
            if sums_by_card[k1] & sums_by_card[k2]:
                return False
    return True


def _range_separation_check(elements):
    """
    Fast sufficient check: for all k1 < k2, max sum of k1 elements < min sum of k2 elements.
    This is necessary but not sufficient for validity.
    """
    n = len(elements)
    elements_sorted = sorted(elements)
    
    min_sums = [0] * (n + 1)
    max_sums = [0] * (n + 1)
    
    for k in range(1, n + 1):
        min_sums[k] = sum(elements_sorted[:k])
        max_sums[k] = sum(elements_sorted[-k:])
    
    for k1 in range(n + 1):
        for k2 in range(k1 + 1, n + 1):
            if max_sums[k1] >= min_sums[k2]:
                return False
    return True


def construct():
    """
    Constructs a set of distinct integers in 1..256 such that subset sums
    of different cardinalities are distinct.
    Uses exact cardinality-indexed subset-sum DP for validation.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Known valid solution: arithmetic progression [226, 256], size 31
    best_result = list(range(226, 257))
    best_size = 31
    
    # Verify the known solution is indeed valid
    if not _exact_valid_check(best_result):
        # Should not happen, but fallback
        return best_result[:1]
    
    # Try to find larger sets using randomized search
    # Strategy: try different sizes from 32 upward, using random sampling
    
    random.seed(42)  # Deterministic for reproducibility
    
    for target_size in range(32, 36):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random attempts for this size
        attempts = 200
        for attempt in range(attempts):
            if time.time() > deadline - 3:
                break
            
            # Generate a random set of target_size distinct integers in 1..256
            # Bias toward larger values to help with separation
            candidates = random.sample(range(1, 257), target_size)
            candidates.sort()
            
            # Quick necessary check: range separation
            if not _range_separation_check(candidates):
                continue
            
            # Exact check
            if _exact_valid_check(candidates):
                best_result = candidates
                best_size = target_size
                break  # Found valid set for this size, try next size
    
    # Also try some structured approaches for sizes 32-34
    if time.time() < deadline - 5 and best_size < 34:
        # Try arithmetic progressions with different starts and diffs
        for size in range(best_size + 1, 35):
            if time.time() > deadline - 3:
                break
            for start in range(1, 257 - size + 1):
                if time.time() > deadline - 3:
                    break
                for diff in range(1, 5):
                    if time.time() > deadline - 3:
                        break
                    end = start + (size - 1) * diff
                    if end > 256:
                        break
                    candidates = [start + i * diff for i in range(size)]
                    if _range_separation_check(candidates):
                        if _exact_valid_check(candidates):
                            best_result = candidates
                            best_size = size
                            break
                else:
                    continue
                break
            else:
                continue
            break
    
    # Final safety: ensure we return a valid list of Python ints
    result = [int(x) for x in best_result]
    assert len(set(result)) == len(result)
    assert all(1 <= x <= 256 for x in result)
    
    return result
# EVOLVE-BLOCK-END
