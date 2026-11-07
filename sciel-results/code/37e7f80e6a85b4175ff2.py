"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _check_cardinality_separation(elements):
    """Check if sums of subsets with different cardinalities are all distinct."""
    n = len(elements)
    # Compute all subset sums grouped by cardinality
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0].add(0)
    for x in elements:
        # Iterate backwards to avoid using x multiple times
        for k in range(n, 0, -1):
            for s in list(sums_by_k[k - 1]):
                sums_by_k[k].add(s + x)
    # Check pairwise disjointness between different cardinalities
    for k in range(n + 1):
        for j in range(k + 1, n + 1):
            if sums_by_k[k] & sums_by_k[j]:
                return False
    return True


def _try_build_set(max_val=256, target_size=60, deadline=None):
    """Try to build a valid set of given size using greedy + local search."""
    # Start with large consecutive numbers
    # Find max n such that (n-1)*(n-1) < a where a = max_val - n + 1
    # Try various starting points
    best = []
    
    # Strategy 1: Consecutive large numbers
    for start in range(max_val, 1, -1):
        # How many consecutive numbers from start going down fit?
        # Elements: start, start-1, ..., start-n+1
        # Need (n-1)*(n-1) < start for separation
        # Actually need (n-1)*(start) < n*(start-n+1) => (n-1)*start < n*start - n*(n-1) => n*(n-1) < start
        # So start > n*(n-1)
        for n in range(60, 1, -1):
            if start > n * (n - 1):
                elems = list(range(start, start - n, -1))
                if all(1 <= e <= max_val for e in elems):
                    if _check_cardinality_separation(elems):
                        if len(elems) > len(best):
                            best = elems
                break
    
    if not best:
        best = list(range(128, 136))
    
    return best


def construct():
    deadline = time.time() + 85  # 5s safety margin
    
    # Get baseline
    result = _try_build_set()
    
    # Try to improve by adding elements
    # For each candidate value not in result, check if adding it maintains validity
    current = list(result)
    used = set(current)
    
    # Try adding elements from 256 down to 1
    for val in range(256, 0, -1):
        if time.time() > deadline:
            break
        if val in used:
            continue
        candidate = current + [val]
        if _check_cardinality_separation(candidate):
            current = candidate
            used.add(val)
    
    # Try removing and re-adding to find better combinations
    # Local search: try swapping elements
    for _ in range(100):
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(current)):
            if time.time() > deadline:
                break
            for val in range(256, 0, -1):
                if time.time() > deadline:
                    break
                if val in used:
                    continue
                new_set = current[:i] + current[i+1:] + [val]
                if _check_cardinality_separation(new_set):
                    current = new_set
                    used.discard(current[i])
                    used.add(val)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    # Ensure types are correct
    result = [int(x) for x in current]
    return result

# EVOLVE-BLOCK-END
