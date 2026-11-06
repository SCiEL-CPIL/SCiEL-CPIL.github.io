"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

def _max_sidon_subset_size(s):
    """Exhaustively find the largest Sidon subset of s (all pair sums unique incl. diagonals)."""
    n = len(s)
    # Sort for consistency
    s_sorted = sorted(s)
    
    # Precompute all pair sums (i, j) with i <= j
    # For a subset to be Sidon, all sums s[i]+s[j] for i<=j must be distinct
    # (except that s[i]+s[j] = s[j]+s[i] is the same pair, so we only count i<=j)
    
    best = 0
    # Try all subsets by size, from largest to smallest
    for size in range(n, best, -1):
        found = False
        for combo in combinations(range(n), size):
            # Check if this subset is Sidon
            sums = set()
            is_sidon = True
            for i_idx in range(len(combo)):
                for j_idx in range(i_idx, len(combo)):
                    si = s_sorted[combo[i_idx]]
                    sj = s_sorted[combo[j_idx]]
                    sm = si + sj
                    if sm in sums:
                        is_sidon = False
                        break
                    sums.add(sm)
                if not is_sidon:
                    break
            if is_sidon:
                best = size
                found = True
                break
        if found:
            break
    return best

def construct():
    """Search for an ambient set of 12 distinct integers in [0,128] that minimizes
    the largest Sidon subset size."""
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin for 90s limit
    
    # Strategy: Create sets with many additive collisions.
    # A powerful construction: take elements in pairs (a, b) where a+b = c for many pairs.
    # Even better: create arithmetic progressions or structured collisions.
    
    # Precompute a good baseline
    # The original: pairs summing to 22: {0,22},{1,21},{2,20},{3,19},{4,18},{5,17}
    baseline = [0, 22, 1, 21, 2, 20, 3, 19, 4, 18, 5, 17]
    baseline_score = _max_sidon_subset_size(baseline)
    
    best_set = baseline
    best_score = baseline_score
    
    # Search strategy: generate candidate sets with heavy collision structure
    # and evaluate them.
    
    # Strategy 1: Multiple pairs summing to the same value
    # Generate sets where many pairs share the same sum
    
    random.seed(42)
    
    # Try various collision structures
    candidates = []
    
    # Type A: All pairs sum to same value (6 pairs)
    for total in range(20, 130):
        # Need 6 pairs (a_i, b_i) with a_i + b_i = total, all distinct, in [0,128]
        # a_i < b_i, a_i + b_i = total
        # a_i ranges from max(0, total-128) to min(63, total//2 - 1)
        a_min = max(0, total - 128)
        a_max = min(63, total // 2 - 1)
        if a_max >= a_min:
            # Choose 6 distinct values for a_i
            available = list(range(a_min, a_max + 1))
            if len(available) >= 6:
                for _ in range(100):
                    a_vals = random.sample(available, 6)
                    b_vals = [total - a for a in a_vals]
                    s = a_vals + b_vals
                    if len(set(s)) == 12 and all(0 <= x <= 128 for x in s):
                        candidates.append(s)
    
    # Type B: Two groups with internal collisions
    # Group 1: 6 elements with many same-sum pairs
    # Group 2: 6 elements with many same-sum pairs
    # Also cross-group collisions
    
    # Type C: Arithmetic progression-like structures
    # Elements clustered to create many sum collisions
    
    # Evaluate candidates
    for s in candidates:
        if time.time() > deadline:
            break
        score = _max_sidon_subset_size(s)
        if score < best_score:
            best_score = score
            best_set = s
    
    # If we found a better score, return it; otherwise return baseline
    return best_set
# EVOLVE-BLOCK-END
