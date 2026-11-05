"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

# Precompute LCM of 1..24
def _lcm(a, b):
    return a * b // gcd(a, b)

LCM_1_24 = reduce(_lcm, range(1, 25))

# Precompute: for each x in 1..24, the scaled value of 1/x
# scaled[x] = LCM_1_24 // x
SCALED = [0] * 25
for i in range(1, 25):
    SCALED[i] = LCM_1_24 // i

def _can_form_target(target_x, others):
    """
    Check if 1/target_x can be formed by a non-empty subset of reciprocals of 'others'.
    Uses integer arithmetic with LCM scaling.
    """
    target = SCALED[target_x]
    # Convert others to scaled values
    vals = [SCALED[y] for y in others]
    n = len(vals)
    if n == 0:
        return False
    
    # Sort descending for better pruning
    vals.sort(reverse=True)
    
    # Suffix sums for pruning
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + vals[i]
    
    # If total sum < target, impossible
    if suffix[0] < target:
        return False
    
    # Meet in the middle for larger sets
    if n <= 16:
        # Direct DFS with memoization on (idx, remaining)
        # But remaining can be large, so use set of achievable sums
        # Actually for n<=16, 2^16 = 65536 subsets, manageable
        # Use subset sum via set
        achievable = {0}
        for v in vals:
            # Add v to all existing sums
            new_sums = set()
            for s in achievable:
                ns = s + v
                if ns == target:
                    return True
                if ns < target:
                    new_sums.add(ns)
            achievable |= new_sums
            # Prune: keep only sums <= target
            achievable = {s for s in achievable if s <= target}
            if not achievable:
                return False
        return False
    else:
        # Meet in the middle
        mid = n // 2
        left = vals[:mid]
        right = vals[mid:]
        
        # Generate all subset sums for left
        left_sums = set([0])
        for v in left:
            new_sums = set()
            for s in left_sums:
                ns = s + v
                if ns <= target:
                    new_sums.add(ns)
            left_sums |= new_sums
        
        # Generate all subset sums for right
        right_sums = set([0])
        for v in right:
            new_sums = set()
            for s in right_sums:
                ns = s + v
                if ns <= target:
                    new_sums.add(ns)
            right_sums |= new_sums
        
        # Check if any left_sum + right_sum == target
        # Both must be non-empty (i.e., not both zero)
        for ls in left_sums:
            if ls == 0:
                continue
            if target - ls in right_sums:
                return True
        # Also check if left alone or right alone works (covered above since 0 is in both sets)
        # But we need non-empty subset, so at least one of ls, rs must be non-zero
        # The loop above skips ls==0, so if target is in right_sums, it will be found when ls==0 is skipped...
        # Wait, if ls==0 and target in right_sums, that's a valid subset (from right only).
        # But we skip ls==0. Let's fix:
        # Actually, we want: exists (ls, rs) in left_sums x right_sums such that ls + rs == target and (ls != 0 or rs != 0)
        # Since 0 is in both, (0,0) gives 0 != target (target > 0), so we just need any pair summing to target.
        # The loop above checks for each non-zero ls, if target-ls is in right_sums.
        # If ls==0, we skip, but target-0=target, so we'd need target in right_sums.
        # Let's add a check for ls==0 case separately, or just don't skip ls==0.
        # But (0,0) is invalid (empty subset). Since target > 0, (0,0) won't match.
        # So we can just iterate all left_sums.
        
        # Let me redo: iterate all left_sums
        for ls in left_sums:
            if target - ls in right_sums:
                # Ensure the subset is non-empty: ls != 0 or (target-ls) != 0
                # Since target > 0, if ls == 0 then target-ls = target > 0, so right part is non-empty.
                # If ls != 0, left part is non-empty.
                # The only invalid case is ls==0 and target-ls==0, i.e., target==0, which doesn't happen.
                return True
        
        return False

def _is_valid(S):
    """
    Check if S is a valid reciprocal-subsum-free set.
    For every x in S, 1/x cannot be formed by a subset of {1/y for y in S, y != x}.
    
    Key insight: If x < y, then 1/x > 1/y. For 1/x to be formed by a sum of reciprocals of other elements,
    those elements must have reciprocals that sum to 1/x. Since all reciprocals are positive, 
    any element y > x has 1/y < 1/x, so it could contribute. But elements y < x have 1/y > 1/x,
    so they alone exceed the target. However, we're summing, so a single element with 1/y > 1/x 
    would make the sum > 1/x if included. So for target 1/x, we can only use elements y where 1/y < 1/x,
    i.e., y > x. Elements with y <= x have 1/y >= 1/x, and including any of them (except x itself, which is excluded)
    would either exceed or equal. If y < x, 1/y > 1/x, so including y makes sum > 1/x. 
    So effectively, for target 1/x, only elements y > x can be used.
    
    This means: for x in S, we only need to check if 1/x can be formed by a subset of {y in S : y > x}.
    """
    S_sorted = sorted(S)
    n = len(S_sorted)
    for i in range(n):
        x = S_sorted[i]
        # Only elements greater than x can contribute
        others = [y for y in S_sorted if y > x]
        if not others:
            continue
        if _can_form_target(x, others):
            return False
    return True

def _find_best_set():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin (90s limit, leave 10s buffer)
    
    # Greedy construction: add elements in order that maximizes final size
    # Key insight: small elements are "dangerous" as targets because many larger elements can sum to them.
    # Large elements are "safe" because few smaller elements can sum to their small reciprocals.
    # So we should add large elements first, then try to add smaller ones.
    
    best_set = []
    best_size = 0
    
    # Try adding elements from 24 down to 1
    # This naturally builds a set of large elements first
    current_set = []
    for x in range(24, 0, -1):
        candidate_set = current_set + [x]
        if _is_valid(candidate_set):
            current_set = candidate_set
    
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = list(current_set)
    
    # Try to improve by adding elements that were skipped
    # The greedy might have skipped some small elements early, but maybe with a different order they could fit
    # Try local search: for each element not in set, try to add it
    current = best_set[:]
    current_set = set(current)
    all_elements = set(range(1, 25))
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Try adding any element not in set
        for x in sorted(all_elements - current_set):
            test_set = current + [x]
            if _is_valid(test_set):
                current = test_set
                current_set = set(current)
                improved = True
                break  # Restart with new set
        
        if not improved:
            # Try 1-removal + 1-addition swaps to escape local optima
            for y in sorted(current):
                reduced = [z for z in current if z != y]
                reduced_set = set(reduced)
                for z in sorted(all_elements - reduced_set):
                    if z == y:
                        continue
                    test_set = reduced + [z]
                    if _is_valid(test_set):
                        # Check if we can then add more
                        # For now, accept the swap if it doesn't decrease size
                        # (it keeps size same, but might allow future additions)
                        # To avoid cycling, only accept if z > y (we're replacing a smaller with larger? no, that reduces size potential)
                        # Actually, replacing y with z where z != y keeps size same.
                        # This might help if z is "less constraining" than y.
                        # Let's try it but with a limit to avoid infinite loops
                        current = test_set
                        current_set = set(current)
                        improved = True
                        break
                if improved:
                    break
    
    # Final check: try to add any remaining elements
    current_set = set(current)
    for x in sorted(all_elements - current_set):
        test_set = current + [x]
        if _is_valid(test_set):
            current = test_set
            current_set = set(current)
    
    return current

def construct():
    start_time = time.time()
    
    result = _find_best_set()
    
    # Ensure we return a valid non-empty set
    if not result:
        result = [24]
    
    if not _is_valid(result):
        # Fallback to a known valid set
        result = list(range(13, 25))
    
    return result
# EVOLVE-BLOCK-END
