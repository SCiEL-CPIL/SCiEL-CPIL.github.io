"""Rewritten solver for: minimum number of distinct divisors of 5040 summing to 4979."""
import time
import math
from collections import deque

N = 5040
TARGET = 4979
TIME_LIMIT = 85  # safety margin before 90s kill

def _divisors_of_n(n):
    divs = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i * i != n:
                divs.append(n // i)
    return sorted(divs, reverse=True)

def _meet_in_the_middle(divs, target, deadline):
    """
    Meet-in-the-middle: split divisors into two halves, enumerate all subset sums
    for each half, then find the minimum total count that sums to target.
    """
    n = len(divs)
    mid = n // 2
    left = divs[:mid]
    right = divs[mid:]
    
    # Enumerate all subset sums for left half
    # Map: sum -> minimum count of elements
    left_sums = {0: 0}
    for d in left:
        # Collect new entries to avoid using same element twice in one iteration
        new_entries = {}
        for s, c in left_sums.items():
            ns = s + d
            nc = c + 1
            if ns <= target:
                if ns not in new_entries or nc < new_entries[ns]:
                    new_entries[ns] = nc
        for s, c in new_entries.items():
            if s not in left_sums or c < left_sums[s]:
                left_sums[s] = c
    
    # Enumerate all subset sums for right half
    right_sums = {0: 0}
    for d in right:
        new_entries = {}
        for s, c in right_sums.items():
            ns = s + d
            nc = c + 1
            if ns <= target:
                if ns not in new_entries or nc < new_entries[ns]:
                    new_entries[ns] = nc
        for s, c in new_entries.items():
            if s not in right_sums or c < right_sums[s]:
                right_sums[s] = c
    
    # Find minimum total count
    best_count = float('inf')
    best_pair = None
    for ls, lc in left_sums.items():
        rs_needed = target - ls
        if rs_needed in right_sums:
            rc = right_sums[rs_needed]
            total = lc + rc
            if total < best_count:
                best_count = total
                best_pair = (ls, rs_needed)
    
    return best_count, best_pair, left_sums, right_sums, left, right

def _recover_solution(best_pair, left_sums, right_sums, left, right):
    """Recover the actual subset that achieves the optimal count."""
    ls, rs = best_pair
    
    # Recover left subset: find subset of 'left' summing to ls with count left_sums[ls]
    left_count = left_sums[ls]
    # Use DP to recover
    left_dp = {0: []}
    for d in left:
        new_entries = {}
        for s, subset in left_dp.items():
            ns = s + d
            if ns <= ls:
                new_subset = subset + [d]
                if ns not in new_entries or len(new_subset) < len(new_entries[ns]):
                    new_entries[ns] = new_subset
        for s, subset in new_entries.items():
            if s not in left_dp or len(subset) < len(left_dp[s]):
                left_dp[s] = subset
    left_subset = left_dp.get(ls, [])
    
    # Recover right subset
    right_count = right_sums[rs]
    right_dp = {0: []}
    for d in right:
        new_entries = {}
        for s, subset in right_dp.items():
            ns = s + d
            if ns <= rs:
                new_subset = subset + [d]
                if ns not in new_entries or len(new_subset) < len(new_entries[ns]):
                    new_entries[ns] = new_subset
        for s, subset in new_entries.items():
            if s not in right_dp or len(subset) < len(right_dp[s]):
                right_dp[s] = subset
    right_subset = right_dp.get(rs, [])
    
    return left_subset + right_subset

def _greedy_solution(divs, target):
    """Greedy: pick largest divisors that fit."""
    chosen = []
    rem = target
    for d in divs:
        if d <= rem:
            chosen.append(d)
            rem -= d
            if rem == 0:
                break
    if rem == 0:
        return chosen
    return None

def _bruteforce_small(divs, target, deadline):
    """Try to find a solution with very few divisors using iterative deepening DFS."""
    # Sort descending
    divs_desc = sorted(divs, reverse=True)
    n = len(divs_desc)
    
    # Try k = 1, 2, 3, ...
    for k in range(1, 8):
        if time.time() > deadline:
            break
        # Prune: if sum of k largest < target, skip
        if sum(divs_desc[:k]) < target:
            continue
        # DFS with iterative deepening
        found = [None]
        
        def dfs(start, remaining, k_left, path):
            if k_left == 0:
                if remaining == 0:
                    found[0] = path[:]
                    return True
                return False
            # Prune: if sum of k_left largest from start onward < remaining
            if start + k_left - 1 < n:
                max_possible = sum(divs_desc[start:start+k_left])
                if max_possible < remaining:
                    return False
            for i in range(start, n):
                d = divs_desc[i]
                if d > remaining:
                    continue
                # Prune: even taking the largest k_left-1 from i+1 onward can't reach remaining-d
                if i + k_left - 1 < n and k_left > 1:
                    max_rest = sum(divs_desc[i+1:i+1+k_left-1])
                    if max_rest < remaining - d:
                        break
                path.append(d)
                if dfs(i + 1, remaining - d, k_left - 1, path):
                    return True
                path.pop()
            return False
        
        if dfs(0, target, k, []):
            return found[0]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    divs = _divisors_of_n(N)
    # Filter divisors <= target
    divs = [d for d in divs if d <= TARGET]
    divs.sort(reverse=True)
    
    # First try: brute force with small k (fast for small k)
    solution = _bruteforce_small(divs, TARGET, deadline)
    if solution is not None:
        assert sum(solution) == TARGET
        assert len(set(solution)) == len(solution)
        assert all(5040 % d == 0 for d in solution)
        return solution
    
    # Second try: meet-in-the-middle
    if time.time() < deadline - 10:
        best_count, best_pair, left_sums, right_sums, left, right = _meet_in_the_middle(
            divs, TARGET, deadline
        )
        if best_pair is not None and best_count < float('inf'):
            solution = _recover_solution(best_pair, left_sums, right_sums, left, right)
            if solution and sum(solution) == TARGET and len(set(solution)) == len(solution):
                assert all(5040 % d == 0 for d in solution)
                return solution
    
    # Fallback: greedy
    solution = _greedy_solution(divs, TARGET)
    if solution is not None:
        assert sum(solution) == TARGET
        assert len(set(solution)) == len(solution)
        assert all(5040 % d == 0 for d in solution)
        return solution
    
    # Last resort: return empty (should not happen)
    return []

# EVOLVE-BLOCK-START
# (The construct function above is the entry point; this block is for compatibility)
# EVOLVE-BLOCK-END
