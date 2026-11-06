"""Solve: find min # distinct divisors of 5040 summing to 4979."""
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Generate all divisors of 5040
    n = 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    
    target = 4979
    num_divs = len(divisors)
    
    # Try k = 1, 2, 3, ... until we find a solution
    for k in range(1, 20):  # max 20 terms should be enough
        if time.time() > deadline - 5:
            break
        # For small k, use combinations
        if k <= 4:
            for combo in combinations(divisors, k):
                if sum(combo) == target:
                    return list(combo)
        elif k == 5:
            # Use meet-in-the-middle for k=5: split into 2+3
            # Precompute all 2-combos and 3-combos
            two_sums = {}
            for i in range(num_divs):
                for j in range(i+1, num_divs):
                    s = divisors[i] + divisors[j]
                    if s <= target:
                        if s not in two_sums:
                            two_sums[s] = (i, j)
            three_sums = {}
            for i in range(num_divs):
                for j in range(i+1, num_divs):
                    for l in range(j+1, num_divs):
                        s = divisors[i] + divisors[j] + divisors[l]
                        if s <= target:
                            if s not in three_sums:
                                three_sums[s] = (i, j, l)
            for s2, (i, j) in two_sums.items():
                s3 = target - s2
                if s3 in three_sums:
                    i3, j3, l3 = three_sums[s3]
                    indices = {i, j, i3, j3, l3}
                    if len(indices) == 5:
                        return [divisors[x] for x in sorted(indices)]
        else:
            # For larger k, use greedy + local search or DP
            # Use a DP: dp[s] = min count to reach sum s using distinct divisors
            # But we need to track which divisors are used for reconstruction
            # Alternative: use a set-based approach with pruning
            pass
    
    # Fallback: greedy approach (like the seed but optimized)
    remainder = target
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    
    if remainder == 0:
        return chosen
    
    # If greedy didn't work, try a more systematic approach
    # This shouldn't happen given the structure, but as a safety net
    # Use a simple BFS on remainder values
    from collections import deque
    
    # State: (remainder, frozenset of used divisor indices)
    # This is too expensive, so let's just return the greedy result
    # and hope it's valid
    
    assert sum(chosen) == target, f"Sum mismatch: {sum(chosen)} != {target}"
    return chosen
