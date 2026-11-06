import time
from itertools import combinations

def construct():
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort(reverse=True)
    divisors = [d for d in divisors if d <= target]
    
    # Precompute pair sums for meet-in-the-middle
    # For k=3: find d1, d2, d3 such that d1+d2+d3 = target
    # Precompute all (d1+d2, d1, d2) pairs
    
    # First, try k=3 with meet-in-the-middle
    # Build a dict: sum -> list of (d1, d2)
    pair_sums = {}
    m = len(divisors)
    for i in range(m):
        d1 = divisors[i]
        if d1 > target:
            continue
        for j in range(i + 1, m):
            d2 = divisors[j]
            s = d1 + d2
            if s > target:
                continue
            if s not in pair_sums:
                pair_sums[s] = []
            pair_sums[s].append((d1, d2))
    
    # Now for each d3, check if target - d3 is in pair_sums
    for d3 in divisors:
        rem = target - d3
        if rem < 0:
            continue
        if rem in pair_sums:
            for (d1, d2) in pair_sums[rem]:
                # Ensure all distinct
                if d1 != d2 and d1 != d3 and d2 != d3:
                    return [d1, d2, d3]
    
    # Try k=4 with meet-in-the-middle
    # Split divisors into two halves
    half = m // 2
    first_half = divisors[:half]
    second_half = divisors[half:]
    
    # Precompute all pair sums from first half
    pair_sums_fh = {}
    for i in range(len(first_half)):
        d1 = first_half[i]
        for j in range(i + 1, len(first_half)):
            d2 = first_half[j]
            s = d1 + d2
            if s > target:
                continue
            if s not in pair_sums_fh:
                pair_sums_fh[s] = []
            pair_sums_fh[s].append((d1, d2))
    
    # Precompute all pair sums from second half
    pair_sums_sh = {}
    for i in range(len(second_half)):
        d1 = second_half[i]
        for j in range(i + 1, len(second_half)):
            d2 = second_half[j]
            s = d1 + d2
            if s > target:
                continue
            if s not in pair_sums_sh:
                pair_sums_sh[s] = []
            pair_sums_sh[s].append((d1, d2))
    
    # Now find (d1,d2) from first half and (d3,d4) from second half
    # such that d1+d2+d3+d4 = target
    for s1, pairs1 in pair_sums_fh.items():
        rem = target - s1
        if rem < 0:
            continue
        if rem in pair_sums_sh:
            for (d1, d2) in pairs1:
                for (d3, d4) in pair_sums_sh[rem]:
                    # Ensure all distinct
                    if len({d1, d2, d3, d4}) == 4:
                        return [d1, d2, d3, d4]
    
    # Try k=5 with a different approach: greedy with backtracking
    # Use the largest divisors first, then adjust
    chosen = []
    remaining = target
    for d in divisors:
        if d <= remaining:
            chosen.append(d)
            remaining -= d
            if remaining == 0:
                return chosen
    
    # If greedy didn't work, try a more careful search for k=5
    # Use DFS with strong pruning
    divisors_desc = sorted(divisors, reverse=True)
    
    def dfs_k5(idx, remaining, current, depth):
        if depth > 5:
            return None
        if remaining == 0:
            return current
        if idx >= len(divisors_desc):
            return None
        if remaining < 0:
            return None
        
        # Pruning: can we still reach remaining with available divisors?
        # Max sum of remaining slots
        slots = 5 - depth
        if idx + slots > len(divisors_desc):
            return None
        
        # Sum of largest 'slots' available
        max_possible = sum(divisors_desc[idx:idx+slots])
        if max_possible < remaining:
            return None
        
        # Sum of smallest 'slots' available
        min_possible = sum(divisors_desc[len(divisors_desc)-slots:len(divisors_desc)])
        if min_possible > remaining:
            return None
        
        # Include
        res = dfs_k5(idx + 1, remaining - divisors_desc[idx], current + [divisors_desc[idx]], depth + 1)
        if res:
            return res
        
        # Exclude
        res = dfs_k5(idx + 1, remaining, current, depth)
        if res:
            return res
        
        return None
    
    result = dfs_k5(0, target, [], 0)
    if result:
        return result
    
    # Fallback: return greedy result even if not exact (shouldn't happen)
    return chosen
