"""Rewritten program using meet-in-the-middle for minimum cardinality subset sum."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit for safety
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    divisors.sort(reverse=True)
    
    # Quick check: can target be a single divisor?
    if target in divisors:
        return [target]
    
    # Try k=2: check all pairs
    for i in range(len(divisors)):
        for j in range(i + 1, len(divisors)):
            if divisors[i] + divisors[j] == target:
                return [divisors[i], divisors[j]]
            if divisors[i] + divisors[j] > target:
                break
    
    # Try k=3: check all triples with pruning
    for i in range(len(divisors)):
        if divisors[i] > target:
            continue
        for j in range(i + 1, len(divisors)):
            if divisors[i] + divisors[j] > target:
                break
            for k in range(j + 1, len(divisors)):
                s = divisors[i] + divisors[j] + divisors[k]
                if s == target:
                    return [divisors[i], divisors[j], divisors[k]]
                if s > target:
                    break
    
    # Try k=4: use meet-in-the-middle
    # Split divisors into two halves, compute all pair sums for each half
    # Then check if any pair sum from first half + pair sum from second half == target
    # with all 4 indices distinct
    
    mid = len(divisors) // 2
    left = divisors[:mid]
    right = divisors[mid:]
    
    # For left half, compute all pair sums with indices
    left_pairs = {}
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            s = left[i] + left[j]
            if s < target:
                if s not in left_pairs:
                    left_pairs[s] = []
                left_pairs[s].append((i, j))
    
    # For right half, compute all pair sums and check against left_pairs
    for i in range(len(right)):
        for j in range(i + 1, len(right)):
            s = right[i] + right[j]
            if s < target:
                need = target - s
                if need in left_pairs:
                    # Found a solution
                    li, lj = left_pairs[need][0]
                    return [left[li], left[lj], right[i], right[j]]
    
    # Try k=5: use the known greedy solution
    remainder = target
    chosen = []
    for d in range(n, 0, -1):
        if n % d == 0 and d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    
    return chosen

# EVOLVE-BLOCK-END
