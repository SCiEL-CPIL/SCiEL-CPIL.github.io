import time
import sys
from fractions import Fraction

def _lcm(a, b):
    return a * b // _gcd(a, b)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _precompute_conflicts():
    """
    Precompute for each x in 1..24, the list of bitmasks S (subsets of {1..24}\{x})
    such that sum_{y in S} 1/y == 1/x.
    We use integer arithmetic: multiply by LCM of all elements in S and x.
    Actually, simpler: use Fraction for correctness, but it's slow.
    Alternative: work with a common denominator.
    
    For a fixed x, we want subsets S of {1..24}\{x} with sum 1/y = 1/x.
    Multiply both sides by L = lcm(1,2,...,24):
    sum_{y in S} L/y = L/x.
    L/y is an integer. L/x is an integer.
    So we're looking for subsets of the multiset {L/1, L/2, ..., L/24} \ {L/x}
    that sum to L/x.
    
    L = lcm(1..24) = 5354228880
    L/x for x=1..24 are all integers.
    The values L/y for y=1..24:
    L/1 = 5354228880
    L/2 = 2677114440
    L/3 = 1784742960
    L/4 = 1338557220
    L/5 = 1070845776
    L/6 = 892371480
    L/7 = 764889840
    L/8 = 669278610
    L/9 = 594914320
    L/10 = 535422888
    L/11 = 486748080
    L/12 = 446185740
    L/13 = 411863760
    L/14 = 382444920
    L/15 = 356948592
    L/16 = 334639305
    L/17 = 314954640
    L/18 = 297457160
    L/19 = 281801520
    L/20 = 267711444
    L/21 = 254963280
    L/22 = 243374040
    L/23 = 232792560
    L/24 = 223092870
    """
    L = 5354228880
    vals = {}
    for y in range(1, 25):
        vals[y] = L // y
    
    conflicts = {}
    for x in range(1, 25):
        target = L // x
        # Available elements: all y != x
        elems = [(y, vals[y]) for y in range(1, 25) if y != x]
        # Sort by value descending for better pruning
        elems.sort(key=lambda t: t[1], reverse=True)
        
        bad_masks = []
        n = len(elems)
        
        # Suffix sums for pruning
        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + elems[i][1]
        
        def dfs(start, current_sum, current_mask):
            if current_sum == target:
                if current_mask != 0:
                    bad_masks.append(current_mask)
                return
            if current_sum > target:
                return
            if start >= n:
                return
            if current_sum + suffix[start] < target:
                return
            
            for i in range(start, n):
                y, v = elems[i]
                new_sum = current_sum + v
                if new_sum > target:
                    continue
                dfs(i + 1, new_sum, current_mask | (1 << (y - 1)))
        
        dfs(0, 0, 0)
        conflicts[x] = bad_masks
    return conflicts

def _is_valid(mask, conflicts):
    """Check if mask is reciprocal-subsum-free."""
    for x in range(1, 25):
        if (mask >> (x - 1)) & 1:
            for bm in conflicts[x]:
                if (bm & mask) == bm:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Precompute conflicts
    conflicts = _precompute_conflicts()
    
    # Strategy: try to find a valid set of maximum size
    # Start with the full set and try to remove minimum elements
    # Or: iterate k from 24 down, check if any subset of size k is valid
    
    # Quick check: is the full set valid?
    full_mask = (1 << 24) - 1
    if _is_valid(full_mask, conflicts):
        return list(range(1, 25))
    
    # Try sizes from 24 down to 1
    # For efficiency, we can use a greedy approach first to get a good starting point
    # then try to improve
    
    # Greedy: add elements in order of "least conflict"
    # Actually, let's try a different approach:
    # Start with empty set, add elements one by one
    # Order: try to add elements that don't create conflicts
    
    # Better: use the known good set {13,14,...,24} as a starting point
    # and try to add more elements
    
    known_good = set(range(13, 25))  # 13 elements
    current_mask = 0
    for x in known_good:
        current_mask |= (1 << (x - 1))
    
    # Try to add remaining elements (1-12) in some order
    # Order: try elements that are "least likely to conflict"
    # Elements 1-12 are more likely to conflict because 1/x is larger
    
    # Try adding in order: 12, 11, 10, ..., 1
    # Or try multiple orders
    
    best_mask = current_mask
    best_size = len(known_good)
    
    # Try different orders of adding elements 1-12
    remaining = list(range(12, 0, -1))
    
    # Try greedy with different starting points
    for start_elem in range(1, 13):
        if time.time() > deadline - 10:
            break
            
        trial_mask = current_mask
        # Try to add start_elem first, then the rest
        order = [start_elem] + [x for x in remaining if x != start_elem]
        
        for x in order:
            trial = trial_mask | (1 << (x - 1))
            if _is_valid(trial, conflicts):
                trial_mask = trial
        
        if bin(trial_mask).count('1') > best_size:
            best_size = bin(trial_mask).count('1')
            best_mask = trial_mask
    
    # Also try: start with a smaller known-good set and build up
    # {23, 24} is valid
    # {22, 23, 24} is valid? 1/22 = sum of others? Only 23,24. 1/23+1/24 < 1/22. OK.
    # Let's try building from {24} upward
    
    for base_size in range(1, 13):
        if time.time() > deadline - 10:
            break
        # Try all subsets of {1..base_size} of size base_size as base
        # This is expensive, so just try a few heuristics
        
        # Heuristic: take the largest base_size elements from 1..24
        base = set(range(25 - base_size, 25))
        base_mask = 0
        for x in base:
            base_mask |= (1 << (x - 1))
        
        # Try to add remaining elements
        remaining = [x for x in range(1, 25) if x not in base]
        remaining.sort(reverse=True)  # Try larger elements first
        
        trial_mask = base_mask
        for x in remaining:
            trial = trial_mask | (1 << (x - 1))
            if _is_valid(trial, conflicts):
                trial_mask = trial
        
        if bin(trial_mask).count('1') > best_size:
            best_size = bin(trial_mask).count('1')
            best_mask = trial_mask
    
    # Try a more systematic approach: for each element, decide include/exclude
    # Use backtracking with pruning
    # But this might be too slow. Let's try a limited search.
    
    # Convert best_mask to list
    result = [x for x in range(1, 25) if (best_mask >> (x - 1)) & 1]
    
    if not result:
        result = [24]
    
    return result
