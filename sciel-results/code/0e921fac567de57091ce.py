"""Rewritten program: maximize cardinality of reciprocal-subsum-free subset of [1,24]."""
import time
import random
from math import gcd
from functools import reduce

# Deadline guard
_START = time.time()
_DEADLINE = 90.0  # seconds; evaluator kills at 90s, use 85s safety margin

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    return reduce(_lcm, lst, 1)

def _is_subsum_free(S):
    """
    Check if set S is reciprocal-subsum-free.
    For each element i in S, 1/i must NOT equal sum of 1/j for any nonempty
    subset of S \\ {i}.
    
    Uses exact integer arithmetic via LCM scaling.
    """
    if len(S) <= 1:
        return True
    
    S_list = sorted(S)
    n = len(S_list)
    
    # Precompute LCM of all elements for exact fraction arithmetic
    L = _lcm_list(S_list)
    
    # Convert reciprocals to integers: 1/x becomes L/x
    weights = [L // x for x in S_list]
    
    # For each target index t, check if weights[t] can be formed by subset of others
    for t in range(n):
        target = weights[t]
        # Available weights are all except index t
        avail = [weights[i] for i in range(n) if i != t]
        
        # Check if target can be formed by a nonempty subset of avail
        # Use set-based DP (meet-in-the-middle for larger sets)
        m = len(avail)
        if m <= 20:
            # Subset sum via set of reachable sums
            reachable = {0}
            for w in avail:
                new_reachable = set(reachable)
                for s in reachable:
                    ns = s + w
                    if ns == target:
                        return False
                    if ns < target:
                        new_reachable.add(ns)
                    elif ns == target:
                        return False
                reachable = new_reachable
            # Also check if target was already in reachable from initial 0 (empty set)
            # But empty set sum is 0, and target > 0, so we need nonempty subset
            # The DP above handles this: we start with {0} and add w to each
            # If target is found, return False
            # If not found after all, continue
        else:
            # Meet-in-the-middle for larger sets
            half = m // 2
            left = avail[:half]
            right = avail[half:]
            
            # Generate all subset sums of left
            left_sums = {0}
            for w in left:
                new_sums = set(left_sums)
                for s in left_sums:
                    ns = s + w
                    if ns <= target:
                        new_sums.add(ns)
                left_sums = new_sums
            
            # Generate all subset sums of right
            right_sums = {0}
            for w in right:
                new_sums = set(right_sums)
                for s in right_sums:
                    ns = s + w
                    if ns <= target:
                        new_sums.add(ns)
                right_sums = new_sums
            
            # Check if any left_sum + right_sum == target, with at least one nonempty
            for ls in left_sums:
                if target - ls in right_sums:
                    # Need to ensure not both are empty (sum=0)
                    if ls == 0 and (target - ls) == 0:
                        continue
                    return False
        
        # If we get here, target was not formed
        continue
    
    return True

def _greedy_build(start_set, all_elements, random_order=None):
    """
    Greedily add elements to start_set while maintaining subsum-free property.
    Returns the final set.
    """
    current = set(start_set)
    remaining = [x for x in all_elements if x not in current]
    
    if random_order:
        random.shuffle(remaining)
    
    for x in remaining:
        if time.time() > _DEADLINE - 5:
            break
        candidate = current | {x}
        if _is_subsum_free(candidate):
            current = candidate
    
    return current

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 85.0  # 85s safety margin
    
    all_elements = list(range(1, 25))
    
    best_set = set(all_elements[12:])  # Seed: 13-24, size 12
    best_size = len(best_set)
    
    # Strategy 1: Greedy from scratch with different orderings
    # Try adding from largest to smallest (likely to avoid small reciprocal sums)
    for order_type in ['desc', 'asc', 'random']:
        if time.time() > _DEADLINE - 10:
            break
        
        if order_type == 'desc':
            order = sorted(all_elements, reverse=True)
        elif order_type == 'asc':
            order = sorted(all_elements)
        else:
            order = all_elements[:]
            random.shuffle(order)
        
        current = set()
        for x in order:
            if time.time() > _DEADLINE - 5:
                break
            candidate = current | {x}
            if _is_subsum_free(candidate):
                current = candidate
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
    
    # Strategy 2: Start from seed and try to add elements
    for x in all_elements:
        if x in best_set:
            continue
        candidate = best_set | {x}
        if _is_subsum_free(candidate):
            best_set = candidate
            best_size = len(best_set)
    
    # Strategy 3: Randomized local search - try removing one and adding two
    improved = True
    while improved and time.time() < _DEADLINE - 10:
        improved = False
        for i in range(len(best_set)):
            if time.time() > _DEADLINE - 5:
                break
            # Try removing element i and adding two others
            removed = list(best_set)[i]
            base = best_set - {removed}
            for a in all_elements:
                if a in base or a == removed:
                    continue
                base_a = base | {a}
                for b in all_elements:
                    if b in base_a or b == removed or b == a:
                        continue
                    candidate = base_a | {b}
                    if len(candidate) > best_size and _is_subsum_free(candidate):
                        best_set = candidate
                        best_size = len(best_set)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    # Final check
    if not best_set:
        return list(range(13, 25))
    
    return sorted(best_set)

# Ensure the function is callable
if __name__ == "__main__":
    result = construct()
    print(result)
