"""Reciprocal-subsum-free set: maximize |S| for S subset [1,24] such that
for every x in S, 1/x != sum_{y in T} 1/y for any T subset (S \\ {x})."""
import time
import random
from math import lcm

DEADLINE = 55.0
_START = None

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE

# Precompute LCM of 1..24 to convert fractions to integers
_LCM = 1
for i in range(1, 25):
    _LCM = lcm(_LCM, i)

def _to_int(val):
    """Convert Fraction(1, x) to integer _LCM // x"""
    # val is expected to be 1/x, but we pass x to avoid Fraction overhead
    return _LCM // val

def _subset_sum_possible_int(vals_int, target_int):
    """
    Check if any subset of vals_int sums exactly to target_int.
    vals_int is a list of integers (scaled reciprocals).
    target_int is the integer target.
    Uses a set of reachable sums.
    """
    if not vals_int:
        return target_int == 0
    
    # If target is negative or 0 (and no empty set allowed? No, subset can be empty? 
    # The problem says "subset of the OTHER selected elements". 
    # Usually subset implies non-empty in this context? 
    # "sum of reciprocals of a subset". If subset is empty, sum is 0. 
    # 1/x is never 0, so empty subset never matches.
    # So we can ignore 0 sum or just start with {0}.
    
    reachable = {0}
    for v in vals_int:
        new_reachable = set()
        for s in reachable:
            new_s = s + v
            if new_s <= target_int: # Pruning: sums > target are useless
                new_reachable.add(new_s)
        reachable.update(new_reachable)
        
    return target_int in reachable


def is_valid(S):
    """Check if S is reciprocal-subsum-free."""
    S_set = set(S)
    if not S_set:
        return True
        
    # Precompute integer values for reciprocals
    int_vals = {x: _to_int(x) for x in S_set}
    
    # Calculate total sum of integer values
    total_sum = sum(int_vals.values())
    
    for x in S_set:
        target = int_vals[x]
        others_sum = total_sum - target
        
        # Quick check: If target > sum of others, it's impossible to form target from others
        if target > others_sum:
            continue
            
        # Exact check needed
        others_ints = [int_vals[y] for y in S_set if y != x]
        if _subset_sum_possible_int(others_ints, target):
            return False
            
    return True


def greedy_add(start_set, order):
    """Greedy add elements in given order to start_set if valid."""
    S = set(start_set)
    for x in order:
        if x in S:
            continue
        test_S = S | {x}
        if is_valid(list(test_S)):
            S = test_S
    return sorted(S)


def construct():
    global _START
    _START = time.time()
    
    best = None
    best_size = 0
    
    # Strategy 1: Greedy Add in descending order (24, 23, ..., 1)
    # This tends to produce large valid sets because large numbers have small reciprocals,
    # so their sum is small, making it hard for a new small number's large reciprocal to be matched.
    order_desc = list(range(24, 0, -1))
    cand = greedy_add([], order_desc)
    if cand and len(cand) > best_size:
        if is_valid(cand):
            best = cand
            best_size = len(cand)
            
    # Strategy 2: Greedy Add in ascending order (1, 2, ..., 24)
    order_asc = list(range(1, 25))
    cand = greedy_add([], order_asc)
    if cand and len(cand) > best_size:
        if is_valid(cand):
            best = cand
            best_size = len(cand)
            
    # Strategy 3: Greedy Add with random shuffles
    rng = random.Random(42)
    for _ in range(50):
        if _deadline_hit():
            break
        order = list(range(1, 25))
        rng.shuffle(order)
        cand = greedy_add([], order)
        if cand and len(cand) > best_size:
            if is_valid(cand):
                best = cand
                best_size = len(cand)
                
    # Strategy 4: Local Search / Improvement from best found
    if best and not _deadline_hit():
        S = set(best)
        improved = True
        while improved and not _deadline_hit():
            improved = False
            # Try adding elements
            for x in range(1, 25):
                if x in S:
                    continue
                test_S = S | {x}
                if is_valid(list(test_S)):
                    S = test_S
                    best = sorted(S)
                    best_size = len(S)
                    improved = True
                    break
            
            # If no add worked, try swapping? Or just stop adding.
            # The problem asks to maximize cardinality, so we just want to add.
            
    # Strategy 5: Greedy Remove from Full Set (slower, but might find different structures)
    if not _deadline_hit():
        S = set(range(1, 25))
        changed = True
        while changed and not _deadline_hit():
            changed = False
            # Try removing elements that are "most likely" to cause conflicts
            # Usually small elements (large reciprocals) are the problem.
            for x in sorted(S):
                if len(S) <= 1:
                    break
                S_test = S - {x}
                if is_valid(list(S_test)):
                    S = S_test
                    changed = True
                    break
                    
        cand = sorted(S)
        if cand and len(cand) > best_size:
            if is_valid(cand):
                best = cand
                best_size = len(cand)

    if best is None or best_size == 0:
        # Fallback: return a known valid set, e.g., {24}
        best = [24]
    
    return best
# EVOLVE-BLOCK-END
