import time
from fractions import Fraction

START_TIME = time.time()
DEADLINE = 55.0  # Safety margin for 60s limit
BEST_SET = []
BEST_SIZE = 0

def subset_sum_exists(elements, target):
    """
    Check if any subset of elements sums to target.
    elements: list of ints
    target: Fraction
    """
    sums = {Fraction(0)}
    for e in elements:
        r = Fraction(1, e)
        new_sums = set()
        for s in sums:
            ns = s + r
            if ns == target:
                return True
            if ns < target:
                new_sums.add(ns)
        sums = new_sums
    return False

def is_valid(S):
    """
    Check if S is a valid reciprocal-subsum-free set.
    """
    S_list = list(S)
    for z in S_list:
        others = [v for v in S_list if v != z]
        target = Fraction(1, z)
        if subset_sum_exists(others, target):
            return False
    return True

def greedy_initial():
    """
    Find a large valid set using a greedy heuristic.
    """
    S = set()
    # Try adding elements from largest to smallest
    for x in range(24, 0, -1):
        if is_valid(S | {x}):
            S.add(x)
    return S

def search(idx, current_set):
    global BEST_SET, BEST_SIZE
    
    if time.time() > DEADLINE:
        return
        
    if idx == 24:
        if len(current_set) > BEST_SIZE:
            BEST_SIZE = len(current_set)
            BEST_SET = list(current_set)
        return
    
    # Pruning: even if we add all remaining, we can't beat best
    if len(current_set) + (24 - idx) <= BEST_SIZE:
        return
    
    x = 24 - idx
    
    # Option 1: Include x
    new_set = current_set | {x}
    if is_valid(new_set):
        search(idx + 1, new_set)
        
    # Option 2: Exclude x
    search(idx + 1, current_set)

def construct():
    global BEST_SET, BEST_SIZE
    
    # Initialize with greedy result
    greedy_set = greedy_initial()
    BEST_SIZE = len(greedy_set)
    BEST_SET = list(greedy_set)
    
    # Perform exhaustive search with pruning
    search(0, set())
    
    if not BEST_SET:
        return [24]
        
    return BEST_SET
