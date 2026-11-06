"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_valid(subset, target=200):
    """Check if no subset of `subset` sums to `target`."""
    # Use DP to check if target is achievable
    reachable = {0}
    for x in subset:
        new = set()
        for s in reachable:
            ns = s + x
            if ns == target:
                return False
            if ns < target:
                new.add(ns)
        reachable |= new
    return True

def _greedy_construct(order, target=200):
    """Greedy: add elements in given order if valid."""
    selected = []
    reachable = {0}
    for x in order:
        new = {s + x for s in reachable if s + x <= target}
        if target not in new:
            selected.append(x)
            reachable |= new
    return selected

def _try_add_element(selected, reachable, x, target=200):
    """Try to add x to selected; return (new_selected, new_reachable, success)."""
    new = {s + x for s in reachable if s + x <= target}
    if target not in new:
        return selected + [x], reachable | new, True
    return selected, reachable, False

def construct():
    deadline = time.time() + 75  # 75s safety margin for 90s limit
    best = None
    best_size = 0
    
    # First: deterministic greedy from high to low (baseline)
    baseline = _greedy_construct(range(100, 0, -1))
    best = baseline
    best_size = len(baseline)
    
    # Try [66, 100] which we know is valid (35 elements)
    candidate = list(range(66, 101))
    if _is_valid(candidate):
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
    
    # Try to improve by adding smaller elements to [66,100]
    base = list(range(66, 101))
    reachable = {0}
    for x in base:
        reachable |= {s + x for s in reachable if s + x <= 200}
    
    # Try adding elements from 1 to 65
    for x in range(1, 66):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            base.append(x)
            reachable |= new
    
    if len(base) > best_size:
        best = base
        best_size = len(base)
    
    # Randomized restarts
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline and attempts < 500:
        attempts += 1
        # Random order with bias toward larger numbers
        order = list(range(1, 101))
        # Shuffle but keep some structure
        rng.shuffle(order)
        # Sort by descending value with random perturbation
        order.sort(key=lambda x: -x + rng.random() * 10)
        
        result = _greedy_construct(order)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Local search: try to add one element at a time to best
    if best and time.time() < deadline - 5:
        # Rebuild reachable for best
        reachable = {0}
        for x in best:
            reachable |= {s + x for s in reachable if s + x <= 200}
        
        # Try adding each missing element
        missing = set(range(1, 101)) - set(best)
        added = True
        while added and time.time() < deadline - 5:
            added = False
            for x in sorted(missing, reverse=True):
                new = {s + x for s in reachable if s + x <= 200}
                if 200 not in new:
                    best.append(x)
                    reachable |= new
                    missing.discard(x)
                    best_size = len(best)
                    added = True
                    break
    
    return best
# EVOLVE-BLOCK-END
