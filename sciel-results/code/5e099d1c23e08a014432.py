"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

TARGET = 200
N = 100

def _feasible_addition(selected, reachable, x):
    """Check if adding x to selected keeps no subset summing to TARGET."""
    # New sums that would be formed by adding x to existing reachable sums
    # We need to ensure that for no s in reachable: s + x == TARGET
    # i.e., TARGET - x not in reachable
    return (TARGET - x) not in reachable

def _update_reachable(reachable, x):
    """Update reachable sums after adding x."""
    new_sums = set()
    for s in reachable:
        ns = s + x
        if ns <= TARGET:
            new_sums.add(ns)
    return reachable | new_sums

def _greedy_largest_first():
    """Greedy: add elements from largest to smallest if feasible."""
    selected = []
    reachable = {0}
    for x in range(N, 0, -1):
        if _feasible_addition(selected, reachable, x):
            selected.append(x)
            reachable = _update_reachable(reachable, x)
    return selected

def _greedy_smallest_first():
    """Greedy: add elements from smallest to largest if feasible."""
    selected = []
    reachable = {0}
    for x in range(1, N + 1):
        if _feasible_addition(selected, reachable, x):
            selected.append(x)
            reachable = _update_reachable(reachable, x)
    return selected

def _greedy_random_order(seed=0):
    """Greedy with random ordering."""
    import random
    rng = random.Random(seed)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    selected = []
    reachable = {0}
    for x in order:
        if _feasible_addition(selected, reachable, x):
            selected.append(x)
            reachable = _update_reachable(reachable, x)
    return selected

def _local_search_improve(initial):
    """Try to improve by swapping out elements for smaller ones, or adding more."""
    selected = list(initial)
    reachable = {0}
    for x in selected:
        reachable = _update_reachable(reachable, x)
    
    improved = True
    while improved:
        improved = False
        # Try adding any element not in selected
        for x in range(1, N + 1):
            if x not in selected:
                if _feasible_addition(selected, reachable, x):
                    selected.append(x)
                    reachable = _update_reachable(reachable, x)
                    improved = True
                    break
        # If no additions, try swapping: remove one, add another
        if not improved:
            for i, x in enumerate(selected):
                # Remove x
                new_selected = selected[:i] + selected[i+1:]
                new_reachable = {0}
                for y in new_selected:
                    new_reachable = _update_reachable(new_reachable, y)
                # Try to add a different element
                for z in range(1, N + 1):
                    if z not in new_selected:
                        if _feasible_addition(new_selected, new_reachable, z):
                            # Check if this actually helps (we're just swapping, same size)
                            # Only accept if z < x (smaller elements allow more room)
                            if z < x:
                                selected = new_selected + [z]
                                reachable = _update_reachable(new_reachable, z)
                                improved = True
                                break
                if improved:
                    break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    # Try multiple strategies
    best = []
    
    # Strategy 1: Greedy largest first
    sol1 = _greedy_largest_first()
    if len(sol1) > len(best):
        best = sol1
    
    # Strategy 2: Greedy smallest first
    if time.time() < deadline:
        sol2 = _greedy_smallest_first()
        if len(sol2) > len(best):
            best = sol2
    
    # Strategy 3: Random orders
    if time.time() < deadline:
        for seed in range(50):
            if time.time() >= deadline:
                break
            sol = _greedy_random_order(seed)
            if len(sol) > len(best):
                best = sol
    
    # Local search improvement on best found
    if time.time() < deadline:
        improved = _local_search_improve(best)
        if len(improved) > len(best):
            best = improved
    
    # Verify and return
    if not best:
        return [1]
    
    # Final verification
    reachable = {0}
    for x in best:
        reachable = _update_reachable(reachable, x)
    if TARGET in reachable:
        # Shouldn't happen, but fallback
        return _greedy_largest_first()
    
    return best
# EVOLVE-BLOCK-END
