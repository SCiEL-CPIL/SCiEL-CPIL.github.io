import time
import random
from itertools import combinations

def _reachable_sums(elements, target):
    """Compute all reachable subset sums up to target."""
    reachable = {0}
    for x in elements:
        if x > target:
            continue
        new = {s + x for s in reachable if s + x <= target}
        reachable |= new
    return reachable

def _has_sum_200(elements):
    """Check if any subset of elements sums to 200."""
    reachable = {0}
    for x in elements:
        if x > 200:
            continue
        new = {s + x for s in reachable if s + x <= 200}
        if 200 in new:
            return True
        reachable |= new
    return False

def _greedy(order, target=200):
    """Greedy: process elements in given order, include if doesn't create target sum."""
    selected = []
    reachable = {0}
    for x in order:
        if x > target:
            selected.append(x)
            continue
        new = {s + x for s in reachable if s + x <= target}
        if target not in new:
            selected.append(x)
            reachable |= new
    return selected

def _local_search(solution, all_elements, target=200, max_iters=1000):
    """Try to improve solution by swapping or adding elements."""
    selected = set(solution)
    excluded = set(all_elements) - selected
    reachable = _reachable_sums(list(selected), target)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try to add back excluded elements
        for x in sorted(excluded):
            if x > target:
                selected.add(x)
                excluded.discard(x)
                improved = True
                continue
            new = {s + x for s in reachable if s + x <= target}
            if target not in new:
                selected.add(x)
                excluded.discard(x)
                reachable |= new
                improved = True
        # Try swaps: remove one included, add one excluded
        if not improved:
            for x in sorted(excluded):
                if x > target:
                    continue
                for y in list(selected):
                    if y > target:
                        continue
                    # Try removing y and adding x
                    new_selected = (selected - {y}) | {x}
                    if not _has_sum_200(list(new_selected)):
                        selected = new_selected
                        excluded = set(all_elements) - selected
                        reachable = _reachable_sums(list(selected), target)
                        improved = True
                        break
                if improved:
                    break
    return list(selected)

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    all_elements = list(range(1, 101))
    
    # Try multiple orderings
    best = None
    best_size = -1
    
    # Largest first
    sol = _greedy(sorted(all_elements, reverse=True))
    if len(sol) > best_size:
        best = sol
        best_size = len(sol)
    
    if time.time() > deadline:
        return best
    
    # Smallest first
    sol = _greedy(sorted(all_elements))
    if len(sol) > best_size:
        best = sol
        best_size = len(sol)
    
    if time.time() > deadline:
        return best
    
    # Try random orderings
    random.seed(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        order = all_elements[:]
        random.shuffle(order)
        sol = _greedy(order)
        if len(sol) > best_size:
            best = sol
            best_size = len(sol)
    
    if time.time() > deadline:
        return best
    
    # Local search on best
    if best is not None:
        improved = _local_search(best, all_elements)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
    
    return best if best else [1]
