"""Construct a large 4-AP-free subset of [1, 512]."""
import time
import random
import numpy as np

N = 512

def _precompute_conflicts():
    """
    For each x in [1, N], precompute the set of 'conflict triples' (a, b, c)
    such that adding x to a set containing {a, b, c} would create a 4-AP.
    
    A 4-AP is (a, a+d, a+2d, a+3d). If x is one of these, the other three must
    be in the set.
    
    So for a given x, we need to check if there exists d and j in {0,1,2,3}
    such that x = a + j*d, and the other three terms are in the current set.
    
    Equivalently, for each x, we look at all possible (a, d) such that x is
    in the AP (a, a+d, a+2d, a+3d). For each such (a, d), the other three
    elements form a conflict triple.
    
    We store for each x a list of tuples (t1, t2, t3) where {t1,t2,t3} are
    the other three elements of a 4-AP containing x.
    """
    conflicts = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            # 4-AP: a, a+d, a+2d, a+3d
            terms = [a, a + d, a + 2 * d, a + 3 * d]
            if terms[3] > N:
                break
            # For each element in the AP, the other three form a conflict triple
            for j in range(4):
                x = terms[j]
                others = [terms[k] for k in range(4) if k != j]
                conflicts[x].append(tuple(others))
    
    return conflicts

def _random_greedy(conflicts, rng, order=None):
    """
    Greedy construction: process elements in a given order (or random order),
    add x if it doesn't create a 4-AP with already-selected elements.
    """
    selected = set()
    if order is None:
        order = list(range(1, N + 1))
        rng.shuffle(order)
    
    for x in order:
        # Check if adding x creates a 4-AP
        ok = True
        for (a, b, c) in conflicts[x]:
            if a in selected and b in selected and c in selected:
                ok = False
                break
        if ok:
            selected.add(x)
    
    return selected

def _local_search(selected, conflicts, rng, max_iters=200):
    """
    Try to improve the set by swapping elements: remove one, add another.
    """
    current = set(selected)
    current_size = len(current)
    
    for _ in range(max_iters):
        improved = False
        # Try removing each element and adding a better one
        for rem in list(current):
            if rng.random() > 0.3:
                continue  # Skip most removals for speed
            trial = current - {rem}
            # Try adding random elements not in trial
            candidates = [x for x in range(1, N + 1) if x not in trial]
            rng.shuffle(candidates)
            for add in candidates[:50]:
                # Check if adding add to trial creates a 4-AP
                ok = True
                for (a, b, c) in conflicts[add]:
                    if a in trial and b in trial and c in trial:
                        ok = False
                        break
                if ok:
                    trial.add(add)
                    if len(trial) > current_size:
                        current = trial
                        current_size = len(current)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    conflicts = _precompute_conflicts()
    
    rng = random.Random(42)
    
    best = None
    best_size = 0
    
    # Phase 1: Random greedy with many seeds
    num_greedy = 0
    while time.time() < deadline - 20:
        selected = _random_greedy(conflicts, rng)
        size = len(selected)
        if size > best_size:
            best = selected
            best_size = size
        num_greedy += 1
    
    # Phase 2: Local search on the best solution
    if time.time() < deadline - 5 and best:
        improved = _local_search(best, conflicts, rng, max_iters=500)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Phase 3: More local search iterations if time permits
    while time.time() < deadline - 2 and best:
        improved = _local_search(best, conflicts, rng, max_iters=100)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
        else:
            break
    
    return sorted(best) if best else [1]

# Precompute at module level for speed
_conflicts = _precompute_conflicts()

def construct():
    global _conflicts
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    rng = random.Random(42)
    
    best = None
    best_size = 0
    
    # Phase 1: Random greedy with many seeds
    num_greedy = 0
    while time.time() < deadline - 20:
        selected = _random_greedy(_conflicts, rng)
        size = len(selected)
        if size > best_size:
            best = selected
            best_size = size
        num_greedy += 1
    
    # Phase 2: Local search on the best solution
    if time.time() < deadline - 5 and best:
        improved = _local_search(best, _conflicts, rng, max_iters=500)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Phase 3: More local search iterations if time permits
    while time.time() < deadline - 2 and best:
        improved = _local_search(best, _conflicts, rng, max_iters=100)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
        else:
            break
    
    return sorted(best) if best else [1]
