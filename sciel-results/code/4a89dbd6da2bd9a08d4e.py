"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _subset_sum_200_exists(selected):
    """Check if any subset of selected sums to 200 using DP."""
    reachable = [False] * 201
    reachable[0] = True
    for x in selected:
        if x > 200:
            continue
        for s in range(200 - x, -1, -1):
            if reachable[s]:
                reachable[s + x] = True
    return reachable[200]

def _greedy(order):
    """Greedy construction with given order."""
    selected = []
    reachable = [False] * 201
    reachable[0] = True
    for x in order:
        if x > 200:
            continue
        # Check if adding x would create sum 200
        bad = False
        for s in range(200 - x, -1, -1):
            if reachable[s] and s + x == 200:
                bad = True
                break
        if not bad:
            selected.append(x)
            for s in range(200 - x, -1, -1):
                if reachable[s]:
                    reachable[s + x] = True
    return selected

def _try_add(selected, x):
    """Check if x can be added to selected without creating sum 200."""
    # Quick check: if x == 200, no
    if x == 200:
        return False
    # Check if there's a subset of selected summing to 200-x
    target = 200 - x
    if target < 0:
        return False
    # DP for subset sum of selected up to target
    if target > 200:
        target = 200
    reachable = [False] * (target + 1)
    reachable[0] = True
    for y in selected:
        if y > target:
            continue
        for s in range(target - y, -1, -1):
            if reachable[s]:
                reachable[s + y] = True
    return not reachable[target]

def _local_search(selected, time_limit=5.0):
    """Local search to improve cardinality."""
    start = time.time()
    best = list(selected)
    best_set = set(best)
    
    while time.time() - start < time_limit:
        improved = False
        # Try adding elements
        for x in range(1, 101):
            if x in best_set:
                continue
            if _try_add(best, x):
                best.append(x)
                best_set.add(x)
                improved = True
                break
        if improved:
            continue
        
        # Try swaps: remove one, add another
        for i in range(len(best)):
            for x in range(1, 101):
                if x in best_set:
                    continue
                trial = [best[j] for j in range(len(best)) if j != i] + [x]
                if not _subset_sum_200_exists(trial):
                    best = trial
                    best_set = set(best)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    random.seed(12345)
    best_result = []
    
    # Multiple greedy strategies
    strategies = []
    strategies.append(list(range(100, 0, -1)))  # Descending
    strategies.append(list(range(1, 101)))  # Ascending
    strategies.append(list(range(1, 101, 2)) + list(range(2, 101, 2)))  # Odd then even
    strategies.append(list(range(2, 101, 2)) + list(range(1, 101, 2)))  # Even then odd
    strategies.append(list(range(100, 0, -2)) + list(range(99, 0, -2)))  # Large odd then large even
    
    # Random orderings
    for _ in range(30):
        order = list(range(1, 101))
        random.shuffle(order)
        strategies.append(order)
    
    for order in strategies:
        if time.time() >= deadline:
            break
        result = _greedy(order)
        if len(result) > len(best_result):
            best_result = result
    
    # Local search on best result
    if time.time() < deadline - 10:
        improved = _local_search(best_result, time_limit=min(10.0, deadline - time.time() - 5))
        if len(improved) > len(best_result):
            best_result = improved
    
    # Ensure nonempty
    if not best_result:
        best_result = [1]
    
    return best_result
# EVOLVE-BLOCK-END
