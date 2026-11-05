import time
import itertools
import random

def _can_add(x, selected):
    """Check if x can be added to selected without violating the constraint."""
    n = len(selected)
    if n >= 3:
        # x must not divide sum of any 3 distinct elements in selected
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if (selected[i] + selected[j] + selected[k]) % x == 0:
                        return False
    
    # For each a in selected, a must not divide x + u + v for any u, v in selected \ {a}
    for a_idx, a in enumerate(selected):
        # Collect indices of selected elements that are not a
        others = [selected[i] for i in range(n) if i != a_idx]
        m = len(others)
        for i in range(m):
            for j in range(i+1, m):
                if (x + others[i] + others[j]) % a == 0:
                    return False
    return True

def _greedy(order, deadline):
    """Run greedy in the given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """Try to improve the set by swapping or adding elements."""
    selected = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding any element not in selected
        in_set = set(selected)
        for x in range(192, 0, -1):
            if time.time() > deadline:
                return selected
            if x not in in_set and _can_add(x, selected):
                selected.append(x)
                in_set.add(x)
                improved = True
                break
        # Try swapping: remove one, add another
        if not improved:
            for i in range(len(selected)):
                if time.time() > deadline:
                    return selected
                candidate = selected[i]
                selected_without = selected[:i] + selected[i+1:]
                for x in range(192, 0, -1):
                    if x in in_set:
                        continue
                    if _can_add(x, selected_without):
                        new_set = selected_without + [x]
                        # Check if this is actually an improvement (same size but maybe enables future adds)
                        # For now, just swap if it doesn't reduce size
                        selected = new_set
                        in_set = set(selected)
                        improved = True
                        break
                if improved:
                    break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    best = []
    
    # Strategy 1: Greedy descending
    t1 = time.time()
    result = _greedy(range(192, 0, -1), deadline)
    if len(result) > len(best):
        best = result
    
    # Strategy 2: Greedy ascending
    if time.time() < deadline - 5:
        result = _greedy(range(1, 193), deadline)
        if len(result) > len(best):
            best = result
    
    # Strategy 3: Random shuffles
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        result = _greedy(order, deadline)
        if len(result) > len(best):
            best = result
    
    # Local search on best
    if time.time() < deadline - 10:
        best = _local_search(best, deadline)
    
    return best
