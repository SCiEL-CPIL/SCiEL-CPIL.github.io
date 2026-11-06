import time
import random

def _greedy(order, target=200):
    """Greedy: iterate in given order, add element if it doesn't create a subset sum of target."""
    selected = []
    reachable = {0}
    for x in order:
        new_reachable = set()
        for s in reachable:
            ns = s + x
            if ns == target:
                break  # would create target sum, skip
            if ns <= target:
                new_reachable.add(ns)
        else:
            selected.append(x)
            reachable |= new_reachable
    return selected

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    best = []
    
    # Deterministic orders first
    base_orders = [
        list(range(100, 0, -1)),   # descending
        list(range(1, 101)),       # ascending
        list(range(50, 0, -1)) + list(range(51, 101)),  # high then low
        list(range(1, 51)) + list(range(100, 50, -1)),  # low then high
    ]
    
    for order in base_orders:
        if time.time() > deadline:
            break
        result = _greedy(order)
        if len(result) > len(best):
            best = result
    
    # Randomized shuffles
    best_len = len(best)
    attempts = 0
    max_attempts = 500
    
    while attempts < max_attempts and time.time() < deadline:
        order = list(range(1, 101))
        random.shuffle(order)
        result = _greedy(order)
        if len(result) > best_len:
            best = result
            best_len = len(result)
        attempts += 1
    
    # Additional targeted strategies
    if time.time() < deadline:
        # Try to add back elements that were skipped, in reverse order
        for start_order in base_orders:
            if time.time() > deadline:
                break
            result = _greedy(start_order)
            # Try to greedily add skipped elements in various orders
            skipped = [x for x in range(1, 101) if x not in set(result)]
            for skip_order in [skipped, skipped[::-1], sorted(skipped, key=lambda x: x % 7)]:
                if time.time() > deadline:
                    break
                # Try adding each skipped element to see if it still works
                current = list(result)
                reachable = {0}
                for x in current:
                    reachable |= {s + x for s in reachable if s + x <= 200}
                for x in skip_order:
                    if 200 not in {s + x for s in reachable if s + x <= 200}:
                        current.append(x)
                        reachable |= {s + x for s in reachable if s + x <= 200}
                if len(current) > best_len:
                    best = current
                    best_len = len(current)
    
    # If best is still empty (shouldn't happen), fallback
    if not best:
        best = list(range(100, 0, -1))
    
    return best
