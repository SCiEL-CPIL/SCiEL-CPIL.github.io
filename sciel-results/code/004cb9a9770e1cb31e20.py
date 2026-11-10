import time
import random

N = 256

def _greedy_with_order(order):
    selected = []
    sums = set()
    for x in order:
        if 2 * x in sums:
            continue
        blocked = False
        for a in selected:
            if (x + a) in sums:
                blocked = True
                break
        if not blocked:
            selected.append(x)
            for a in selected[:-1]:
                sums.add(x + a)
            sums.add(2 * x)
    return selected

def _verify(selected):
    if not selected:
        return False
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if 2 * x in sums:
            continue
        blocked = False
        for a in selected:
            if (x + a) in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_size = float('inf')
    best_set = None
    
    # Baseline
    order = list(range(1, N + 1))
    candidate = _greedy_with_order(order)
    if _verify(candidate):
        best_size = len(candidate)
        best_set = candidate

    # Randomized restarts
    while time.time() < deadline:
        order = list(range(1, N + 1))
        random.shuffle(order)
        candidate = _greedy_with_order(order)
        
        # Quick size check
        if len(candidate) >= best_size:
            continue
            
        if _verify(candidate):
            best_size = len(candidate)
            best_set = candidate
            if best_size <= 10:
                break

    return best_set
