"""Construct a large 4-term-AP-free subset of [1,512]."""
import time
import random
import numpy as np

def _has_4ap(S_set):
    """Check if set contains a nonconstant 4-term AP."""
    S = S_set
    for a in S:
        for b in S:
            if b <= a:
                continue
            d = b - a
            if d == 0:
                continue
            if (a + 3*d) in S:
                # a, a+d, a+2d, a+3d
                if (a + d) in S and (a + 2*d) in S:
                    return True
    return False

def _count_4aps(S_set):
    """Count number of 4-term APs in the set."""
    S = S_set
    count = 0
    for a in S:
        for b in S:
            if b <= a:
                continue
            d = b - a
            if d == 0:
                continue
            if (a + 3*d) in S:
                if (a + d) in S and (a + 2*d) in S:
                    count += 1
    return count

def _greedy_construction(n=512, seed=0):
    """Greedy construction: add elements in order if no 4-AP is formed."""
    rng = random.Random(seed)
    order = list(range(1, n+1))
    rng.shuffle(order)
    
    selected = set()
    for x in order:
        # Check if adding x creates a 4-AP
        creates_ap = False
        # x could be the 4th term: a, a+d, a+2d, x where x = a+3d
        # So a = x - 3d, and we need a+d, a+2d in selected
        for d in range(1, (x-1)//3 + 1):
            a = x - 3*d
            if a < 1:
                break
            if a in selected and (a+d) in selected and (a+2*d) in selected:
                creates_ap = True
                break
        # x could be the 3rd term: a, a+d, x, a+3d where x = a+2d
        # So a = x - 2d, and we need a+d in selected and a+3d in selected
        # But a+3d = x + d, which may not be in selected yet
        for d in range(1, (x-1)//2 + 1):
            a = x - 2*d
            if a < 1:
                break
            if a in selected and (a+d) in selected and (a+3*d) in selected:
                creates_ap = True
                break
        # x could be the 2nd term: a, x, x+d, x+2d where x = a+d
        # So a = x-d, and we need x+d in selected and x+2d in selected
        for d in range(1, x + 1):
            a = x - d
            if a < 1:
                break
            if a in selected and (x+d) in selected and (x+2*d) in selected:
                creates_ap = True
                break
        # x could be the 1st term: x, x+d, x+2d, x+3d
        for d in range(1, (n - x)//3 + 1):
            if (x+d) in selected and (x+2*d) in selected and (x+3*d) in selected:
                creates_ap = True
                break
        
        if not creates_ap:
            selected.add(x)
    
    return selected

def _local_search(S, max_iter=5000, deadline=None):
    """Improve set S via local search with swaps."""
    S = set(S)
    n = 512
    all_elements = set(range(1, n+1))
    
    def is_valid(S_set):
        return _has_4ap(S_set) == False
    
    current_size = len(S)
    best_S = set(S)
    best_size = current_size
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try to add a random element not in S
        candidates = [x for x in all_elements if x not in S]
        if candidates:
            x = random.choice(candidates)
            new_S = S | {x}
            if not _has_4ap(new_S):
                S = new_S
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
        
        # Try swap: remove one element, add another
        if random.random() < 0.5:
            in_S = list(S)
            if in_S:
                x_remove = random.choice(in_S)
                candidates = [x for x in all_elements if x not in S]
                if candidates:
                    x_add = random.choice(candidates)
                    new_S = (S - {x_remove}) | {x_add}
                    if not _has_4ap(new_S):
                        S = new_S
                        if len(S) > best_size:
                            best_S = set(S)
                            best_size = len(S)
    
    return best_S

def construct():
    deadline = time.time() + 80  # Leave 10s safety margin
    
    best_set = None
    best_size = 0
    
    # Try multiple greedy constructions with different seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        S = _greedy_construction(512, seed)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    if best_set is None:
        # Fallback: simple greedy in natural order
        best_set = _greedy_construction(512, 0)
        best_size = len(best_set)
    
    # Local search to improve
    if time.time() < deadline - 5:
        improved = _local_search(best_set, max_iter=10000, deadline=deadline - 2)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    return sorted(best_set)
