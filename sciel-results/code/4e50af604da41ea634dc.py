"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 192
MAX_K = 48

def _is_valid_set(selected):
    """Check if the selected set satisfies the constraint."""
    sel = list(selected)
    n = len(sel)
    for i, a in enumerate(sel):
        others = [v for j, v in enumerate(sel) if j != i]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _is_valid_addition(selected, x):
    """Check if adding x to selected keeps the set valid.
    We only need to check:
    1. x does not divide sum of any 3 distinct others in selected
    2. No element in selected divides sum of any triple that includes x and 2 others from selected
    """
    sel = list(selected)
    # Check 1: x as the divisor
    for t in combinations(sel, 3):
        if sum(t) % x == 0:
            return False
    # Check 2: each element a in sel as divisor, with triples that include x
    for a in sel:
        for pair in combinations(sel, 2):
            s = x + pair[0] + pair[1]
            if s % a == 0:
                return False
    return True

def _greedy(order):
    """Greedy construction with given order of candidates."""
    selected = []
    for x in order:
        if len(selected) >= MAX_K:
            break
        if _is_valid_addition(selected, x):
            selected.append(x)
    return selected

def _local_search(selected, max_iters=50000, deadline=None):
    """Local search: try to add elements or swap to improve cardinality."""
    selected = list(selected)
    best = list(selected)
    best_len = len(selected)
    rng = random.Random(42)
    
    # Precompute which candidates are not in selected
    in_set = set(selected)
    candidates = [x for x in range(1, N + 1) if x not in in_set]
    
    for iteration in range(max_iters):
        if deadline and time.time() > deadline:
            break
        if best_len >= MAX_K:
            break
        
        # Try to add a random candidate
        if candidates:
            idx = rng.randrange(len(candidates))
            x = candidates[idx]
            if _is_valid_addition(selected, x):
                selected.append(x)
                in_set.add(x)
                candidates.pop(idx)
                if len(selected) > best_len:
                    best_len = len(selected)
                    best = list(selected)
        
        # Try swaps: remove one element, try to add a better one
        if selected:
            # Try removing a random element and adding a random candidate
            if candidates and rng.random() < 0.3:
                rem_idx = rng.randrange(len(selected))
                rem = selected[rem_idx]
                # Try adding a random candidate after removing rem
                temp_selected = selected[:rem_idx] + selected[rem_idx+1:]
                # Try a few candidates
                tried = 0
                added = False
                while tried < 10 and not added:
                    c_idx = rng.randrange(len(candidates))
                    x = candidates[c_idx]
                    if _is_valid_addition(temp_selected, x):
                        temp_selected.append(x)
                        # Check if this improves (it maintains same size, but might allow future additions)
                        # For now, accept if valid
                        selected = temp_selected
                        in_set.discard(rem)
                        in_set.add(x)
                        candidates.pop(c_idx)
                        candidates.append(rem)
                        added = True
                    tried += 1
    
    return best

def construct():
    deadline = time.time() + 80  # Safety margin before 90s limit
    
    # Multiple greedy strategies with different orderings
    best_set = []
    best_len = 0
    
    strategies = [
        list(range(N, 0, -1)),           # Descending
        list(range(1, N + 1)),           # Ascending
        list(range(N, 0, -1, 2)),        # Descending odds
        list(range(N, 1, -2)),           # Descending evens
        [x for x in range(N, 0, -1) if x > N // 2],  # Upper half
        [x for x in range(N, 0, -1) if x > N // 3],  # Upper 2/3
    ]
    
    # Add some randomized orderings
    rng = random.Random(123)
    for _ in range(10):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        strategies.append(order)
    
    for order in strategies:
        if time.time() > deadline:
            break
        s = _greedy(order)
        if len(s) > best_len:
            best_len = len(s)
            best_set = list(s)
    
    # Local search from the best greedy result
    if best_set and time.time() < deadline:
        improved = _local_search(best_set, max_iters=200000, deadline=deadline)
        if len(improved) > best_len:
            best_len = len(improved)
            best_set = improved
    
    # Try local search from other good greedy results
    for order in strategies[:6]:
        if time.time() > deadline - 5:
            break
        s = _greedy(order)
        if len(s) >= best_len - 2:  # Only try if close to best
            improved = _local_search(s, max_iters=100000, deadline=deadline)
            if len(improved) > best_len:
                best_len = len(improved)
                best_set = improved
    
    return best_set
# EVOLVE-BLOCK-END
