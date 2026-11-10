import time
import random

_N = 256
_DEADLINE = 90.0
_MARGIN = 5.0

# Precompute reciprocals
_RECIP = [0.0] * (_N + 1)
for i in range(1, _N + 1):
    _RECIP[i] = 1.0 / i

def _init_block_counts(S):
    """Calculate block counts for a given set S."""
    counts = [0] * (_N + 1)
    S_list = list(S)
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            # a, b in S. They block:
            # 1. c = 2*b - a
            c1 = 2 * b - a
            if 1 <= c1 <= _N and c1 != a and c1 != b:
                counts[c1] += 1
            # 2. c = 2*a - b
            c2 = 2 * a - b
            if 1 <= c2 <= _N and c2 != a and c2 != b:
                counts[c2] += 1
            # 3. c = (a + b) / 2
            if (a + b) % 2 == 0:
                c3 = (a + b) // 2
                if 1 <= c3 <= _N and c3 != a and c3 != b:
                    counts[c3] += 1
    return counts

def _update_block_add(counts, S, x):
    """Update block counts when adding x to S."""
    for y in S:
        # x, y in S. They block:
        # 1. c = 2*y - x
        c1 = 2 * y - x
        if 1 <= c1 <= _N and c1 != x and c1 != y:
            counts[c1] += 1
        # 2. c = 2*x - y
        c2 = 2 * x - y
        if 1 <= c2 <= _N and c2 != x and c2 != y:
            counts[c2] += 1
        # 3. c = (x + y) / 2
        if (x + y) % 2 == 0:
            c3 = (x + y) // 2
            if 1 <= c3 <= _N and c3 != x and c3 != y:
                counts[c3] += 1

def _update_block_remove(counts, S, x):
    """Update block counts when removing x from S."""
    for y in S:
        # x, y in S. They blocked:
        # 1. c = 2*y - x
        c1 = 2 * y - x
        if 1 <= c1 <= _N and c1 != x and c1 != y:
            counts[c1] -= 1
        # 2. c = 2*x - y
        c2 = 2 * x - y
        if 1 <= c2 <= _N and c2 != x and c2 != y:
            counts[c2] -= 1
        # 3. c = (x + y) / 2
        if (x + y) % 2 == 0:
            c3 = (x + y) // 2
            if 1 <= c3 <= _N and c3 != x and c3 != y:
                counts[c3] -= 1

def _greedy_init():
    """Greedy initialization: add smallest available valid elements."""
    S = set()
    counts = [0] * (_N + 1)
    for x in range(1, _N + 1):
        if counts[x] == 0:
            S.add(x)
            _update_block_add(counts, S, x)
    return S, counts

def _local_search(S, counts, max_iters=2000):
    """Local search with 1-swap improvements."""
    S = set(S)
    counts = list(counts)
    
    for _ in range(max_iters):
        improved = False
        
        # 1. Try to add any valid element
        # Find smallest x not in S with counts[x] == 0
        add_candidates = []
        for x in range(1, _N + 1):
            if x not in S and counts[x] == 0:
                add_candidates.append(x)
        
        if add_candidates:
            # Pick the one with largest reciprocal (smallest x)
            best_add = min(add_candidates)
            S.add(best_add)
            _update_block_add(counts, S, best_add)
            improved = True
            continue
        
        # 2. No add possible. Try 1-swap.
        # For each r in S, remove r, see if we can add x such that 1/x > 1/r
        # To be efficient, iterate r in order of decreasing 1/r (increasing r)
        # If we swap r for x, we need x < r to improve sum.
        # So we only care about x < r.
        
        swap_done = False
        S_list = list(S)
        random.shuffle(S_list)
        
        for r in S_list:
            # Temporarily remove r
            _update_block_remove(counts, S, r)
            S.discard(r)
            
            # Find best x to add
            # x must be < r to improve sum (1/x > 1/r)
            # And counts[x] must be 0
            best_x = None
            for x in range(1, r):
                if x not in S and counts[x] == 0:
                    best_x = x
                    break
            
            if best_x is not None:
                # Perform swap
                S.add(best_x)
                _update_block_add(counts, S, best_x)
                improved = True
                swap_done = True
                break
            else:
                # Restore r
                S.add(r)
                _update_block_add(counts, S, r)
        
        if not improved:
            break
            
    return S, counts

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - _MARGIN
    
    best_S = set()
    best_sum = 0.0
    
    # Initial greedy
    S, counts = _greedy_init()
    best_S = set(S)
    best_sum = sum(_RECIP[x] for x in S)
    
    # Local search from greedy
    S, counts = _local_search(S, counts)
    curr_sum = sum(_RECIP[x] for x in S)
    if curr_sum > best_sum:
        best_S = set(S)
        best_sum = curr_sum
    
    # Multi-start with random perturbations
    random.seed(42)
    
    while time.time() < deadline:
        # Perturb best_S
        S = set(best_S)
        
        # Randomly remove some elements
        n_remove = random.randint(1, 5)
        to_remove = random.sample(list(S), min(n_remove, len(S)))
        for x in to_remove:
            S.discard(x)
        
        # Recalculate counts for perturbed S
        counts = _init_block_counts(S)
        
        # Run local search
        S, counts = _local_search(S, counts)
        
        curr_sum = sum(_RECIP[x] for x in S)
        if curr_sum > best_sum:
            best_S = set(S)
            best_sum = curr_sum
    
    return list(best_S)
