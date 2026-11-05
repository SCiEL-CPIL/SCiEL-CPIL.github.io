"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 256
DEADLINE = 55.0  # safety margin before 60s kill

# EVOLVE-BLOCK-START
def _check_validity(selected):
    """Check if selected is a valid Sidon set (all pairwise sums including diagonals distinct)."""
    n = len(selected)
    if n == 0:
        return True
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected):
    """Check if selected is inclusion-maximal: no missing element can be added."""
    sel_set = set(selected)
    n = len(selected)
    # Precompute all existing sums
    sums = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(selected[i] + selected[j])
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if adding x would create a collision
        # New sums involving x: x + a for all a in selected, and 2*x
        ok = True
        for a in selected:
            s = x + a
            if s in sums:
                ok = False
                break
        if ok and 2 * x in sums:
            ok = False
        if ok:
            return False
    return True

def _greedy_saturation(seed_order, time_limit):
    """Greedy saturation following a given order of elements."""
    start = time.time()
    selected = []
    sums = set()
    
    for x in seed_order:
        if time.time() - start > time_limit:
            break
        # Check if x can be added
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    
    return selected

def _try_improve(selected, time_limit):
    """Try to reduce the size of the maximal Sidon set via removals and swaps."""
    start = time.time()
    current = list(selected)
    
    while time.time() - start < time_limit:
        improved = False
        
        # Try removing each element
        for i in range(len(current)):
            if time.time() - start > time_limit:
                return current
            candidate = current[:i] + current[i+1:]
            if not _check_validity(candidate):
                continue
            if not _is_maximal(candidate):
                continue
            # Check if it's still maximal
            if _is_maximal(candidate):
                current = candidate
                improved = True
                break
        
        if not improved:
            break
    
    return current

def _greedy_with_pruning(order, time_limit):
    """Greedy saturation with early pruning for maximality."""
    start = time.time()
    selected = []
    sums = set()
    
    for x in order:
        if time.time() - start > time_limit:
            break
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    
    return selected

def _random_order(rng):
    """Generate a random ordering of [1, N]."""
    order = list(range(1, N + 1))
    rng.shuffle(order)
    return order

def _prioritized_order(rng, bias_low=0.7):
    """Generate an ordering biased toward lower values."""
    order = list(range(1, N + 1))
    # Mix: some low-first, some random
    for i in range(len(order) - 1):
        j = min(len(order) - 1, int(i * bias_low + rng.random() * (1 - bias_low) * len(order)))
        order[i], order[j] = order[j], order[i]
    return order

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = random.Random(42)
    
    best = None
    best_size = N + 1
    
    # Phase 1: Multiple random restarts with different orderings
    while time.time() < deadline - 10:
        order = _random_order(rng)
        candidate = _greedy_saturation(order, time_limit=0.5)
        
        if candidate and len(candidate) < best_size:
            best = candidate
            best_size = len(candidate)
    
    # Phase 2: Try different bias strategies
    for bias in [0.3, 0.5, 0.7, 0.9]:
        if time.time() > deadline - 10:
            break
        order = _prioritized_order(rng, bias_low=bias)
        candidate = _greedy_saturation(order, time_limit=0.5)
        
        if candidate and len(candidate) < best_size:
            best = candidate
            best_size = len(candidate)
    
    # Phase 3: Local improvement on best candidates
    if best is not None and time.time() < deadline - 5:
        improved = _try_improve(best, time_limit=deadline - time.time() - 2)
        if len(improved) < best_size:
            best = improved
            best_size = len(best)
    
    # Phase 4: More restarts if time remains
    while time.time() < deadline - 3:
        order = _random_order(rng)
        candidate = _greedy_saturation(order, time_limit=0.3)
        
        if candidate and len(candidate) < best_size:
            best = candidate
            best_size = len(candidate)
            
            # Quick improvement attempt
            if time.time() < deadline - 3:
                improved = _try_improve(candidate, time_limit=1.0)
                if len(improved) < best_size:
                    best = improved
                    best_size = len(best)
    
    # Final: ensure we have a valid, maximal set
    if best is None:
        # Fallback: simple ascending greedy
        selected = []
        sums = set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best = selected
    
    # Verify final result
    if not _check_validity(best):
        # Should not happen, but fallback
        selected = []
        sums = set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best = selected
    
    if not _is_maximal(best):
        # Add elements until maximal
        sel_set = set(best)
        sums = set()
        for i in range(len(best)):
            for j in range(i, len(best)):
                sums.add(best[i] + best[j])
        
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            added = [x + a for a in best] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                best.append(x)
                sel_set.add(x)
                sums.update(added)
    
    return best
# EVOLVE-BLOCK-END
