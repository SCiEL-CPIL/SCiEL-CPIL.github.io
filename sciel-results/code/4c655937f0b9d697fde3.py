"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def _is_valid_sidon(selected):
    """Check if a set is a valid Sidon set (all pair sums distinct)."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, n=N):
    """Check if the Sidon set is maximal (cannot add any missing element)."""
    selected_set = set(selected)
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    
    for x in range(1, n + 1):
        if x in selected_set:
            continue
        # Check if adding x would create a duplicate sum
        can_add = True
        for a in selected:
            s = x + a
            if s in sums:
                can_add = False
                break
        if can_add:
            s = 2 * x
            if s in sums:
                can_add = False
        if can_add:
            return False  # Not maximal
    return True

def _greedy_maximal_sidon(order, n=N):
    """Greedy construction given an order of elements."""
    selected = []
    sums = set()
    for x in order:
        if x < 1 or x > n:
            continue
        # Check if x can be added
        conflict = False
        for a in selected:
            s = x + a
            if s in sums:
                conflict = True
                break
        if not conflict:
            s = 2 * x
            if s in sums:
                conflict = True
        if not conflict:
            selected.append(x)
            # Add all new sums
            for a in selected:
                sums.add(x + a)
            sums.add(2 * x)
    return selected

def _try_removal(selected, n=N):
    """Try to remove elements while maintaining maximality."""
    selected = list(selected)
    improved = True
    while improved:
        improved = False
        for i in range(len(selected)):
            candidate = selected[:i] + selected[i+1:]
            if _is_maximal(candidate, n):
                selected = candidate
                improved = True
                break
    return selected

def _local_search(selected, n=N, max_iters=200):
    """Local search: try swapping elements to reduce size while maintaining maximality."""
    selected = list(selected)
    best = selected[:]
    best_len = len(selected)
    
    for _ in range(max_iters):
        improved = False
        for i in range(len(best)):
            # Try removing element i
            candidate = best[:i] + best[i+1:]
            if len(candidate) < best_len and _is_maximal(candidate, n):
                best = candidate
                best_len = len(best)
                improved = True
                break
        if not improved:
            # Try adding an element that allows a later removal
            selected_set = set(best)
            for x in range(1, n + 1):
                if x in selected_set:
                    continue
                candidate = best + [x]
                if _is_valid_sidon(candidate):
                    # Now try to remove something
                    for j in range(len(candidate)):
                        cand2 = candidate[:j] + candidate[j+1:]
                        if len(cand2) < best_len and _is_maximal(cand2, n):
                            best = cand2
                            best_len = len(cand2)
                            improved = True
                            break
                if improved:
                    break
        if not improved:
            break
    return best

def construct():
    """Construct a small inclusion-maximal Sidon set in [1,256]."""
    start_time = time.time()
    deadline = start_time + 55.0  # 55 second safety margin
    
    best = None
    best_len = float('inf')
    
    # Strategy 1: Multiple random orderings with greedy construction
    random.seed(42)
    
    # Try several different orderings
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Different ordering strategies
        if attempt < 50:
            # Random order
            order = list(range(1, N + 1))
            random.shuffle(order)
        elif attempt < 100:
            # Descending order with some randomization
            order = list(range(N, 0, -1))
            random.shuffle(order[:N//2])
        elif attempt < 150:
            # Grouped by residue classes
            order = []
            for r in range(N):
                for x in range(r + 1, N + 1, N):
                    order.append(x)
            random.shuffle(order)
        else:
            # Biased towards larger numbers first
            order = list(range(1, N + 1))
            weights = [1.0 / (x + 1) for x in order]
            # Sample without replacement using weights
            remaining = list(range(1, N + 1))
            order = []
            for _ in range(N):
                total_w = sum(weights[i-1] for i in remaining)
                r = random.random() * total_w
                cum = 0
                for i in remaining:
                    cum += weights[i-1]
                    if r <= cum:
                        order.append(i)
                        remaining.remove(i)
                        break
    
    selected = _greedy_maximal_sidon(order, N)
    
    if len(selected) < best_len:
        best_len = len(selected)
        best = selected[:]
    
    # Try to improve via removal
    if time.time() < deadline - 10:
        improved = _try_removal(best, N)
        if len(improved) < best_len:
            best_len = len(improved)
            best = improved
    
    # Try local search on the best found
    if time.time() < deadline - 5 and best is not None:
        improved = _local_search(best, N, max_iters=50)
        if len(improved) < best_len:
            best_len = len(improved)
            best = improved
    
    # Final attempt: try more aggressive local search with time check
    if time.time() < deadline - 2 and best is not None:
        # Try swapping in different elements
        selected_set = set(best)
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if x in selected_set:
                continue
            # Try adding x and removing one existing element
            for i in range(len(best)):
                candidate = best[:i] + best[i+1:] + [x]
                if _is_valid_sidon(candidate) and _is_maximal(candidate, N):
                    if len(candidate) < best_len:
                        best_len = len(candidate)
                        best = candidate
                        break
    
    return best if best is not None else [1, 2]

# EVOLVE-BLOCK-END
