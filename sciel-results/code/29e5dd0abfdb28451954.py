"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256
import time
import random

# EVOLVE-BLOCK-START
def _is_sidon(selected):
    """Check if a set is Sidon (all pairwise sums distinct)."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_sums(selected):
    """Get all pairwise sums of a Sidon set."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    return sums

def _is_maximal(selected, N):
    """Check if a Sidon set is maximal (can't add any missing element)."""
    sums = _get_sums(selected)
    selected_set = set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Check if adding x would violate Sidon property
        new_sums = [x + a for a in selected] + [2 * x]
        if len(set(new_sums)) == len(new_sums) and not sums.intersection(new_sums):
            return False
    return True

def _greedy_construct(order, N):
    """Greedy construction using a given order."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _try_remove_elements(selected, N, max_iter=100):
    """Try to remove elements while maintaining maximality."""
    selected = list(selected)
    for _ in range(max_iter):
        removed_any = False
        for i in range(len(selected)):
            candidate = selected[:i] + selected[i+1:]
            if _is_sidon(candidate) and _is_maximal(candidate, N):
                selected = candidate
                removed_any = True
                break
        if not removed_any:
            break
    return selected

def construct():
    deadline = time.time() + 55  # 55 second deadline with safety margin
    
    best = None
    best_size = float('inf')
    
    # Try ascending order first (baseline)
    asc_order = list(range(1, N + 1))
    result = _greedy_construct(asc_order, N)
    if len(result) < best_size:
        best = result
        best_size = len(result)
    
    # Try multiple random orderings
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        result = _greedy_construct(order, N)
        if len(result) < best_size:
            best = result
            best_size = len(result)
            # Try to further minimize by removing elements
            improved = _try_remove_elements(result, N, max_iter=20)
            if len(improved) < best_size:
                best = improved
                best_size = len(improved)
    
    # Also try descending order
    desc_order = list(range(N, 0, -1))
    result = _greedy_construct(desc_order, N)
    if len(result) < best_size:
        best = result
        best_size = len(result)
        improved = _try_remove_elements(result, N, max_iter=50)
        if len(improved) < best_size:
            best = improved
            best_size = len(improved)
    
    if best is None:
        # Fallback to ascending greedy
        best = _greedy_construct(asc_order, N)
    
    return best
# EVOLVE-BLOCK-END
