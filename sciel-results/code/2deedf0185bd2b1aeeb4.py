"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256
DEADLINE = 55.0  # safety margin before 60s kill

# EVOLVE-BLOCK-START
def _is_sidon(selected):
    """Check if selected is a valid Sidon set."""
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _maximality_violations(selected, sums):
    """Return set of elements in [1,N] that can be added without breaking Sidon property.
    If empty, the set is inclusion-maximal."""
    if not _is_sidon(selected):
        return set(range(1, N + 1))
    violators = set()
    for x in range(1, N + 1):
        if x in selected:
            continue
        # Check if adding x breaks Sidon
        ok = True
        for a in selected:
            s = x + a
            if s in sums:
                ok = False
                break
        if ok:
            s2 = 2 * x
            if s2 in sums:
                ok = False
        if ok:
            violators.add(x)
    return violators

def _greedy_ascending():
    """Standard ascending greedy saturation."""
    selected, sums = [], set()
    for x in range(1, N + 1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _greedy_descending():
    """Descending greedy saturation."""
    selected, sums = [], set()
    for x in range(N, 0, -1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _greedy_random_order():
    """Greedy with random ordering."""
    order = list(range(1, N + 1))
    random.shuffle(order)
    selected, sums = [], set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _greedy_weighted(priority_fn):
    """Greedy where at each step we pick the best available element by priority."""
    available = set(range(1, N + 1))
    selected, sums = [], set()
    while available:
        best_x = None
        best_score = -1
        for x in sorted(available):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                score = priority_fn(x, len(selected), len(available))
                if score > best_score:
                    best_score = score
                    best_x = x
        if best_x is None:
            break
        selected.append(best_x)
        sums.update([best_x + a for a in selected[:-1]] + [2 * best_x])
        available.discard(best_x)
    return selected

def _local_search(selected):
    """Try to reduce size by removing elements and re-saturating."""
    if not _is_sidon(selected):
        return selected
    # Check maximality
    violators = _maximality_violations(selected, set())
    if len(violators) == 0:
        return selected  # already maximal
    
    # Try removing each element and re-saturating
    best = selected
    best_len = len(selected)
    
    for i in range(len(selected)):
        candidate = selected[:i] + selected[i+1:]
        # Re-saturate: add back elements that can be added
        sums = set()
        for a in candidate:
            for b in candidate:
                if a <= b:
                    sums.add(a + b)
        
        # Try to add all missing elements in greedy fashion
        for x in range(1, N + 1):
            if x in candidate:
                continue
            added = [x + a for a in candidate] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                candidate.append(x)
                sums.update(added)
        
        if len(candidate) < best_len:
            # Check if maximal
            viol = _maximality_violations(candidate, set())
            if len(viol) == 0:
                best = candidate
                best_len = len(candidate)
    
    return best

def construct():
    start_time = time.time()
    
    # Phase 1: Try simple greedy variants
    candidates = []
    
    # Ascending greedy
    c1 = _greedy_ascending()
    if len(c1) < N:
        candidates.append(c1)
    
    # Descending greedy
    c2 = _greedy_descending()
    if len(c2) < N:
        candidates.append(c2)
    
    # Phase 2: Random order greedy with multiple restarts
    best_size = min(len(c) for c in candidates) if candidates else N
    best_set = candidates[0] if candidates else []
    
    deadline = start_time + DEADLINE
    
    # Do random greedy restarts
    attempts = 0
    while time.time() < deadline - 5.0:
        c = _greedy_random_order()
        attempts += 1
        if len(c) < best_size:
            best_size = len(c)
            best_set = c
        # Occasionally do local search on best
        if attempts % 10 == 0 and best_set:
            improved = _local_search(best_set)
            if len(improved) < best_size:
                best_size = len(improved)
                best_set = improved
    
    # Phase 3: Try to improve best_set with more aggressive local search
    if best_set and time.time() < deadline - 2.0:
        # Try multiple removal-readd cycles
        current = list(best_set)
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            improved = _local_search(current)
            if len(improved) < len(current):
                current = improved
            else:
                break
        if len(current) < best_size:
            best_size = len(current)
            best_set = current
    
    # Final verification
    if best_set and _is_sidon(best_set):
        return best_set
    
    # Fallback to ascending greedy
    return _greedy_ascending()
# EVOLVE-BLOCK-END
