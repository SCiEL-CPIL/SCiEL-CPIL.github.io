import time
import random

N = 256

def _compute_sums(selected):
    """Compute all pairwise sums a+b for a<=b in selected, return set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        sums.add(2 * a)
        for j in range(i + 1, n):
            sums.add(a + selected[j])
    return sums

def _is_sidon(selected):
    """Check if selected is a Sidon set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        if 2 * a in sums:
            return False
        sums.add(2 * a)
        for j in range(i + 1, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, sums):
    """Check if selected is maximal. sums is the set of all pairwise sums of selected."""
    selected_set = set(selected)
    for m in range(1, N + 1):
        if m in selected_set:
            continue
        # Check if adding m creates a collision
        if 2 * m in sums:
            continue
        collision = False
        for a in selected:
            if m + a in sums:
                collision = True
                break
        if not collision:
            return False
    return True

def _greedy_ascending():
    """Greedy: pick smallest possible element."""
    selected = []
    sums = set()
    for x in range(1, N + 1):
        if 2 * x in sums:
            continue
        ok = True
        for a in selected:
            if x + a in sums:
                ok = False
                break
        if ok:
            selected.append(x)
            sums.add(2 * x)
            for a in selected[:-1]:
                sums.add(x + a)
    return selected

def _greedy_descending():
    """Greedy: pick largest possible element first."""
    selected = []
    sums = set()
    for x in range(N, 0, -1):
        if 2 * x in sums:
            continue
        ok = True
        for a in selected:
            if x + a in sums:
                ok = False
                break
        if ok:
            selected.append(x)
            sums.add(2 * x)
            for a in selected[:-1]:
                sums.add(x + a)
    return selected

def _greedy_random(deadline, rng):
    """Generate a Sidon set via randomized greedy."""
    elements = list(range(1, N + 1))
    rng.shuffle(elements)
    selected = []
    sums = set()
    for x in elements:
        if time.time() > deadline:
            break
        if 2 * x in sums:
            continue
        ok = True
        for a in selected:
            if x + a in sums:
                ok = False
                break
        if ok:
            selected.append(x)
            sums.add(2 * x)
            for a in selected[:-1]:
                sums.add(x + a)
    return selected, sums

def _local_search_reduce(start_set, deadline, rng):
    """Try to reduce cardinality by removing elements that preserve maximality."""
    current = list(start_set)
    current_sums = _compute_sums(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(current)))
        rng.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                break
            candidate = current[:idx] + current[idx+1:]
            if not candidate:
                continue
            cand_sums = _compute_sums(candidate)
            if _is_maximal(candidate, cand_sums):
                current = candidate
                current_sums = cand_sums
                improved = True
                break
    return current

def _local_search_swap(start_set, deadline, rng):
    """Try to improve by swapping: remove one, add another, check Sidon + maximal."""
    current = list(start_set)
    current_size = len(current)
    
    improved = True
    iterations = 0
    while improved and time.time() < deadline and iterations < 100:
        improved = False
        iterations += 1
        current_set = set(current)
        remove_candidates = list(current)
        rng.shuffle(remove_candidates)
        add_candidates = [x for x in range(1, N+1) if x not in current_set]
        rng.shuffle(add_candidates)
        
        for r in remove_candidates:
            if time.time() > deadline:
                break
            for a in add_candidates:
                if time.time() > deadline:
                    break
                candidate = [x for x in current if x != r] + [a]
                candidate = sorted(set(candidate))
                if len(candidate) != current_size:
                    continue
                if not _is_sidon(candidate):
                    continue
                cand_sums = _compute_sums(candidate)
                if _is_maximal(candidate, cand_sums):
                    current = candidate
                    improved = True
                    break
            if improved:
                break
    return current

def _combined_local_search(start_set, deadline, rng):
    """Alternate between reduction and swap to find smaller maximal sets."""
    current = list(start_set)
    
    # Phase 1: Reduce
    current = _local_search_reduce(current, deadline, rng)
    
    # Phase 2: Swap to enable further reductions
    if time.time() < deadline:
        current = _local_search_swap(current, deadline, rng)
    
    # Phase 3: Reduce again after swaps
    if time.time() < deadline:
        current = _local_search_reduce(current, deadline, rng)
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best = None
    best_size = N + 1
    
    # Generate diverse starting candidates
    candidates = []
    
    # 1. Greedy ascending
    c1 = _greedy_ascending()
    candidates.append(c1)
    
    # 2. Greedy descending
    c2 = _greedy_descending()
    candidates.append(c2)
    
    # 3. Randomized greedy with different seeds
    for seed in range(30):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        selected, sums = _greedy_random(deadline, rng)
        if selected:
            candidates.append(selected)
    
    # Process each candidate with local search
    for cand in candidates:
        if time.time() > deadline:
            break
        if not cand:
            continue
        if not _is_sidon(cand):
            continue
        
        # Local search to minimize
        rng = random.Random(hash(tuple(sorted(cand))) % (2**31))
        minimized = _combined_local_search(cand, deadline, rng)
        
        if len(minimized) < best_size:
            best_size = len(minimized)
            best = list(minimized)
    
    if best is None:
        # Fallback: ascending greedy
        return _greedy_ascending()
    
    return best
