"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import random
import time

def _compute_sums(selected):
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            sums.add(a + selected[j])
    return sums

def _check_sidon(selected):
    if not selected:
        return False
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal_fast(selected, N):
    """Fast maximality check using precomputed conflict set."""
    sel_set = set(selected)
    sums = _compute_sums(selected)
    # conflict[x] = True if x cannot be added (i.e., x is "blocked")
    # x is blocked if exists a in selected: x+a in sums, OR 2x in sums
    blocked = [False] * (N + 1)
    
    # Mark x blocked if 2x in sums (diagonal conflict)
    for s in sums:
        if s % 2 == 0:
            x = s // 2
            if 1 <= x <= N:
                blocked[x] = True
                
    # Mark x blocked if x+a in sums (off-diagonal conflict)
    for a in selected:
        for s in sums:
            x = s - a
            if 1 <= x <= N:
                blocked[x] = True
                
    # Also: for each x not in selected, if not blocked[x], not maximal
    for x in range(1, N + 1):
        if x not in sel_set and not blocked[x]:
            return False
    return True

def _greedy_build(order, N):
    selected = []
    sums = set()
    for x in order:
        added_sums = set()
        can_add = True
        for a in selected:
            s = x + a
            if s in sums or s in added_sums:
                can_add = False
                break
            added_sums.add(s)
        if can_add:
            s = 2 * x
            if s in sums or s in added_sums:
                can_add = False
            else:
                added_sums.add(s)
        if can_add:
            selected.append(x)
            sums.update(added_sums)
    return selected

def _greedy_extend(base, N):
    ext = list(base)
    ext_set = set(ext)
    ext_sums = _compute_sums(ext)
    for x in range(1, N + 1):
        if x in ext_set:
            continue
        added_sums = set()
        can_add = True
        for a in ext:
            s = x + a
            if s in ext_sums or s in added_sums:
                can_add = False
                break
            added_sums.add(s)
        if can_add:
            s = 2 * x
            if s in ext_sums or s in added_sums:
                can_add = False
            else:
                added_sums.add(s)
        if can_add:
            ext.append(x)
            ext_set.add(x)
            ext_sums.update(added_sums)
    return ext

def _try_remove_and_extend(selected, N, deadline):
    n = len(selected)
    if n <= 1:
        return None
    best = None
    best_len = n + 1
    for i in range(n):
        if time.time() > deadline:
            break
        reduced = [selected[j] for j in range(n) if j != i]
        if not reduced:
            continue
        if not _check_sidon(reduced):
            continue
        ext = _greedy_extend(reduced, N)
        if _check_sidon(ext) and _is_maximal_fast(ext, N):
            if len(ext) < best_len:
                best_len = len(ext)
                best = list(ext)
    return best

def _try_swap(selected, N, deadline):
    n = len(selected)
    if n <= 2:
        return None
    best = None
    best_len = n + 1
    selected_set = set(selected)
    for i in range(n):
        if time.time() > deadline:
            break
        removed = selected[i]
        reduced = [selected[j] for j in range(n) if j != i]
        if not _check_sidon(reduced):
            continue
        reduced_set = set(reduced)
        reduced_sums = _compute_sums(reduced)
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if x in reduced_set or x == removed:
                continue
            can_add = True
            for a in reduced:
                if x + a in reduced_sums:
                    can_add = False
                    break
            if can_add and 2 * x in reduced_sums:
                can_add = False
            if not can_add:
                continue
            new_base = reduced + [x]
            new_set = _greedy_extend(new_base, N)
            if _check_sidon(new_set) and _is_maximal_fast(new_set, N):
                if len(new_set) < best_len:
                    best_len = len(new_set)
                    best = list(new_set)
    return best

def _try_double_remove(selected, N, deadline):
    n = len(selected)
    if n <= 3:
        return None
    best = None
    best_len = n + 1
    for i in range(n):
        for j in range(i + 1, n):
            if time.time() > deadline:
                break
            reduced = [selected[k] for k in range(n) if k != i and k != j]
            if len(reduced) < 2:
                continue
            if not _check_sidon(reduced):
                continue
            ext = _greedy_extend(reduced, N)
            if _check_sidon(ext) and _is_maximal_fast(ext, N):
                if len(ext) < best_len:
                    best_len = len(ext)
                    best = list(ext)
    return best

def _local_improve(current, N, deadline):
    """
    Local search focused on removing elements while maintaining maximality.
    Tries 1-removal and 2-removal.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try 1-removal
        n = len(current)
        for i in range(n):
            if time.time() > deadline:
                return current
            reduced = [current[j] for j in range(n) if j != i]
            if not reduced:
                continue
            # reduced is always Sidon
            if _is_maximal_fast(reduced, N):
                current = reduced
                improved = True
                break # Restart loop with new size
        
        if improved:
            continue
            
        # Try 2-removal
        n = len(current)
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return current
                reduced = [current[k] for k in range(n) if k != i and k != j]
                if len(reduced) < 2:
                    continue
                if _is_maximal_fast(reduced, N):
                    current = reduced
                    improved = True
                    break
            if improved:
                break
    return current

def _anneal(current, N, deadline, rng):
    """Simulated annealing on removal/swap moves to escape local minima."""
    cur = list(current)
    best = list(cur)
    best_size = len(cur)
    temp = 5.0
    cooling = 0.95
    min_temp = 0.3
    while time.time() < deadline and temp > min_temp:
        # Pick a random move type
        r = rng.random()
        if r < 0.5:
            new = _try_remove_and_extend(cur, N, deadline)
        elif r < 0.8:
            new = _try_swap(cur, N, deadline)
        else:
            new = _try_double_remove(cur, N, deadline)
        if new is not None:
            delta = len(new) - len(cur)
            if delta < 0:
                cur = new
                if len(cur) < best_size:
                    best = list(cur)
                    best_size = len(cur)
            elif delta == 0:
                # Accept same-size with some probability to explore
                if rng.random() < 0.3:
                    cur = new
            else:
                # Accept worsening with probability exp(-delta/temp)
                import math
                if rng.random() < math.exp(-delta / temp):
                    cur = new
        temp *= cooling
    if len(best) < len(current):
        return best
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    rng = random.Random(42)

    best_set = None
    best_size = N + 1

    # Phase 1: Diverse greedy starts with structured orderings
    # Ascending
    baseline = _greedy_build(list(range(1, N + 1)), N)
    if _check_sidon(baseline) and _is_maximal_fast(baseline, N):
        best_size = len(baseline)
        best_set = list(baseline)

    # Descending
    desc = _greedy_build(list(range(N, 0, -1)), N)
    if _check_sidon(desc) and _is_maximal_fast(desc, N):
        if len(desc) < best_size:
            best_size = len(desc)
            best_set = list(desc)

    # Spaced orderings (stride patterns)
    for stride in [2, 3, 4, 5, 7, 11]:
        if time.time() > deadline - 25.0:
            break
        order = list(range(1, N + 1, stride))
        order += list(range(stride, N + 1, stride * 2))
        # Ensure all elements present
        order_set = set(order)
        for x in range(1, N + 1):
            if x not in order_set:
                order.append(x)
        candidate = _greedy_build(order, N)
        if _check_sidon(candidate) and _is_maximal_fast(candidate, N):
            if len(candidate) < best_size:
                best_size = len(candidate)
                best_set = list(candidate)

    # Random restarts
    for attempt in range(400):
        if time.time() > deadline - 20.0:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        candidate = _greedy_build(order, N)
        if _check_sidon(candidate) and _is_maximal_fast(candidate, N):
            if len(candidate) < best_size:
                best_size = len(candidate)
                best_set = list(candidate)
                if best_size <= 14:
                    break

    if best_set is None:
        best_set = baseline
        best_size = len(best_set)

    # Phase 2: Local improvement on best (Removal-based)
    current = list(best_set)
    current = _local_improve(current, N, deadline - 5.0)
    if len(current) < best_size:
        best_size = len(current)
        best_set = list(current)
        
    # Phase 3: Try to improve other good candidates found in Phase 1
    # We don't have a list of candidates, so we just restart if time permits
    if time.time() < deadline - 2.0:
         # Perform a few more random restarts + local improvement
         for attempt in range(50):
             if time.time() > deadline - 2.0:
                 break
             order = list(range(1, N + 1))
             rng.shuffle(order)
             candidate = _greedy_build(order, N)
             if not (_check_sidon(candidate) and _is_maximal_fast(candidate, N)):
                 continue
             if len(candidate) >= best_size:
                 continue
             cur = list(candidate)
             cur = _local_improve(cur, N, deadline - 2.0)
             if len(cur) < best_size:
                 best_size = len(cur)
                 best_set = list(cur)

    # Phase 4: More random restarts with local improvement + annealing
    for attempt in range(200):
        if time.time() > deadline - 3.0:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        candidate = _greedy_build(order, N)
        if not (_check_sidon(candidate) and _is_maximal_fast(candidate, N)):
            continue
        if len(candidate) >= best_size:
            continue
        cur = list(candidate)
        cur = _local_improve(cur, N, deadline - 3.0)
        if time.time() < deadline - 3.0:
            cur = _anneal(cur, N, deadline - 3.0, rng)
        if len(cur) < best_size:
            best_size = len(cur)
            best_set = list(cur)
            if best_size <= 14:
                break

    return best_set
# EVOLVE-BLOCK-END
