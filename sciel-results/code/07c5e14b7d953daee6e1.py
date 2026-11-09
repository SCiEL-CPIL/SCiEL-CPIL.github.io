"""Sidon set: small inclusion-maximal subset of [1,256]."""
import time
import random

N = 256
DEADLINE = time.time() + 25.0  # safety margin before 30s kill


def _sidon_sums(selected):
    """Return the set of all unordered pair sums a+b with a<=b from selected."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    return sums


def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set (all unordered pair sums distinct)."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _is_maximal(selected):
    """Check if selected is inclusion-maximal in [1, N]."""
    sums = _sidon_sums(selected)
    selected_set = set(selected)
    for y in range(1, N + 1):
        if y in selected_set:
            continue
        # Check if y can be added
        can_add = True
        for a in selected:
            if y + a in sums:
                can_add = False
                break
        if can_add and 2 * y in sums:
            can_add = False
        if can_add:
            return False
    return True


def _greedy_maximal_sidon(order):
    """Greedy construction: pick elements in the given order if valid."""
    selected = []
    sums = set()
    selected_set = set()
    
    for x in order:
        if time.time() > DEADLINE:
            break
        if x in selected_set:
            continue
        # Check if x can be added
        can_add = True
        for a in selected:
            if x + a in sums:
                can_add = False
                break
        if can_add and 2 * x in sums:
            can_add = False
        
        if can_add:
            selected.append(x)
            selected_set.add(x)
            for a in selected[:-1]:
                sums.add(x + a)
            sums.add(2 * x)
    
    # Check maximality
    if not _is_maximal(selected):
        return None
    return selected


def _try_removal(selected, sums):
    """Try to remove elements from a maximal Sidon set to get a smaller one.
    Returns (new_selected, new_sums, removed_count) or (None, None, 0)."""
    selected_set = set(selected)
    n = len(selected)
    
    # Try removing each element in random order
    indices = list(range(n))
    random.shuffle(indices)
    
    for idx in indices:
        if time.time() > DEADLINE - 2.0:
            break
        
        x = selected[idx]
        # Try removing x
        new_selected = [selected[i] for i in range(n) if i != idx]
        new_sums = _sidon_sums(new_selected)
        
        # Check if the result is still maximal
        if _is_maximal(new_selected):
            return new_selected, new_sums, 1
    
    return None, None, 0


def _local_search_with_swaps(selected, sums):
    """Local search: try swapping elements to reduce size or maintain maximality."""
    selected_set = set(selected)
    n = len(selected)
    
    # Try removing elements and see if we can repair maximality
    indices = list(range(n))
    random.shuffle(indices)
    
    for idx in indices:
        if time.time() > DEADLINE - 2.0:
            break
        
        x = selected[idx]
        # Try removing x
        new_selected = [selected[i] for i in range(n) if i != idx]
        new_sums = _sidon_sums(new_selected)
        
        # Check if the result is still maximal
        if _is_maximal(new_selected):
            return new_selected, new_sums, True
    
    return None, None, False


def _repair_minimize(selected):
    """Repeatedly try to remove elements from a maximal Sidon set."""
    sums = _sidon_sums(selected)
    improved = True
    while improved:
        improved = False
        indices = list(range(len(selected)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > DEADLINE - 2.0:
                return selected
            
            x = selected[idx]
            new_selected = [selected[i] for i in range(len(selected)) if i != idx]
            
            if _is_maximal(new_selected):
                selected = new_selected
                improved = True
                break
    
    return selected


def _greedy_valid_sidon(order):
    """Greedy: build a valid Sidon set (not necessarily maximal)."""
    selected = []
    sums = set()
    
    for x in order:
        if time.time() > DEADLINE:
            break
        can_add = True
        for a in selected:
            if x + a in sums:
                can_add = False
                break
        if can_add and 2 * x in sums:
            can_add = False
        
        if can_add:
            selected.append(x)
            for a in selected[:-1]:
                sums.add(x + a)
            sums.add(2 * x)
    
    return selected


def _simulated_annealing(initial, temperature=10.0, cooling=0.995, max_iter=5000):
    """Simulated annealing to minimize size while maintaining maximality."""
    current = list(initial)
    current_sums = _sidon_sums(current)
    current_size = len(current)
    best = list(current)
    best_size = current_size
    
    for iteration in range(max_iter):
        if time.time() > DEADLINE - 2.0:
            break
        
        # Random move: try to remove an element
        if current_size > 1:
            idx = random.randint(0, current_size - 1)
            x = current[idx]
            new_current = [current[i] for i in range(current_size) if i != idx]
            new_sums = _sidon_sums(new_current)
            new_size = len(new_current)
            
            # Check if new set is maximal
            if _is_maximal(new_current):
                # Accept if smaller
                if new_size < best_size:
                    best = list(new_current)
                    best_size = new_size
                current = new_current
                current_sums = new_sums
                current_size = new_size
            else:
                # Reject
                pass
        else:
            break
    
    return best


def _greedy_maximal_from_valid(valid_set):
    """Given a valid Sidon set, try to make it maximal by adding elements."""
    selected = list(valid_set)
    sums = _sidon_sums(selected)
    selected_set = set(selected)
    
    # Try adding missing elements in random order
    missing = [x for x in range(1, N + 1) if x not in selected_set]
    random.shuffle(missing)
    
    for x in missing:
        if time.time() > DEADLINE - 2.0:
            break
        can_add = True
        for a in selected:
            if x + a in sums:
                can_add = False
                break
        if can_add and 2 * x in sums:
            can_add = False
        
        if can_add:
            selected.append(x)
            selected_set.add(x)
            for a in selected[:-1]:
                sums.add(x + a)
            sums.add(2 * x)
    
    return selected


def construct():
    start = time.time()
    best_size = N + 1
    best_set = None

    rng = random.Random(42)
    base = list(range(1, N + 1))

    # Strategy 1: Randomized greedy with multiple orderings
    for _ in range(200):
        if time.time() > DEADLINE - 5.0:
            break
        perm = base[:]
        rng.shuffle(perm)
        result = _greedy_maximal_sidon(perm)
        if result is not None and len(result) < best_size:
            best_size = len(result)
            best_set = result
            # Try to minimize further
            minimized = _repair_minimize(result)
            if len(minimized) < best_size:
                best_size = len(minimized)
                best_set = minimized

    # Strategy 2: Ascending and descending order
    for ordering in [base, base[::-1]]:
        if time.time() > DEADLINE - 5.0:
            break
        result = _greedy_maximal_sidon(ordering)
        if result is not None and len(result) < best_size:
            best_size = len(result)
            best_set = result
            minimized = _repair_minimize(result)
            if len(minimized) < best_size:
                best_size = len(minimized)
                best_set = minimized

    # Strategy 3: Simulated annealing on best found
    if best_set is not None and time.time() < DEADLINE - 3.0:
        sa_result = _simulated_annealing(best_set)
        if len(sa_result) < best_size:
            best_size = len(sa_result)
            best_set = sa_result

    # Strategy 4: More aggressive local search
    if best_set is not None and time.time() < DEADLINE - 1.0:
        # Try multiple random removals and repairs
        for _ in range(50):
            if time.time() > DEADLINE - 1.0:
                break
            candidate = list(best_set)
            # Randomly remove a few elements
            n = len(candidate)
            if n > 3:
                num_remove = min(3, n - 2)
                to_remove = set(random.sample(range(n), num_remove))
                candidate = [candidate[i] for i in range(n) if i not in to_remove]
            
            # Try to repair maximality
            sums = _sidon_sums(candidate)
            selected_set = set(candidate)
            missing = [x for x in range(1, N + 1) if x not in selected_set]
            random.shuffle(missing)
            
            for x in missing:
                if time.time() > DEADLINE - 1.0:
                    break
                can_add = True
                for a in candidate:
                    if x + a in sums:
                        can_add = False
                        break
                if can_add and 2 * x in sums:
                    can_add = False
                
                if can_add:
                    candidate.append(x)
                    selected_set.add(x)
                    for a in candidate[:-1]:
                        sums.add(x + a)
                    sums.add(2 * x)
            
            # Check if maximal
            if _is_maximal(candidate) and len(candidate) < best_size:
                best_size = len(candidate)
                best_set = candidate

    if best_set is None:
        # Fallback: ascending greedy
        selected = []
        sums = set()
        for x in range(1, N + 1):
            can_add = True
            for a in selected:
                if x + a in sums:
                    can_add = False
                    break
            if can_add and 2 * x in sums:
                can_add = False
            
            if can_add:
                selected.append(x)
                for a in selected[:-1]:
                    sums.add(x + a)
                sums.add(2 * x)
        best_set = selected

    return best_set
