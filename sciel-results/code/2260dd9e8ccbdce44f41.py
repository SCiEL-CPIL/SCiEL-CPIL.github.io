"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _is_sidon_fast(selected):
    """Fast Sidon check using numpy."""
    arr = np.array(selected, dtype=np.int32)
    n = len(arr)
    if n == 0:
        return True
    # All pairwise sums a+b for a<=b
    sums = np.zeros(n*(n+1)//2, dtype=np.int32)
    idx = 0
    for i in range(n):
        sums[idx] = 2 * arr[i]
        idx += 1
        for j in range(i+1, n):
            sums[idx] = arr[i] + arr[j]
            idx += 1
    # Check for duplicates
    return len(np.unique(sums)) == len(sums)

def _is_maximal_fast(selected):
    """Check if selected is inclusion-maximal Sidon in [1,N]."""
    selected_set = set(selected)
    arr = np.array(selected, dtype=np.int32)
    n = len(arr)
    
    # Compute all sums
    sums_list = []
    for i in range(n):
        sums_list.append(2 * arr[i])
        for j in range(i+1, n):
            sums_list.append(arr[i] + arr[j])
    sums_set = set(sums_list)
    
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Check if x can be added
        if 2 * x in sums_set:
            continue
        conflict = False
        for s in arr:
            if x + s in sums_set:
                conflict = True
                break
        if not conflict:
            return False
    return True

def _greedy_construct(order):
    """Greedy Sidon set construction following the given order."""
    selected = []
    sums = set()
    for x in order:
        new_sums = [x + a for a in selected] + [2 * x]
        if len(set(new_sums)) == len(new_sums) and not sums.intersection(new_sums):
            selected.append(x)
            sums.update(new_sums)
    return selected

def _try_remove(selected, x):
    """Try removing x from selected, check if still Sidon and maximal."""
    new_list = [y for y in selected if y != x]
    if not _is_sidon_fast(new_list):
        return None
    if _is_maximal_fast(new_list):
        return new_list
    return None

def _local_search_reduce(selected, deadline):
    """Try to remove elements while maintaining maximality. Multiple passes."""
    current = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try each element for removal
        for x in list(current):
            if time.time() > deadline:
                break
            result = _try_remove(current, x)
            if result is not None:
                current = result
                improved = True
                break  # Restart scan after a removal
    return current

def _construct_random_order(seed_val):
    """Construct using a random order with given seed."""
    rng = random.Random(seed_val)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    return _greedy_construct(order)

def _construct_weighted(seed_val):
    """Weighted construction favoring certain positions."""
    rng = random.Random(seed_val)
    # Try different weighting strategies
    weights = []
    for x in range(1, N + 1):
        # Favor numbers that are "spread out"
        w = 1.0
        if x <= N // 4:
            w = 1.5
        elif x >= 3 * N // 4:
            w = 1.5
        else:
            w = 0.8
        weights.append(w)
    
    # Sample without replacement using weights
    order = []
    remaining = list(range(1, N + 1))
    rem_weights = list(weights)
    while remaining:
        total_w = sum(rem_weights)
        r = rng.random() * total_w
        cum = 0
        for i, w in enumerate(rem_weights):
            cum += w
            if r <= cum:
                order.append(remaining[i])
                remaining.pop(i)
                rem_weights.pop(i)
                break
    return _greedy_construct(order)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    best = None
    best_size = N + 1
    
    # Strategy 1: Ascending greedy (baseline)
    order = list(range(1, N + 1))
    selected = _greedy_construct(order)
    if len(selected) < best_size:
        best = selected
        best_size = len(selected)
    
    # Local search to reduce size
    if time.time() < deadline:
        reduced = _local_search_reduce(best, deadline)
        if len(reduced) < best_size:
            best = reduced
            best_size = len(reduced)
    
    # Strategy 2: Try many different random orders
    if time.time() < deadline - 3.0:
        for trial in range(200):
            if time.time() > deadline:
                break
            selected = _construct_random_order(trial)
            if len(selected) < best_size:
                best = selected
                best_size = len(selected)
                # Try local search on this better result
                if time.time() < deadline - 2.0:
                    reduced = _local_search_reduce(best, deadline)
                    if len(reduced) < best_size:
                        best = reduced
                        best_size = len(reduced)
    
    # Strategy 3: Weighted constructions
    if time.time() < deadline - 3.0:
        for trial in range(100):
            if time.time() > deadline:
                break
            selected = _construct_weighted(trial + 1000)
            if len(selected) < best_size:
                best = selected
                best_size = len(selected)
                if time.time() < deadline - 2.0:
                    reduced = _local_search_reduce(best, deadline)
                    if len(reduced) < best_size:
                        best = reduced
                        best_size = len(reduced)
    
    # Strategy 4: Try starting from smaller sets and building up
    # Start with a few well-chosen elements, then greedily add
    if time.time() < deadline - 5.0:
        # Try different initial seeds
        for init_size in range(3, 8):
            if time.time() > deadline:
                break
            for trial in range(20):
                if time.time() > deadline:
                    break
                rng = random.Random(trial * 7919 + init_size * 104729)
                # Pick initial elements spread out
                initial = rng.sample(range(1, N + 1), min(init_size, N))
                initial.sort()
                
                # Check if initial is Sidon
                if not _is_sidon_fast(initial):
                    continue
                
                # Now greedily add remaining elements
                remaining = [x for x in range(1, N + 1) if x not in set(initial)]
                rng.shuffle(remaining)
                
                selected = list(initial)
                sums = set()
                for i in range(len(selected)):
                    sums.add(2 * selected[i])
                    for j in range(i + 1, len(selected)):
                        sums.add(selected[i] + selected[j])
                
                for x in remaining:
                    new_sums = [x + a for a in selected] + [2 * x]
                    if len(set(new_sums)) == len(new_sums) and not sums.intersection(new_sums):
                        selected.append(x)
                        sums.update(new_sums)
                
                if len(selected) < best_size:
                    best = selected
                    best_size = len(selected)
                    if time.time() < deadline - 2.0:
                        reduced = _local_search_reduce(best, deadline)
                        if len(reduced) < best_size:
                            best = reduced
                            best_size = len(reduced)
    
    return best
# EVOLVE-BLOCK-END
