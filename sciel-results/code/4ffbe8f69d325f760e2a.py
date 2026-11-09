"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected):
    """Check if selected is inclusion-maximal in [1,N]."""
    sel_set = set(selected)
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if adding x preserves Sidon property
        new_sums = set()
        valid = True
        for a in selected:
            s = a + x
            if s in sums or s in new_sums:
                valid = False
                break
            new_sums.add(s)
            if not valid:
                break
        if valid:
            s = 2 * x
            if s in sums or s in new_sums:
                valid = False
            else:
                new_sums.add(s)
        if valid:
            return False  # Can add x, so not maximal
    return True

def _greedy_construct():
    """Greedy ascending construction."""
    selected = []
    sums = set()
    for x in range(1, N + 1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _greedy_random(seed_val):
    """Greedy construction with randomized ordering."""
    rng = random.Random(seed_val)
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    selected = []
    sums = set()
    for x in candidates:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _try_remove(selected):
    """Try to remove elements while maintaining maximality. Returns best (smallest) maximal set found."""
    best = list(selected)
    best_size = len(best)
    
    # Try removing each element
    for i in range(len(best)):
        trial = best[:i] + best[i+1:]
        if _is_valid_sidon(trial) and _is_maximal(trial):
            best = trial
            best_size = len(best)
            break  # Restart from new state
    
    return best

def _try_swap(selected):
    """Try swapping an element with a missing one to potentially enable further reductions."""
    best = list(selected)
    sel_set = set(best)
    missing = [x for x in range(1, N + 1) if x not in sel_set]
    
    improved = True
    while improved:
        improved = False
        for i in range(len(best)):
            for m in missing:
                trial = best[:i] + [m] + best[i+1:]
                if _is_valid_sidon(trial) and _is_maximal(trial):
                    if len(trial) < len(best):
                        best = trial
                        sel_set = set(best)
                        missing = [x for x in range(1, N + 1) if x not in sel_set]
                        improved = True
                        break
            if improved:
                break
    return best

def _local_search(selected, max_iters=2000):
    """Local search: try removals and swaps to minimize size while maintaining maximality."""
    best = list(selected)
    best_size = len(best)
    
    for _ in range(max_iters):
        sel_set = set(best)
        missing = [x for x in range(1, N + 1) if x not in sel_set]
        
        # Try removals
        removed = False
        for i in range(len(best)):
            trial = best[:i] + best[i+1:]
            if _is_valid_sidon(trial) and _is_maximal(trial):
                best = trial
                best_size = len(best)
                removed = True
                break
        
        if removed:
            continue
        
        # Try swaps (replace one element with another to enable future removals)
        swapped = False
        for i in range(len(best)):
            for m in missing:
                trial = best[:i] + [m] + best[i+1:]
                if _is_valid_sidon(trial) and _is_maximal(trial):
                    # Check if this swap enables a subsequent removal
                    trial_set = set(trial)
                    trial_missing = [x for x in range(1, N + 1) if x not in trial_set]
                    for j in range(len(trial)):
                        trial2 = trial[:j] + trial[j+1:]
                        if _is_valid_sidon(trial2) and _is_maximal(trial2):
                            best = trial2
                            best_size = len(best)
                            swapped = True
                            break
                    if swapped:
                        break
            if swapped:
                break
        
        if not swapped and not removed:
            break
    
    return best

def _construct_with_deadline(deadline):
    """Main construction with deadline guard."""
    start_time = time.time()
    
    # Start with greedy ascending
    best = _greedy_construct()
    best_size = len(best)
    
    # Try local search on greedy result
    candidate = _local_search(best, max_iters=500)
    if len(candidate) < best_size:
        best = candidate
        best_size = len(candidate)
    
    elapsed = time.time() - start_time
    if elapsed > deadline * 0.5:
        return best
    
    # Try randomized greedy constructions
    for seed_val in range(50):
        if time.time() > deadline - 5:
            break
        candidate = _greedy_random(seed_val)
        if len(candidate) < best_size:
            candidate = _local_search(candidate, max_iters=300)
            if len(candidate) < best_size:
                best = candidate
                best_size = len(candidate)
    
    # Final local search pass
    if time.time() < deadline - 3:
        candidate = _local_search(best, max_iters=1000)
        if len(candidate) < best_size:
            best = candidate
            best_size = len(candidate)
    
    return best

def construct():
    """Construct a small inclusion-maximal Sidon set in [1,256]."""
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s kill
    
    best = _construct_with_deadline(deadline)
    
    # Verify the result is valid
    if not _is_valid_sidon(best) or not _is_maximal(best):
        # Fallback to greedy
        best = _greedy_construct()
    
    return best

# EVOLVE-BLOCK-END
