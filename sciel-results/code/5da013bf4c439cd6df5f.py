"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256
import time
import random

# EVOLVE-BLOCK-START
def _is_sidon(selected):
    """Check if a set is Sidon (all pairwise sums distinct, incl. diagonals)."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_sums(selected):
    """Get all pairwise sums a+b with a<=b."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    return sums

def _can_add(sums, selected, x):
    """Can x be added to selected (given its sums) while staying Sidon?"""
    for a in selected:
        s = a + x
        if s in sums:
            return False
    if 2 * x in sums:
        return False
    return True

def _is_maximal_fast(selected, sums):
    """Check maximality: no missing element can be added. Early exit."""
    selected_set = set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        if _can_add(sums, selected, x):
            return False
    return True

def _greedy_construct(order, N):
    """Greedy construction using a given order; runs to completion -> maximal."""
    selected = []
    sums = set()
    for x in order:
        if _can_add(sums, selected, x):
            selected.append(x)
            for a in selected[:-1]:
                sums.add(a + x)
            sums.add(2 * x)
    return selected

def _complete_to_maximal(selected):
    """
    Takes a Sidon set and greedily adds elements to make it maximal.
    Returns the maximal Sidon set.
    """
    selected = list(selected)
    sums = _get_sums(selected)
    selected_set = set(selected)
    
    # We iterate through elements. If we add an element, it might block others.
    # To be deterministic and fast, we just scan 1..N and add if possible.
    # This ensures maximality because after the pass, no element 1..N can be added
    # (since we checked all of them, and adding only increases sums, making it harder to add others? 
    # Wait. If we add x, it might make y addable? No, adding x creates MORE sums, 
    # so it can only BLOCK more elements, not unblock them.
    # So a single pass in any order will result in a maximal set?
    # Yes. If y was not addable before adding x, it remains not addable.
    # If y was addable before, and we didn't add it, but added x, y might become NOT addable.
    # So we must be careful. 
    # If we scan 1..N:
    # 1. Check 1. If addable, add.
    # 2. Check 2. If addable, add.
    # ...
    # At the end, for any x, if it wasn't added, it was either not addable at the time of check,
    # or it was skipped? No, we check all.
    # If x was addable at step i, we add it.
    # If x was not addable at step i, it stays not addable because we only add elements later.
    # So a single forward pass ensures maximality.
    
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        if _can_add(sums, selected, x):
            selected.append(x)
            selected_set.add(x)
            for a in selected[:-1]:
                sums.add(a + x)
            sums.add(2 * x)
            
    return selected

def construct():
    deadline = time.time() + 25  # safety margin before 30s kill
    rng = random.Random(42)

    best_maximal = None
    best_size = float('inf')

    # 1. Initialize with a few greedy baselines
    asc_order = list(range(1, N + 1))
    init_set = _greedy_construct(asc_order, N)
    best_maximal = init_set
    best_size = len(init_set)

    desc_order = list(range(N, 0, -1))
    init_set2 = _greedy_construct(desc_order, N)
    if len(init_set2) < best_size:
        best_maximal = init_set2
        best_size = len(init_set2)

    # 2. Simulated Annealing on Sidon sets
    # State: A list of elements (Sidon set)
    # We start from the best baseline
    current = list(best_maximal)
    current_sums = _get_sums(current)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    steps = 0
    last_eval_step = 0
    eval_interval = 100 # Check completion every 100 steps
    
    while time.time() < deadline - 1.0:
        if T < T_min:
            break
            
        # Generate move
        if rng.random() < 0.5:
            # Remove
            if not current:
                continue
            idx = rng.randint(0, len(current) - 1)
            x = current[idx]
            new_current = current[:idx] + current[idx+1:]
            # Recalculate sums? Or just try to add back later?
            # For SA, we just need to know if it's Sidon (it is, subset) and size.
            # We don't need sums for acceptance, only for next move generation if we add.
            # But we need to update current_sums for the next step if we accept.
            # Let's just store the set and recompute sums if accepted? 
            # Sum recomputation is O(K^2). K~15. 225 ops. Cheap.
            
            candidate = new_current
            candidate_size = len(candidate)
        else:
            # Add
            x = rng.randint(1, N)
            if x in current:
                continue
            if not _can_add(current_sums, current, x):
                continue
            candidate = current + [x]
            candidate_size = len(candidate)
            
        # Acceptance
        delta = candidate_size - len(current)
        if delta < 0 or rng.random() < pow(2.71828, -delta / T):
            current = candidate
            current_sums = _get_sums(current)
            
        T *= cooling
        steps += 1
        
        # Periodic evaluation: complete to maximal and check size
        if steps - last_eval_step >= eval_interval:
            last_eval_step = steps
            # Complete current Sidon set to maximal
            completed = _complete_to_maximal(current)
            if len(completed) < best_size:
                best_size = len(completed)
                best_maximal = completed
                # Restart SA from this better state?
                # Or just continue? Continuing might be good to explore.
                # Let's restart temperature if we found a new best
                # T = max(T, 0.5) 

    # Final check on the best found
    if best_maximal is not None:
        # Ensure it's valid
        if _is_sidon(best_maximal) and _is_maximal_fast(best_maximal, _get_sums(best_maximal)):
            return sorted(best_maximal)
            
    # Fallback
    return sorted(_greedy_construct(asc_order, N))
# EVOLVE-BLOCK-END
