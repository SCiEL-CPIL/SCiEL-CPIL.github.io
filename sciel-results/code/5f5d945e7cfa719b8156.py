"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256
import time
import random
import sys

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

def _complete_to_maximal(selected, order=None):
    """
    Takes a Sidon set and greedily adds elements to make it maximal.
    If order is given, use that order; otherwise use 1..N.
    Returns the maximal Sidon set.
    """
    selected = list(selected)
    sums = _get_sums(selected)
    selected_set = set(selected)
    
    if order is None:
        order = list(range(1, N + 1))
    
    for x in order:
        if x in selected_set:
            continue
        if _can_add(sums, selected, x):
            selected.append(x)
            selected_set.add(x)
            for a in selected[:-1]:
                sums.add(a + x)
            sums.add(2 * x)
            
    return selected

def _local_search_minimize(selected, deadline):
    """
    Optimized Local search to minimize a maximal Sidon set.
    Uses incremental sum updates where possible, though recomputation is safer.
    """
    selected = list(selected)
    best = list(selected)
    best_size = len(selected)
    sums = _get_sums(selected)
    
    while time.time() < deadline:
        improved = False
        # Shuffle indices to avoid bias?
        indices = list(range(len(selected)))
        # random.shuffle(indices) # Keep deterministic for reproducibility in this block, or use rng? 
        # The function signature doesn't take rng. Let's stick to sequential but maybe reverse?
        
        for i in indices:
            if time.time() >= deadline:
                break
            
            # Try removing element i
            x = selected[i]
            # Construct trial sums by removing x's contributions
            # This is complex. Let's just recompute for simplicity, it's O(K^2) where K is size.
            # K is small (~15-20), N is 256. 256*20 = 5120 ops. Very fast.
            
            trial = selected[:i] + selected[i+1:]
            if not trial:
                continue
            
            trial_sums = _get_sums(trial)
            
            # Check if trial is still maximal
            if _is_maximal_fast(trial, trial_sums):
                selected = trial
                sums = trial_sums
                best = list(selected)
                best_size = len(selected)
                improved = True
                # Restart the loop to check new indices
                break
        
        if not improved:
            break
    
    return best

def _randomized_greedy(rng, N):
    """Randomized greedy: shuffle the order and greedily construct."""
    order = list(range(1, N + 1))
    rng.shuffle(order)
    return _greedy_construct(order, N)

def _backtracking_search(k, deadline, rng):
    """
    Try to find a maximal Sidon set of exactly size k using backtracking.
    Returns the set if found, None otherwise.
    """
    def backtrack(current, current_sums, start_idx):
        if len(current) == k:
            # Check if maximal
            if _is_maximal_fast(current, current_sums):
                return list(current)
            return None
        
        if time.time() >= deadline:
            return None
        
        # Try adding elements from start_idx onwards
        for x in range(start_idx, N + 1):
            if not _can_add(current_sums, current, x):
                continue
            
            new_current = current + [x]
            new_sums = set(current_sums)
            for a in current:
                new_sums.add(a + x)
            new_sums.add(2 * x)
            
            result = backtrack(new_current, new_sums, x + 1)
            if result is not None:
                return result
        
        return None
    
    return backtrack([], set(), 1)

def _swap_remove_search(selected, deadline, rng):
    """
    Try to remove one element and add another (possibly different) to keep maximality.
    This is a 1-swap local search move.
    """
    selected = list(selected)
    best = list(selected)
    best_size = len(selected)
    
    while time.time() < deadline:
        improved = False
        n = len(selected)
        for i in range(n):
            if time.time() >= deadline:
                break
            # Try removing element i
            trial = selected[:i] + selected[i+1:]
            if not trial:
                continue
            trial_sums = _get_sums(trial)
            
            # First check if simply removing works
            if _is_maximal_fast(trial, trial_sums):
                selected = trial
                best = list(selected)
                best_size = len(selected)
                improved = True
                break
            
            # Try swapping: remove i, add some x not in trial
            trial_set = set(trial)
            for x in range(1, N + 1):
                if x in trial_set:
                    continue
                if not _can_add(trial_sums, trial, x):
                    continue
                # Check if trial + [x] is maximal
                new_trial = trial + [x]
                new_sums = set(trial_sums)
                for a in trial:
                    new_sums.add(a + x)
                new_sums.add(2 * x)
                if _is_maximal_fast(new_trial, new_sums):
                    selected = new_trial
                    best = list(selected)
                    best_size = len(selected)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best

def _iterated_local_search(rng, deadline, initial_solution):
    """
    Iterated Local Search: Perturb, Complete, Local Search.
    """
    best = list(initial_solution)
    best_size = len(best)
    
    while time.time() < deadline:
        # Perturb: Remove a random element
        current = list(best)
        if not current:
            break
        
        idx = rng.randint(0, len(current) - 1)
        current.pop(idx)
        
        # Complete to maximal with random order
        order = list(range(1, N + 1))
        rng.shuffle(order)
        current = _complete_to_maximal(current, order)
        
        # Local search
        # Use a short deadline for this step to allow many iterations
        step_deadline = min(deadline, time.time() + 0.3)
        current = _local_search_minimize(current, step_deadline)
        
        if len(current) < best_size:
            best = current
            best_size = len(current)
            
    return best

def _min_max_greedy(rng, deadline):
    """
    Greedy construction for Minimal Maximal Sidon Set.
    Strategy: At each step, select the element that blocks the most other
    potential elements. This creates a "dense" blocker, reducing the number
    of elements needed to cover the whole range [1, N].
    """
    selected = []
    sums = set()
    selected_set = set()
    
    # Pre-compute conflicts? No, dynamic.
    # To speed up, we can maintain a list of available candidates.
    available = set(range(1, N + 1))
    
    while available:
        if time.time() >= deadline:
            break
            
        best_candidate = None
        best_block_count = -1
        best_new_sums = None
        
        # Iterate through candidates
        # Optimization: We can stop early if we find a candidate that blocks all others?
        # Unlikely.
        
        for x in available:
            # Check if x can be added to current selected
            if not _can_add(sums, selected, x):
                continue
            
            # Calculate blocking power
            new_sums = set(sums)
            for a in selected:
                new_sums.add(a + x)
            new_sums.add(2 * x)
            
            block_count = 0
            # Count how many other available elements y cannot be added if x is added
            for y in available:
                if y == x:
                    continue
                if not _can_add(new_sums, selected + [x], y):
                    block_count += 1
            
            # Tie-breaking: random or prefer larger/smaller?
            # Prefer larger block count. If tie, randomize to allow diversity in restarts?
            # For greedy, deterministic is fine, but we can add slight randomness.
            if block_count > best_block_count:
                best_block_count = block_count
                best_candidate = x
                best_new_sums = new_sums
        
        if best_candidate is None:
            # Should not happen if available is not empty and _can_add works correctly
            # But if no candidate can be added, we are stuck? No, available contains only addable ones?
            # Wait, available is updated by removal. We need to re-filter available.
            # The loop above checks _can_add. If none pass, available is effectively empty for this state.
            break
        
        # Select best_candidate
        selected.append(best_candidate)
        selected_set.add(best_candidate)
        available.remove(best_candidate)
        sums = best_new_sums
        
        # Remove from available any element that is now blocked
        # We can do this in the next iteration's check, but it's faster to prune now.
        new_available = set()
        for y in available:
            if _can_add(sums, selected, y):
                new_available.add(y)
        available = new_available
        
    return selected

def construct():
    deadline = time.time() + 27  # safety margin before 30s kill
    rng = random.Random(12345)
    
    best_maximal = None
    best_size = float('inf')
    
    def update_best(cand):
        nonlocal best_maximal, best_size
        if cand is not None and len(cand) < best_size:
            best_size = len(cand)
            best_maximal = list(cand)
    
    # 1. Run the new Min-Max Greedy multiple times with different seeds/times
    # This is the primary strategy based on the directive.
    for i in range(10):
        if time.time() >= deadline - 5:
            break
        step_deadline = min(deadline - 5, time.time() + 0.5)
        cand = _min_max_greedy(rng, step_deadline)
        update_best(cand)
        
        # Immediate local search on the result
        if best_maximal is not None:
            local_best = _local_search_minimize(best_maximal, min(deadline - 5, time.time() + 1.0))
            update_best(local_best)
            
    # 2. Fallback to standard greedy if Min-Max fails or is slow (shouldn't be)
    if best_maximal is None or best_size > 20: # Threshold
         cand = _greedy_construct(list(range(1, N + 1)), N)
         update_best(cand)
         cand = _greedy_construct(list(range(N, 0, -1)), N)
         update_best(cand)

    # 3. Refine with Iterated Local Search
    if best_maximal is not None and time.time() < deadline - 3:
        ils_best = _iterated_local_search(rng, deadline - 3, best_maximal)
        update_best(ils_best)
        
    # 4. Final polish
    if best_maximal is not None and time.time() < deadline - 0.5:
        final_local = _local_search_minimize(best_maximal, deadline - 0.5)
        update_best(final_local)
        
    # Final validation
    if best_maximal is not None:
        if _is_sidon(best_maximal) and _is_maximal_fast(best_maximal, _get_sums(best_maximal)):
            return sorted(best_maximal)
    
    # Fallback
    return sorted(_greedy_construct(list(range(1, N + 1)), N))
# EVOLVE-BLOCK-END
