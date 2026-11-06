"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import time
import random

def _compute_sums(selected):
    """Compute all pairwise sums a+b for a<=b in selected."""
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

def _is_maximal(selected):
    """Check if selected is inclusion-maximal Sidon in [1,N]."""
    if not selected:
        return False
    selected_set = set(selected)
    sums = _compute_sums(selected)
    
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Adding x creates a duplicate iff 2x or x+s is already in sums
        if 2 * x in sums:
            continue
        conflict = False
        for s in selected:
            if x + s in sums:
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
        # Check if x can be added
        new_sums = [x + a for a in selected] + [2 * x]
        # Check for internal conflicts in new_sums
        if len(set(new_sums)) != len(new_sums):
            continue
        # Check conflicts with existing sums
        if sums.intersection(new_sums):
            continue
        
        selected.append(x)
        sums.update(new_sums)
    return selected

def _local_search_reduce(selected, deadline):
    """
    Try to remove elements while maintaining maximality.
    Returns the reduced set.
    """
    current = list(selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element
        for x in list(current):
            if time.time() > deadline:
                break
            
            new_set = [y for y in current if y != x]
            
            # A subset of a Sidon set is always Sidon, so we don't need _is_sidon
            # We only need to check if it is still Maximal
            if _is_maximal(new_set):
                current = new_set
                improved = True
                # Restart scan to allow further removals
                break
                
    return current

def _construct_weighted(seed_val, strategy_id):
    """
    Construct using a weighted random order.
    strategy_id 0: favor edges (low and high numbers)
    strategy_id 1: favor middle numbers
    strategy_id 2: uniform random (fallback)
    """
    rng = random.Random(seed_val)
    remaining = list(range(1, N + 1))
    order = []
    
    while remaining:
        if strategy_id == 0:
            # Favor edges
            weights = []
            for x in remaining:
                # Distance from center
                dist = abs(x - (N + 1) / 2.0)
                weights.append(dist + 1.0)
        elif strategy_id == 1:
            # Favor middle
            weights = []
            for x in remaining:
                dist = abs(x - (N + 1) / 2.0)
                weights.append(N - dist + 1.0)
        else:
            weights = [1.0] * len(remaining)
        
        total_w = sum(weights)
        r = rng.random() * total_w
        cum = 0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                order.append(remaining[i])
                remaining.pop(i)
                break
        
    return _greedy_construct(order)

def construct():
    start_time = time.time()
    # Safety margin: 60s limit, use 55s to be safe
    deadline = start_time + 55.0
    
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
            
    # Strategy 2: Try different random orders
    # Use multiple trials with different strategies
    # We have ~50s left. Each trial takes ~0.1-0.5s for greedy + local search.
    # Let's try as many as we can.
    
    # Pre-generate some diverse orders
    # Strategy 0: Edge-favoring
    # Strategy 1: Middle-favoring
    # Strategy 2: Uniform random
    
    trial = 0
    while time.time() < deadline - 2.0:
        # Determine strategy based on trial number
        # Cycle through strategies
        strat_id = trial % 3
        seed_val = trial * 1000 + 42
        
        selected = _construct_weighted(seed_val, strat_id)
        
        # If this greedy result is already better, refine it
        if len(selected) < best_size:
            best = selected
            best_size = len(selected)
            if time.time() < deadline - 2.0:
                reduced = _local_search_reduce(best, deadline)
                if len(reduced) < best_size:
                    best = reduced
                    best_size = len(reduced)
        else:
            # Even if not better than best, it might reduce to a better size?
            # Local search is the key reducer. 
            # To save time, only refine if it's promising or if we have time.
            # Let's refine any candidate that is <= best_size + 2 to be safe, or just the best one.
            # Actually, local search is fast. Let's refine this candidate.
            if time.time() < deadline - 2.0:
                reduced = _local_search_reduce(selected, deadline)
                if len(reduced) < best_size:
                    best = reduced
                    best_size = len(reduced)
                    
        trial += 1

    return best
# EVOLVE-BLOCK-END
