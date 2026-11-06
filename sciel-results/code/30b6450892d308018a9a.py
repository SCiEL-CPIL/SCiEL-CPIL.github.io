import time
import itertools
import random

def _is_sidon_with_diagonals_fast(subset):
    """Check if subset has all unordered pair sums unique including diagonals (i.e. i<=j).
    Uses a set for O(n^2) check."""
    n = len(subset)
    seen = set()
    for i in range(n):
        ai = subset[i]
        for j in range(i, n):
            v = ai + subset[j]
            if v in seen:
                return False
            seen.add(v)
    return True

def _max_sidon_size_ambient(ambient):
    """Find maximum k such that some k-subset of ambient is Sidon (with diagonals).
    ambient has 12 elements, so k is in [1..12].
    Exact search by checking subsets in decreasing size order."""
    n = len(ambient)
    # Check full set first
    if _is_sidon_with_diagonals_fast(ambient):
        return n
    
    # Check subsets of size n-1, n-2, etc.
    for k in range(n - 1, 0, -1):
        for combo in itertools.combinations(ambient, k):
            if _is_sidon_with_diagonals_fast(list(combo)):
                return k
    return 0

def _count_collisions(ambient):
    """Count the number of sum collisions in the ambient set.
    Higher collision count implies harder to find large Sidon subsets."""
    n = len(ambient)
    sums = {}
    for i in range(n):
        for j in range(i, n):
            s = ambient[i] + ambient[j]
            if s in sums:
                sums[s] += 1
            else:
                sums[s] = 1
    
    collisions = 0
    for count in sums.values():
        if count > 1:
            # Number of pairs of pairs with same sum is C(count, 2)
            # But we want to count "bad" quadruples.
            # If a sum appears 'c' times, it contributes to collisions.
            # For the purpose of surrogate optimization, just summing (c-1) or c^2 works.
            collisions += count * (count - 1)
    return collisions

def _random_ambient(seed=None):
    """Generate a random ambient set of 12 distinct integers in [0,128]."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    return sorted(rng.sample(range(129), 12))

def _local_search(initial, deadline, rng):
    """Simulated Annealing / Hill Climbing to minimize max Sidon size."""
    current = list(initial)
    current_k = _max_sidon_size_ambient(current)
    current_collisions = _count_collisions(current)
    
    best = list(current)
    best_k = current_k
    best_collisions = current_collisions
    
    temp = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random index to change
        i = rng.randint(0, 11)
        old_val = current[i]
        
        # Generate a new value
        new_val = rng.randint(0, 128)
        
        # Ensure uniqueness
        if new_val in current:
            continue
            
        candidate = list(current)
        candidate[i] = new_val
        candidate.sort()
        
        new_k = _max_sidon_size_ambient(candidate)
        new_collisions = _count_collisions(candidate)
        
        # Acceptance criteria
        accept = False
        if new_k < best_k:
            accept = True
        elif new_k == best_k:
            # Prefer more collisions (harder set)
            if new_collisions > best_collisions:
                accept = True
            else:
                # Random walk to explore
                if rng.random() < 0.1:
                    accept = True
        else:
            # new_k > best_k, accept with low probability to escape local opt
            if rng.random() < 0.01 * temp:
                accept = True
        
        if accept:
            current = candidate
            current_k = new_k
            current_collisions = new_collisions
            
            if new_k < best_k:
                best_k = new_k
                best = list(current)
                best_collisions = new_collisions
                if best_k <= 3:
                    return best, best_k
            elif new_k == best_k and new_collisions > best_collisions:
                best = list(current)
                best_collisions = new_collisions
                if best_k <= 3:
                    return best, best_k
            
            temp *= cooling
        else:
            temp *= cooling

    return best, best_k

def _select_best_ambient():
    deadline = time.time() + 85 # Safety margin for 90s limit
    rng = random.Random(42)
    
    best = None
    best_k = 13
    best_collisions = -1
    
    # Start with structured candidates that are known to have high collision density
    # Arithmetic progressions are good starting points
    structured_cands = [
        list(range(12)),
        list(range(0, 24, 2)),
        list(range(0, 36, 3)),
        list(range(0, 48, 4)),
        list(range(0, 60, 5)),
        list(range(0, 72, 6)),
        list(range(0, 84, 7)),
        list(range(0, 96, 8)),
        list(range(0, 108, 9)),
        list(range(0, 120, 10)),
        list(range(0, 132, 11)), # 132 > 128, invalid, but sample handles range
    ]
    
    # Filter valid structured candidates
    valid_structured = []
    for cand in structured_cands:
        if len(cand) == 12 and len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            valid_structured.append(cand)
    
    # Evaluate structured candidates
    for cand in valid_structured:
        k = _max_sidon_size_ambient(cand)
        c = _count_collisions(cand)
        if k < best_k or (k == best_k and c > best_collisions):
            best_k = k
            best_collisions = c
            best = list(cand)
    
    # If no valid structured candidate found, use random
    if best is None:
        best = _random_ambient(1)
        best_k = _max_sidon_size_ambient(best)
        best_collisions = _count_collisions(best)
    
    # Run local search from the best structured candidate
    if time.time() < deadline - 10:
        best, best_k = _local_search(best, deadline, rng)
    
    # Try multiple random restarts if time permits
    # We can afford many restarts because evaluation is fast
    restarts = 0
    while time.time() < deadline - 5:
        rand_cand = _random_ambient()
        # Quick check: if this random set is already better, great.
        # Otherwise, run local search on it.
        
        # To save time, we can first check if it's promising based on collisions
        # But local search is robust.
        
        local_best, local_k = _local_search(rand_cand, min(time.time() + 10, deadline - 2), rng)
        
        # Compare with global best
        local_collisions = _count_collisions(local_best)
        if local_k < best_k or (local_k == best_k and local_collisions > best_collisions):
            best_k = local_k
            best = local_best
            best_collisions = local_collisions
            if best_k <= 3:
                break
        
        restarts += 1
        # If we've done many restarts, maybe stop?
        # The deadline guard handles this.
        
    return best

def construct():
    # Return the best ambient set found
    return _select_best_ambient()
