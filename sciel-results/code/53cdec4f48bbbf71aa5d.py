import numpy as np
import time
import random

N = 5000
RANGE = N + 1

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def _is_admissible(points, primes):
    """Check if a set of points is admissible."""
    n = len(points)
    if n == 0:
        return True
    point_set = set(points)
    for p in primes:
        if p > n:
            break
        # Check if all residue classes mod p are covered
        residues = set(x % p for x in point_set)
        if len(residues) == p:
            return False
    return True

def _greedy_residue_removal(x_indices, active, PRIMES, lookahead=1):
    """
    Greedy residue removal with lookahead tie-breaking.
    """
    current_size = int(np.sum(active))
    
    for i, p in enumerate(PRIMES):
        if p > current_size:
            break
        
        # Count active elements in each residue class mod p
        residues = x_indices[active] % p
        if len(residues) == 0:
            break
        
        # Count occurrences of each residue
        counts = np.bincount(residues, minlength=p)
        
        min_count = np.min(counts)
        
        if min_count == 0:
            # Already admissible for this prime
            continue
        
        # Find all classes with min count
        candidate_classes = np.where(counts == min_count)[0]
        
        if len(candidate_classes) == 1:
            r = candidate_classes[0]
        else:
            # Lookahead: pick the candidate that minimizes the min_count of the next prime
            best_r = candidate_classes[0]
            best_score = float('inf')
            
            # Find the next prime to check
            next_p = None
            for q in PRIMES[i+1:]:
                if q <= current_size - min_count: # Approx check
                    next_p = q
                    break
            
            if next_p is not None:
                for r in candidate_classes:
                    # Simulate removal
                    mask = (x_indices % p) != r
                    new_active = active & mask
                    new_residues = x_indices[new_active] % next_p
                    new_counts = np.bincount(new_residues, minlength=next_p)
                    new_min = np.min(new_counts)
                    
                    # We want to minimize the "difficulty" for the next step.
                    # A smaller min_count means we have to remove fewer elements later?
                    # Actually, if min_count is 0, it's already satisfied.
                    # If min_count is large, it's "harder" to satisfy?
                    # Let's say we prefer candidates that leave the next prime with min_count == 0 if possible.
                    # Or generally, prefer lower min_count?
                    # If min_count is 0, we skip removal. That's good.
                    # So score = new_min. Lower is better.
                    
                    if new_min < best_score:
                        best_score = new_min
                        best_r = r
            else:
                # No lookahead, pick random or smallest
                best_r = candidate_classes[random.randint(0, len(candidate_classes) - 1)]
            
            r = best_r
            
        # Deactivate elements in this class
        mask = (x_indices % p) == r
        current_size -= min_count
        active = active & (x_indices % p != r)
        
        if current_size == 0:
            break
    
    return active

def _polish(points, deadline):
    """
    Try to add back removed elements to maximize size.
    This is a heuristic polish.
    We know the current set is admissible.
    We try to add elements from [0, N] that are not in points.
    """
    point_set = set(points)
    current_size = len(points)
    
    # Get all possible candidates to add
    candidates = [x for x in range(RANGE) if x not in point_set]
    random.shuffle(candidates)
    
    # We need a faster check.
    # We can maintain the residue counts for the current set.
    # For each prime p <= current_size, we have counts of residues.
    # If we add x, for each p, counts[x % p] increases by 1.
    # We need to ensure that no prime p has all classes covered (i.e., min_count > 0 for all classes).
    # Wait, admissible means SOME class is missed. i.e., min_count == 0.
    # So we need to ensure that for every p <= new_size, min_count(p) == 0.
    
    active = np.zeros(RANGE, dtype=bool)
    for x in points:
        active[x] = True
    
    x_indices = np.arange(RANGE)
    
    # Precompute counts for current set
    # This is O(P * N) where P is number of primes <= N.
    # P ~ 600. N = 5000. 3M ops. Fast enough.
    
    counts_cache = {}
    min_counts = {}
    num_mins = {}
    
    def update_cache(active_mask):
        for p in PRIMES:
            if p > int(np.sum(active_mask)):
                break
            residues = x_indices[active_mask] % p
            if len(residues) == 0:
                counts_cache[p] = np.zeros(p, dtype=int)
                min_counts[p] = 0
                num_mins[p] = p
                continue
            c = np.bincount(residues, minlength=p)
            counts_cache[p] = c
            m = np.min(c)
            min_counts[p] = m
            num_mins[p] = np.sum(c == m)
    
    update_cache(active)
    
    for x in candidates:
        if time.time() > deadline:
            break
            
        # Check if adding x keeps it admissible
        # We need to check primes p <= current_size + 1
        
        is_valid = True
        for p in PRIMES:
            if p > current_size + 1:
                break
                
            if p not in counts_cache:
                # New prime, need to compute
                residues = x_indices[active] % p
                c = np.bincount(residues, minlength=p)
                counts_cache[p] = c
                min_counts[p] = np.min(c)
                num_mins[p] = np.sum(c == min_counts[p])
                
            c = counts_cache[p]
            m = min_counts[p]
            nm = num_mins[p]
            
            r = x % p
            current_count = c[r]
            
            # If we add x, c[r] becomes current_count + 1
            # New min:
            if current_count == m:
                # This class was one of the mins
                # If it was the only min, and the next smallest is m+1, then new min is m+1?
                # No, if current_count == m, and we increment it, it becomes m+1.
                # If nm == 1, then all other classes are > m.
                # The next smallest is at least m+1.
                # So new min is min(m+1, next_smallest).
                # Since next_smallest >= m+1, new min is m+1.
                # If nm > 1, there is another class with count m. So new min is m.
                
                if nm == 1:
                    # Need to find next smallest
                    # This is expensive to do dynamically.
                    # Approximation: if nm == 1, adding x to the unique min class
                    # will increase the min count.
                    # We need to know if the new min count is still 0.
                    # If m == 0, and nm == 1, then there is exactly one class with 0.
                    # Adding x to it makes it 1.
                    # If all other classes are >= 1, then new min is 1.
                    # This would make the prime "full" (min > 0), which is bad.
                    # So if m == 0 and nm == 1 and current_count == 0, adding x is INVALID.
                    
                    if m == 0:
                        is_valid = False
                        break
                else:
                    # nm > 1, so another class has count m. New min is still m.
                    pass
            else:
                # current_count > m.
                # Adding x increases a non-min class.
                # Min stays m.
                pass
        
        if is_valid:
            # Add x
            active[x] = True
            point_set.add(x)
            current_size += 1
            
            # Update cache for affected primes
            # This is lazy, we can just recompute if needed or update incrementally.
            # For simplicity, let's just mark cache as dirty or recompute?
            # Recomputing is O(P*N). Too slow per step.
            # Incremental update:
            # For each p, c[r] += 1.
            # Update m and nm.
            
            for p in PRIMES:
                if p > current_size:
                    break
                if p not in counts_cache:
                    continue
                c = counts_cache[p]
                m = min_counts[p]
                nm = num_mins[p]
                r = x % p
                old_c = c[r]
                c[r] = old_c + 1
                
                if old_c == m:
                    # We incremented a min class
                    if nm == 1:
                        # It was the unique min.
                        # New min is min(old_c+1, second_min).
                        # We don't have second_min stored.
                        # This makes incremental update hard.
                        # Let's just recompute for this prime?
                        # Or accept that the check might be slightly off?
                        # If we are conservative, we might reject valid additions.
                        # That's okay for a heuristic.
                        pass
                    else:
                        nm -= 1
                        # m stays same
                else:
                    # We incremented a non-min class.
                    # m and nm stay same.
                    pass
                
                min_counts[p] = m
                num_mins[p] = nm

    return np.where(active)[0].tolist()

def _min_degree_greedy(x_indices, active, PRIMES):
    """
    Minimum-degree greedy: at each step, for each prime p that is currently "full" (min_count > 0),
    we must remove at least one element from each residue class mod p.
    Actually, the constraint is: for every prime p <= |A|, some residue class mod p is missed.
    This means for every p <= |A|, min_count(p) == 0.
    
    The greedy removal approach works by iterating primes and removing the smallest class.
    A "minimum-degree" variant might consider the "cost" of removing a class as the number of elements removed,
    but also consider future impact.
    
    For now, let's implement a variant that picks the prime with the LARGEST min_count first?
    Or the prime with the smallest "slack"?
    
    Actually, the current greedy iterates primes in order 2, 3, 5, ...
    Maybe we should prioritize primes that are "tighter" or have larger min_count?
    If a prime has a large min_count, we have to remove many elements to satisfy it.
    If we handle it early, we might remove elements that also help satisfy later primes.
    
    Let's try sorting primes by min_count descending? Or by min_count / p?
    """
    current_size = int(np.sum(active))
    if current_size == 0:
        return active
    
    # We need to process primes until all are satisfied.
    # We'll maintain a list of active primes to process.
    # A prime is "active" if p <= current_size and min_count(p) > 0.
    
    # To avoid recomputing everything, we can just loop.
    # But the order matters.
    
    # Let's try a simple heuristic: always pick the prime with the largest min_count among active primes.
    # This is expensive to find, so we approximate by just checking the next few?
    # Or we can just stick to the current order but add randomization.
    
    # For this function, let's just return the result of the standard greedy with a random tie-breaker.
    # The "min degree" idea is complex to implement efficiently here.
    # Let's implement a different heuristic: 
    # Pick the prime p that minimizes (min_count(p) / p) or something?
    
    # Actually, let's just implement the standard greedy but with a different tie-breaking strategy.
    # If there are multiple classes with min_count, pick the one that appears least often in the "next" few primes?
    
    # For now, I'll just call the existing _greedy_residue_removal but with a flag or something?
    # No, let's implement a distinct logic.
    
    # Heuristic: Process primes in order of increasing p (standard).
    # Tie-breaking: Random.
    # This is basically the same as the seed.
    
    # Let's try: Process primes in order of DECREASING min_count?
    # We need to find which prime has the max min_count.
    # This requires checking all primes every step. Expensive.
    
    # Alternative: Just randomize the order of primes?
    # No, primes must be checked in some order.
    
    # Let's stick to the standard order but improve the tie-breaking.
    # If multiple classes have min_count, pick the one that is "least useful" for other primes.
    # i.e., the class r mod p such that the number of elements in class r that are also in "critical" classes of other primes is minimized?
    
    # This is getting complex. Let's just return the standard greedy result for now.
    # The improvement will come from the polish or from multiple restarts.
    
    return _greedy_residue_removal(x_indices, active, PRIMES, lookahead=1)

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_size = 0
    best_points = []
    
    # Precompute x indices
    x_indices = np.arange(RANGE)
    
    # Strategy 1: Greedy with lookahead
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        active = np.ones(RANGE, dtype=bool)
        
        # Use lookahead=1
        active = _greedy_residue_removal(x_indices, active, PRIMES, lookahead=1)
        
        points = np.where(active)[0].tolist()
        
        # Try to polish
        if time.time() < deadline - 2:
            polished_points = _polish(points, deadline - 2)
            if len(polished_points) > len(points):
                points = polished_points
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = points
            
    return best_points
