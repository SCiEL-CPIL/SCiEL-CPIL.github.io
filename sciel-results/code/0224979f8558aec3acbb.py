import time
import random

def _compute_cardinality_sums_bitset(elements):
    """Compute bitsets of achievable sums for each cardinality.
    sums[k] is a Python int where bit s is 1 iff sum s is achievable with exactly k elements."""
    n = len(elements)
    if n == 0:
        return [1]
    sums = [0] * (n + 1)
    sums[0] = 1  # Sum 0 achievable with 0 elements
    for elem in elements:
        for k in range(n, 0, -1):
            sums[k] |= (sums[k - 1] << elem)
    return sums

def _is_valid(elements):
    """Check if the set satisfies cardinality-separated subset sum constraint."""
    n = len(elements)
    if n == 0:
        return True
    sums = _compute_cardinality_sums_bitset(elements)
    # Check pairwise disjointness across different cardinalities
    for k1 in range(n + 1):
        for k2 in range(k1 + 1, n + 1):
            if sums[k1] & sums[k2]:
                return False
    return True

def _check_add(elements, new_elem):
    """Check if adding new_elem to elements maintains validity."""
    if new_elem in elements:
        return False
    new_elements = sorted(elements + [new_elem])
    return _is_valid(new_elements)

def _find_max_consecutive_set():
    """Find the largest set of consecutive integers in 1..256 that is valid.
    For {L, L+1, ..., L+n-1}, validity iff L > max_k k(n-k-1) for k=1..n-1."""
    best = []
    for n in range(60, 0, -1):
        # max of k(n-k-1) for k in 1..n-1
        # This is maximized at k = floor((n-1)/2) or ceil((n-1)/2)
        max_k_val = 0
        for k in range(1, n):
            val = k * (n - k - 1)
            if val > max_k_val:
                max_k_val = val
        min_L = max_k_val + 1
        max_L = 256 - n + 1
        if min_L > max_L:
            continue
        # Try L = min_L (smallest valid L gives most room, but we just need one valid)
        L = min_L
        S = [L + i for i in range(n)]
        if _is_valid(S):
            best = S
            break
    return best

def _grow_greedy(initial_elements, candidates, deadline):
    """Grow the set greedily by adding elements that maintain validity."""
    current = sorted(initial_elements)
    used = set(current)
    
    for candidate in candidates:
        if time.time() > deadline:
            break
        if candidate in used:
            continue
        if _check_add(current, candidate):
            current.append(candidate)
            current.sort()
            used.add(candidate)
    
    return current

def _local_search(elements, candidates, deadline, max_iter=100):
    """Local search with 1-for-1 swaps to escape local optima."""
    current = sorted(elements)
    best = list(current)
    used = set(current)
    n = len(current)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            elem_to_remove = current[i]
            
            for candidate in candidates:
                if candidate in used or candidate == elem_to_remove:
                    continue
                
                new_set = current[:i] + current[i+1:] + [candidate]
                new_set.sort()
                
                if _is_valid(new_set):
                    current = new_set
                    used = set(current)
                    improved = True
                    if len(current) > len(best):
                        best = list(current)
                    break  # Accept first improvement, restart search
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill
    
    # Phase 0: Find the largest valid consecutive set
    base_solution = _find_max_consecutive_set()
    if not base_solution:
        base_solution = [1]
    
    best = list(base_solution)
    
    # Phase 1: Greedy growth from base, trying smaller elements first
    candidates = [x for x in range(1, 257) if x not in set(base_solution)]
    candidates.sort()  # Try smaller values first (they're more likely to fit)
    
    grown = _grow_greedy(base_solution, candidates, deadline)
    if len(grown) > len(best):
        best = list(grown)
    
    if time.time() > deadline:
        return [int(x) for x in best]
    
    # Phase 2: Local search with swaps
    remaining_candidates = [x for x in range(1, 257) if x not in set(grown)]
    searched = _local_search(grown, remaining_candidates, deadline, max_iter=50)
    if len(searched) > len(best):
        best = list(searched)
    
    if time.time() > deadline:
        return [int(x) for x in best]
    
    # Phase 3: Try a few random restarts if time permits
    for trial in range(5):
        if time.time() > deadline:
            break
        
        random.shuffle(candidates)
        
        # Try greedy from base with shuffled order
        grown = _grow_greedy(base_solution, candidates, deadline)
        if len(grown) > len(best):
            best = list(grown)
        
        # Try local search
        if time.time() > deadline:
            break
        remaining = [x for x in range(1, 257) if x not in set(grown)]
        searched = _local_search(grown, remaining, deadline, max_iter=20)
        if len(searched) > len(best):
            best = list(searched)
    
    # Final verification
    if _is_valid(best):
        return [int(x) for x in best]
    
    # Fallback
    return [int(x) for x in base_solution]
