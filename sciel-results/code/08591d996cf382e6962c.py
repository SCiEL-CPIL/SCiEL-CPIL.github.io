"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _compute_cardinality_sums_bitset(elements):
    """
    Compute bitsets of achievable sums for each cardinality.
    sums[k] is an integer where bit s is 1 if sum s is achievable with exactly k elements.
    """
    n = len(elements)
    if n == 0:
        return [1]
    
    sums = [0] * (n + 1)
    sums[0] = 1  # Sum 0 is achievable with 0 elements
    
    for elem in elements:
        # Update in reverse to avoid using elem multiple times
        for k in range(n, 0, -1):
            sums[k] |= (sums[k-1] << elem)
            
    return sums

def _is_valid(elements):
    """
    Check if the set satisfies the cardinality-separated subset sum constraint using bitsets.
    Returns True if for all k1 != k2, the sets of achievable sums are disjoint.
    """
    sums = _compute_cardinality_sums_bitset(elements)
    n = len(elements)
    # Check for intersection between any two different cardinality classes
    for k in range(n + 1):
        for j in range(k + 1, n + 1):
            if sums[k] & sums[j]:
                return False
    return True

def _check_add(elements, new_elem):
    """
    Check if adding new_elem to elements maintains validity.
    Uses a fast incremental check by computing the new sums and checking against old sums.
    """
    n = len(elements)
    if new_elem in elements:
        return False
    
    # Get current sums
    current_sums = _compute_cardinality_sums_bitset(elements)
    
    # Compute new sums with the added element
    new_elements = list(elements)
    new_elements.append(new_elem)
    new_elements.sort()
    
    new_sums = [0] * (n + 2)
    new_sums[0] = 1
    
    for elem in new_elements:
        for k in range(n + 1, 0, -1):
            new_sums[k] |= (new_sums[k-1] << elem)
    
    # Check disjointness
    for k in range(n + 2):
        for j in range(k + 1, n + 2):
            if new_sums[k] & new_sums[j]:
                return False
    return True

def _grow_greedy(initial_elements, candidates, deadline):
    """
    Grow the set greedily by adding elements that maintain validity.
    Tries candidates in the order given.
    """
    current = sorted(initial_elements)
    used = set(current)
    
    for candidate in candidates:
        if time.time() > deadline:
            break
        if candidate in used:
            continue
        if _check_add(current, candidate):
            current.append(candidate)
            used.add(candidate)
    
    return current

def _local_search(elements, candidates, deadline, max_iter=100):
    """
    Local search with 1-for-1 swaps to escape local optima.
    """
    current = sorted(elements)
    best = list(current)
    used = set(current)
    n = len(current)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        # Try removing one element and adding another
        for i in range(n):
            if time.time() > deadline:
                break
            
            elem_to_remove = current[i]
            
            # Try each candidate to add
            for candidate in candidates:
                if candidate in used or candidate == elem_to_remove:
                    continue
                
                # Form the new set
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
    deadline = start_time + 80  # 80s budget, 10s safety margin for overhead/return
    
    # Start with the known good solution: 226-256 (31 elements)
    base_solution = list(range(226, 257))
    
    # Candidates are the remaining numbers
    candidates = list(range(1, 226))
    random.shuffle(candidates)
    
    best = list(base_solution)
    
    # Phase 1: Greedy growth
    grown = _grow_greedy(base_solution, candidates, deadline)
    if len(grown) > len(best):
        best = list(grown)
    
    if time.time() > deadline:
        return best
    
    # Phase 2: Local search with swaps
    searched = _local_search(grown, candidates, deadline, max_iter=50)
    if len(searched) > len(best):
        best = list(searched)
    
    if time.time() > deadline:
        return best
    
    # Phase 3: Try a few more random restarts if time permits
    for trial in range(5):
        if time.time() > deadline:
            break
        
        random.shuffle(candidates)
        
        # Try greedy from base
        grown = _grow_greedy(base_solution, candidates, deadline)
        if len(grown) > len(best):
            best = list(grown)
            
        # Try local search
        if time.time() > deadline:
            break
        searched = _local_search(grown, candidates, deadline, max_iter=20)
        if len(searched) > len(best):
            best = list(searched)
            
    # Ensure we return a list of Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
