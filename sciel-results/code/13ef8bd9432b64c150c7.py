"""Program to construct a set of 12 distinct positive integers with distinct subset sums, minimizing the largest element."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _check_valid(elements):
    """Check if all 4096 subset sums are distinct. Returns True if valid."""
    subset_sums = {0}
    for elem in elements:
        new_sums = set()
        for s in subset_sums:
            ns = s + elem
            if ns in subset_sums:
                return False
            new_sums.add(ns)
        subset_sums = subset_sums | new_sums
    return len(subset_sums) == 4096

def _smallest_valid_next(subset_sums):
    """Find the smallest positive integer that can be added without collision."""
    candidate = 1
    while True:
        has_collision = False
        for s in subset_sums:
            if (s + candidate) in subset_sums:
                has_collision = True
                break
        if not has_collision:
            return candidate
        candidate += 1

def _greedy_construct():
    """Greedy minimal extension: pick smallest valid next element."""
    elements = [1]
    subset_sums = {0, 1}
    for _ in range(11):
        candidate = _smallest_valid_next(subset_sums)
        elements.append(candidate)
        new_sums = set()
        for s in subset_sums:
            new_sums.add(s + candidate)
        subset_sums = subset_sums | new_sums
    return elements

def _search_shortest_path(deadline, rng):
    """
    Shortest-path search over sorted tuples of 12 distinct positive integers.
    State: sorted tuple of k elements (k from 1 to 12).
    Edge: replace one element with a different value, re-sort, check validity.
    Weight: current maximum element.
    Goal: find a valid 12-tuple with minimum max element.
    
    We use a priority queue keyed by (max_element, sum_of_elements) to explore
    promising states first.
    """
    import heapq
    
    # Start from greedy solution
    greedy = _greedy_construct()
    best = greedy[:]
    best_max = max(best)
    
    # Priority queue: (max_elem, sum_elems, tuple_of_elements)
    pq = [(best_max, sum(best), tuple(best))]
    visited = {tuple(best)}
    
    while pq and time.time() < deadline:
        max_elem, sum_elems, state = heapq.heappop(pq)
        
        if max_elem >= best_max:
            # Can't improve, and heap is ordered by max_elem
            continue
        
        # Try to improve by replacing each element with a smaller valid value
        for i in range(len(state)):
            old_val = state[i]
            
            # Try replacing with values from 1 to old_val-1
            # But we need to be smart: only try values that could lead to improvement
            for new_val in range(1, old_val):
                # Quick check: new_val must not already be in the set
                if new_val in state:
                    continue
                
                # Create candidate
                candidate = list(state)
                candidate[i] = new_val
                candidate.sort()
                
                # Check distinctness
                if len(set(candidate)) != len(candidate):
                    continue
                
                # Check validity
                if _check_valid(candidate):
                    cand_max = max(candidate)
                    cand_sum = sum(candidate)
                    cand_tuple = tuple(candidate)
                    
                    if cand_max < best_max:
                        best = candidate
                        best_max = cand_max
                    
                    if cand_tuple not in visited:
                        visited.add(cand_tuple)
                        heapq.heappush(pq, (cand_max, cand_sum, cand_tuple))
    
    return best

def _local_search_polish(initial, deadline, rng):
    """
    Final polish: local search to try to reduce the maximum element further.
    Uses simulated annealing with a cooling schedule.
    """
    current = initial[:]
    current_max = max(current)
    best = current[:]
    best_max = current_max
    
    temp = 10.0
    cooling = 0.99
    
    while time.time() < deadline and temp > 0.1:
        # Pick a random element to modify
        idx = rng.randint(0, len(current) - 1)
        old_val = current[idx]
        
        # Try a random perturbation
        if rng.random() < 0.7:
            # Try smaller value
            if old_val <= 1:
                temp *= cooling
                continue
            new_val = rng.randint(1, old_val - 1)
        else:
            # Try slightly larger or smaller
            delta = rng.randint(-5, 5)
            new_val = old_val + delta
        
        if new_val <= 0 or new_val in current:
            temp *= cooling
            continue
        
        candidate = current[:]
        candidate[idx] = new_val
        candidate.sort()
        
        if len(set(candidate)) != len(candidate):
            temp *= cooling
            continue
        
        if _check_valid(candidate):
            cand_max = max(candidate)
            delta_max = cand_max - current_max
            
            # Accept if better or with probability exp(-delta/temp)
            if delta_max < 0 or (delta_max > 0 and rng.random() < pow(2.71828, -delta_max / temp)):
                current = candidate
                current_max = cand_max
                if cand_max < best_max:
                    best = candidate
                    best_max = cand_max
        
        temp *= cooling
    
    return best

def construct():
    """
    Multi-phase search:
    1. Greedy construction for initial solution
    2. Shortest-path search over state space
    3. Local search polish
    """
    deadline = time.time() + 85  # safety margin before 90s kill
    rng = random.Random(42)
    
    # Phase 1: Greedy
    best = _greedy_construct()
    best_max = max(best)
    
    # Phase 2: Shortest-path search
    if time.time() < deadline - 30:
        sp_result = _search_shortest_path(deadline - 30, rng)
        if max(sp_result) < best_max:
            best = sp_result
            best_max = max(best)
    
    # Phase 3: Local search polish
    if time.time() < deadline - 5:
        polish_result = _local_search_polish(best, deadline - 5, rng)
        if max(polish_result) < best_max:
            best = polish_result
            best_max = max(best)
    
    return sorted(best)

# EVOLVE-BLOCK-END
