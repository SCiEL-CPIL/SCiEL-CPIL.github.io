"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys

N = 192
MAX_K = 48
K = 3  # number of distinct other summands


def _is_valid(selected, x):
    """Check if x can be added to selected without violating the condition.
    
    Condition: x must NOT divide the sum of any 3 DISTINCT OTHER selected integers.
    """
    if len(selected) < K:
        return True
    
    # Build set for O(1) lookups
    sel_set = set(selected)
    n = len(selected)
    
    # For each triple of distinct elements from selected, check if sum % x == 0
    # But we can optimize: for each pair (a, b), compute a+b, then check if there exists c in selected
    # such that (a + b + c) % x == 0, i.e., c % x == (-a-b) % x
    
    # Group elements by their residue mod x
    residue_map = {}
    for s in selected:
        r = s % x
        residue_map.setdefault(r, []).append(s)
    
    # For each pair (a, b) in selected, a != b:
    # We need c in selected, c != a, c != b, such that (a + b + c) % x == 0
    # => c % x == (-a - b) % x
    
    # To be efficient, iterate over pairs and check residue_map
    # But pairs are O(n^2), and for each pair we do O(1) lookup
    
    # Actually, let's think more carefully. We need to check if there exist DISTINCT a, b, c in selected
    # (all different from each other) such that (a+b+c) % x == 0.
    
    # For each residue r, we know how many elements have that residue.
    # We need to find triples of residues (r1, r2, r3) such that (r1+r2+r3) % x == 0,
    # and the corresponding elements are distinct.
    
    # This is complex. Let's just do the direct check but optimized:
    # For each pair (i, j) with i < j:
    #   target_residue = (-selected[i] - selected[j]) % x
    #   Need an element c in selected with c % x == target_residue, c != selected[i], c != selected[j]
    
    sel = selected
    for i in range(n):
        for j in range(i + 1, n):
            target_res = (-sel[i] - sel[j]) % x
            # Check if there's an element in sel with residue target_res that is not sel[i] and not sel[j]
            elems_with_res = residue_map.get(target_res, [])
            for c in elems_with_res:
                if c != sel[i] and c != sel[j]:
                    return False  # Found a violating triple
    
    return True


def _greedy(order, deadline):
    """Greedy construction using a given order of candidates."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if len(selected) >= MAX_K:
            break
        if _is_valid(selected, x):
            selected.append(x)
    return selected


def _local_search(selected, deadline, max_iterations=2000):
    """Try to improve the set by swapping out one element for another."""
    if not selected:
        return selected
    
    best = list(selected)
    best_set = set(best)
    remaining = [x for x in range(1, N + 1) if x not in best_set]
    
    iteration = 0
    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        improved = False
        
        for idx in range(len(best)):
            if time.time() > deadline:
                return best
            # Try removing best[idx] and adding each remaining element
            trial_base = best[:idx] + best[idx+1:]
            trial_set = set(trial_base)
            
            for candidate in remaining:
                if candidate in trial_set:
                    continue
                # Check if candidate is valid with trial_base
                if _is_valid(trial_base, candidate):
                    # This swap improves or maintains the set (same size, but different composition)
                    # For local search, we want to explore. Let's do a random restart style.
                    # Actually, let's do hill-climbing: only accept if it allows further additions?
                    # Simpler: do random swaps and check validity.
                    new_set = trial_base + [candidate]
                    new_set.sort()
                    best = new_set
                    best_set = set(best)
                    remaining = [x for x in range(1, N + 1) if x not in best_set]
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # No single swap improves; try a random perturbation
            if len(remaining) > 0 and len(best) > 0:
                # Remove a random element from best
                rem_idx = random.randint(0, len(best) - 1)
                removed = best[rem_idx]
                trial_base = best[:rem_idx] + best[rem_idx+1:]
                
                # Try to add a random candidate
                candidate = random.choice(remaining)
                if _is_valid(trial_base, candidate):
                    best = trial_base + [candidate]
                    best.sort()
                    best_set = set(best)
                    remaining = [x for x in range(1, N + 1) if x not in best_set]
    
    # After local search, try to add more elements greedily
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        if len(best) >= MAX_K:
            break
        if x not in best_set:
            if _is_valid(best, x):
                best.append(x)
                best_set.add(x)
                best.sort()
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    best_selected = []
    
    # Order 1: Descending
    order1 = list(range(N, 0, -1))
    result1 = _greedy(order1, deadline)
    best_selected = result1
    
    if time.time() > deadline:
        return best_selected
    
    # Order 2: Ascending
    order2 = list(range(1, N + 1))
    result2 = _greedy(order2, deadline)
    if len(result2) > len(best_selected):
        best_selected = result2
    
    if time.time() > deadline:
        return best_selected
    
    # Order 3: Multiple random shuffles
    random.seed(42)
    for attempt in range(100):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        result = _greedy(order, deadline)
        if len(result) > len(best_selected):
            best_selected = result
    
    if time.time() > deadline:
        return best_selected
    
    # Local search on the best found
    if time.time() < deadline:
        improved = _local_search(best_selected, deadline, max_iterations=3000)
        if len(improved) > len(best_selected):
            best_selected = improved
    
    return best_selected
# EVOLVE-BLOCK-END
