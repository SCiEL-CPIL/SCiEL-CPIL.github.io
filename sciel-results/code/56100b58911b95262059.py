# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a maximum (or near-maximum) square-difference-free subset of Z/205Z.
    
    Uses randomized greedy initialization followed by local search (swap moves)
    with multiple restarts. Includes a deadline guard to ensure completion within 600s.
    """
    m = 205
    deadline = time.time() + 580  # safety margin before 630s limit
    
    # Precompute forbidden differences: nonzero quadratic residues mod 205
    forbidden = set()
    for x in range(1, m):
        d = (x * x) % m
        if d != 0:
            forbidden.add(d)
    
    # For fast membership check, use a boolean array
    forbidden_arr = np.zeros(m, dtype=bool)
    for d in forbidden:
        forbidden_arr[d] = True
    
    def is_valid_addition(candidate, selected_set, selected_list):
        """Check if candidate can be added to selected_list without conflict."""
        for s in selected_list:
            d1 = (candidate - s) % m
            d2 = (s - candidate) % m
            if forbidden_arr[d1] or forbidden_arr[d2]:
                return False
        return True
    
    def greedy_init(rng):
        """Randomized greedy initialization."""
        candidates = list(range(m))
        rng.shuffle(candidates)
        selected = []
        selected_set = set()
        
        for cand in candidates:
            if is_valid_addition(cand, selected_set, selected):
                selected.append(cand)
                selected_set.add(cand)
        
        return selected
    
    def local_search(initial_selected, rng, max_iterations=50000):
        """Local search using swap moves: remove one element, add another."""
        selected = list(initial_selected)
        selected_set = set(selected)
        best_size = len(selected)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try swap moves: for each element in selected, try to remove it
            # and add a candidate not in selected
            n = len(selected)
            if n == 0:
                break
                
            # Try all possible removals
            for i in range(n):
                if time.time() > deadline:
                    break
                    
                removed = selected[i]
                remaining = [s for j, s in enumerate(selected) if j != i]
                remaining_set = set(remaining)
                
                # Try to add each candidate not in remaining
                for cand in range(m):
                    if cand in remaining_set:
                        continue
                    
                    # Check if cand is compatible with remaining
                    valid = True
                    for s in remaining:
                        d1 = (cand - s) % m
                        d2 = (s - cand) % m
                        if forbidden_arr[d1] or forbidden_arr[d2]:
                            valid = False
                            break
                    
                    if valid:
                        # This swap improves or maintains size (we removed 1, added 1)
                        # But we want to INCREASE size, so this is a lateral move
                        # For size increase, we need to add without removing
                        # Let's also try pure additions
                        pass
            
            # Try pure additions: can we add any element not in selected?
            for cand in range(m):
                if cand in selected_set:
                    continue
                if is_valid_addition(cand, selected_set, selected):
                    selected.append(cand)
                    selected_set.add(cand)
                    best_size = len(selected)
                    improved = True
                    break  # restart search after improvement
            
            if not improved:
                # Try swap moves to potentially enable future additions
                # Randomly pick an element to remove and try to add a different one
                i = rng.randint(0, n - 1)
                removed = selected[i]
                remaining = [s for j, s in enumerate(selected) if j != i]
                remaining_set = set(remaining)
                
                # Find a candidate that is compatible with remaining but not with selected
                # This allows us to change the set composition
                found_swap = False
                for cand in range(m):
                    if cand == removed:
                        continue
                    if cand in remaining_set:
                        continue
                    
                    # Check compatibility with remaining
                    valid = True
                    for s in remaining:
                        d1 = (cand - s) % m
                        d2 = (s - cand) % m
                        if forbidden_arr[d1] or forbidden_arr[d2]:
                            valid = False
                            break
                    
                    if valid:
                        # Check that cand was NOT compatible with the full selected set
                        # (otherwise it would have been added already)
                        was_incompatible = False
                        for s in selected:
                            if s == removed:
                                continue
                            d1 = (cand - s) % m
                            d2 = (s - cand) % m
                            if forbidden_arr[d1] or forbidden_arr[d2]:
                                was_incompatible = True
                                break
                        
                        # Even if it was compatible, we might still want to swap
                        # to change the structure. Let's just do the swap.
                        selected = remaining + [cand]
                        selected_set = set(selected)
                        found_swap = True
                        break
                
                if not found_swap:
                    # No swap found, try another iteration
                    pass
        
        return selected
    
    # Main search loop with multiple restarts
    best_solution = []
    rng = random.Random(42)
    
    # First, try a few deterministic greedy inits
    for seed in range(10):
        rng2 = random.Random(seed)
        sol = greedy_init(rng2)
        if len(sol) > len(best_solution):
            best_solution = sol
    
    # Now do local search from the best greedy solution
    if len(best_solution) > 0:
        sol = local_search(best_solution, rng, max_iterations=20000)
        if len(sol) > len(best_solution):
            best_solution = sol
    
    # Multiple restarts with local search
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        sol = greedy_init(rng)
        sol = local_search(sol, rng, max_iterations=10000)
        if len(sol) > len(best_solution):
            best_solution = sol
    
    # Final: try to add any remaining elements to the best solution
    final_selected = list(best_solution)
    final_set = set(final_selected)
    for cand in range(m):
        if cand not in final_set:
            if is_valid_addition(cand, final_set, final_selected):
                final_selected.append(cand)
                final_set.add(cand)
    
    return np.array(final_selected, dtype=np.int64)


# EVOLVE-BLOCK-END
