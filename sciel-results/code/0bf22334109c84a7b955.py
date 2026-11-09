"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin from 60s limit
    
    # Greedy initial solution
    selected, total = [], Fraction(0)
    for i, d in enumerate(denominators):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    
    best_deficit = 1 - total
    best_indices = selected[:]
    
    # Local search: try to swap out a selected index for an unselected one
    # to reduce the deficit (i.e., increase the sum while keeping it < 1)
    
    # Precompute unit fractions
    unit_fracs = [Fraction(1, d) for d in denominators]
    
    selected_set = set(selected)
    unselected = [i for i in range(n) if i not in selected_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try all possible swaps: remove one from selected, add one from unselected
        # Sort selected by index, unselected by index
        for i_out in selected[:]:
            for i_in in unselected[:]:
                if i_out == i_in:
                    continue
                
                new_total = total - unit_fracs[i_out] + unit_fracs[i_in]
                
                if new_total < 1 and new_total > total:
                    # This swap improves the sum (reduces deficit)
                    total = new_total
                    best_deficit = 1 - total
                    best_indices = [i for i in selected if i != i_out] + [i_in]
                    best_indices.sort()
                    
                    selected_set.remove(i_out)
                    selected_set.add(i_in)
                    
                    selected = best_indices[:]
                    unselected = [i for i in range(n) if i not in selected_set]
                    
                    improved = True
                    break
            
            if improved:
                break
    
    return best_indices
# EVOLVE-BLOCK-END
