"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    fractions = [Fraction(1, d) for d in denominators]
    
    # Greedy: include in order if sum stays < 1
    selected = set()
    total = Fraction(0)
    for i in range(n):
        if total + fractions[i] < 1:
            selected.add(i)
            total += fractions[i]
    
    best_selected = set(selected)
    best_total = total
    best_deficit = 1 - total
    
    # Local search: try to improve
    # Operations: add one, remove one, swap one
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_deficit = 1 - best_total
        
        # Try adding each unselected fraction
        for i in range(n):
            if i not in best_selected:
                new_total = best_total + fractions[i]
                if new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < best_deficit:
                        best_selected.add(i)
                        best_total = new_total
                        best_deficit = new_deficit
                        improved = True
        
        # Try removing each selected fraction (might allow adding larger ones later)
        # Actually, removing makes deficit worse, so skip unless it enables better swaps
        
        # Try swapping: remove one selected, add one unselected
        for i in range(n):
            if i in best_selected:
                for j in range(n):
                    if j not in best_selected:
                        new_total = best_total - fractions[i] + fractions[j]
                        if new_total < 1:
                            new_deficit = 1 - new_total
                            if new_deficit < best_deficit:
                                best_selected.remove(i)
                                best_selected.add(j)
                                best_total = new_total
                                best_deficit = new_deficit
                                improved = True
        
        # Also try multi-step improvements: add multiple small fractions
        # After swaps, try adding again
        for i in range(n):
            if i not in best_selected:
                new_total = best_total + fractions[i]
                if new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < best_deficit:
                        best_selected.add(i)
                        best_total = new_total
                        best_deficit = new_deficit
                        improved = True
    
    return sorted(list(best_selected))
# EVOLVE-BLOCK-END
