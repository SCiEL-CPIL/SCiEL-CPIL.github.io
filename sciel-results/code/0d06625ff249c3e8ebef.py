"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    recip = [Fraction(1, d) for d in denominators]
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def deficit_of(total):
        return Fraction(1, 1) - total
    
    def is_valid(total):
        return total < Fraction(1, 1)
    
    def greedy_solution(order):
        """Greedy: add fractions in given order if sum stays < 1."""
        selected = set()
        total = Fraction(0, 1)
        for idx in order:
            new_total = total + recip[idx]
            if new_total < Fraction(1, 1):
                selected.add(idx)
                total = new_total
        return selected, total
    
    def local_search_single(selected, total, max_passes=200):
        """Single-swap local search: remove one, add one."""
        for _ in range(max_passes):
            if time_left() < 2.0:
                break
            improved = False
            # Try all single swaps: remove i, add j
            for i in list(selected):
                base = total - recip[i]
                for j in range(n):
                    if j in selected:
                        continue
                    new_total = base + recip[j]
                    if new_total < Fraction(1, 1):
                        # Check if this improves deficit
                        old_def = Fraction(1,1) - total
                        new_def = Fraction(1,1) - new_total
                        if new_def < old_def:
                            selected.remove(i)
                            selected.add(j)
                            total = new_total
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        return selected, total
    
    def local_search_double(selected, total, max_passes=50):
        """Double-swap local search: remove two, add two."""
        for _ in range(max_passes):
            if time_left() < 2.0:
                break
            improved = False
            best_new_def = Fraction(1,1) - total
            best_swap = None
            # Try all pairs to remove
            for i1, i2 in combinations(selected, 2):
                base = total - recip[i1] - recip[i2]
                # Find best pair to add
                for j1, j2 in combinations(range(n), 2):
                    if j1 in selected or j2 in selected:
                        continue
                    new_total = base + recip[j1] + recip[j2]
                    if new_total < Fraction(1, 1):
                        new_def = Fraction(1, 1) - new_total
                        if new_def < best_new_def:
                            best_new_def = new_def
                            best_swap = (i1, i2, j1, j2, new_total)
            if best_swap:
                i1, i2, j1, j2, new_total = best_swap
                selected.discard(i1)
                selected.discard(i2)
                selected.add(j1)
                selected.add(j2)
                total = new_total
                improved = True
            if not improved:
                break
        return selected, total
    
    # Generate initial solutions
    best_selected = None
    best_total = None
    best_def = None
    
    # Greedy: ascending order (largest fractions first)
    order_asc = list(range(n))
    sel, tot = greedy_solution(order_asc)
    if sel:
        d = Fraction(1,1) - tot
        if best_def is None or d < best_def:
            best_selected, best_total, best_def = set(sel), tot, d
    
    # Greedy: descending order
    order_desc = list(range(n-1, -1, -1))
    sel, tot = greedy_solution(order_desc)
    if sel:
        d = Fraction(1,1) - tot
        if best_def is None or d < best_def:
            best_selected, best_total, best_def = set(sel), tot, d
    
    # Random restarts for initial solutions
    import random
    random.seed(42)
    for _ in range(50):
        if time_left() < 5.0:
            break
        order = list(range(n))
        random.shuffle(order)
        sel, tot = greedy_solution(order)
        if sel:
            d = Fraction(1,1) - tot
            if best_def is None or d < best_def:
                best_selected, best_total, best_def = set(sel), tot, d
    
    # Local search on best solution
    if best_selected is not None:
        best_selected, best_total = local_search_single(best_selected, best_total)
        best_def = Fraction(1,1) - best_total
        
        # Double swap search
        if time_left() > 10.0:
            best_selected, best_total = local_search_double(best_selected, best_total)
            best_def = Fraction(1,1) - best_total
            
            # More single swaps after double
            best_selected, best_total = local_search_single(best_selected, best_total)
            best_def = Fraction(1,1) - best_total
    
    # If time remains, try more random restarts with local search
    for _ in range(200):
        if time_left() < 3.0:
            break
        order = list(range(n))
        random.shuffle(order)
        sel, tot = greedy_solution(order)
        if not sel:
            continue
        sel, tot = local_search_single(sel, tot, max_passes=50)
        d = Fraction(1,1) - tot
        if best_def is None or d < best_def:
            best_selected, best_total, best_def = set(sel), tot, d
    
    if best_selected is None or not best_selected:
        # Fallback: single element
        return [0]
    
    return list(best_selected)
# EVOLVE-BLOCK-END
