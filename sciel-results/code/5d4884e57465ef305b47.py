import time
from fractions import Fraction
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85  # 85 second budget, 5s margin

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocal fractions
    recips = [Fraction(1, d) for d in denominators]
    
    # Greedy initialization: sort by 1/d descending, add if sum < 1
    order = sorted(range(n), key=lambda i: -recips[i])
    selected = set()
    total = Fraction(0)
    for idx in order:
        if total + recips[idx] < 1:
            selected.add(idx)
            total += recips[idx]
    
    best_total = total
    best_selected = set(selected)
    best_deficit = 1 - total
    
    def current_deficit():
        return 1 - sum(recips[i] for i in best_selected)
    
    # Local search with time budget
    search_deadline = start + 80
    
    # Helper: try to improve by swapping one out and one in
    def try_swap():
        nonlocal best_total, best_selected, best_deficit
        improved = False
        sel_list = list(best_selected)
        unsel_list = [i for i in range(n) if i not in best_selected]
        
        for rem in sel_list:
            rem_r = recips[rem]
            new_total = best_total - rem_r
            for add in unsel_list:
                add_r = recips[add]
                if new_total + add_r < 1:
                    new_def = 1 - (new_total + add_r)
                    if new_def < best_deficit:
                        best_deficit = new_def
                        best_total = new_total + add_r
                        best_selected.discard(rem)
                        best_selected.add(add)
                        improved = True
                        break
                if time.time() > search_deadline:
                    return True
            if time.time() > search_deadline:
                return True
        return improved
    
    # Try adding items that fit
    def try_add():
        nonlocal best_total, best_selected, best_deficit
        improved = False
        unsel_list = [i for i in range(n) if i not in best_selected]
        for add in unsel_list:
            add_r = recips[add]
            if best_total + add_r < 1:
                new_def = 1 - (best_total + add_r)
                if new_def < best_deficit:
                    best_deficit = new_def
                    best_total += add_r
                    best_selected.add(add)
                    improved = True
            if time.time() > search_deadline:
                break
        return improved
    
    # Iterate: try swaps and adds until no improvement or time runs out
    while time.time() < search_deadline:
        improved = False
        # Try multiple rounds of swaps
        for _ in range(5):
            if try_swap():
                improved = True
            if time.time() > search_deadline:
                break
        if time.time() > search_deadline:
            break
        # Try adds
        if try_add():
            improved = True
        if not improved:
            break
    
    # Advanced: try 2-swap moves (remove 2, add 2) if time permits
    if time.time() < search_deadline - 5:
        sel_list = list(best_selected)
        unsel_list = [i for i in range(n) if i not in best_selected]
        # Only try if sets are small enough
        if len(sel_list) <= 15 and len(unsel_list) <= 20:
            for rem1 in sel_list:
                if time.time() > search_deadline - 2:
                    break
                for rem2 in sel_list:
                    if rem2 <= rem1:
                        continue
                    rem_sum = recips[rem1] + recips[rem2]
                    new_total = best_total - rem_sum
                    for add1 in unsel_list:
                        if add1 in best_selected:
                            continue
                        if time.time() > search_deadline - 2:
                            break
                        for add2 in unsel_list:
                            if add2 <= add1 or add2 in best_selected:
                                continue
                            add_sum = recips[add1] + recips[add2]
                            if new_total + add_sum < 1:
                                new_def = 1 - (new_total + add_sum)
                                if new_def < best_deficit:
                                    best_deficit = new_def
                                    best_total = new_total + add_sum
                                    best_selected.discard(rem1)
                                    best_selected.discard(rem2)
                                    best_selected.add(add1)
                                    best_selected.add(add2)
    
    # Final verification
    final_total = sum(recips[i] for i in best_selected)
    if final_total < 1 and len(best_selected) > 0:
        return list(best_selected)
    else:
        # Fallback
        return [0]
