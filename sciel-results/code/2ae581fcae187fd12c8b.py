"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction
    
    deadline = time.time() + 85.0
    denominators = list(range(2, 49))
    n = len(denominators)
    inv_vals = [Fraction(1, d) for d in denominators]
    
    best_indices = None
    best_deficit = None
    
    # Start with the greedy solution as initial best
    selected, total = [], Fraction()
    for i, d in enumerate(range(2, 49)):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    if selected:
        best_indices = selected[:]
        best_deficit = 1 - total
    
    # Local search: try to improve by swapping elements
    # We want to minimize deficit, i.e., maximize sum while keeping sum < 1
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(best_indices)
        current_sum = sum(inv_vals[i] for i in best_indices)
        
        # Try adding one element
        for i in range(n):
            if i in current_set:
                continue
            new_sum = current_sum + inv_vals[i]
            if new_sum < 1:
                new_deficit = 1 - new_sum
                if best_deficit is None or new_deficit < best_deficit:
                    best_indices = best_indices + [i]
                    best_deficit = new_deficit
                    improved = True
                    break
        
        if time.time() >= deadline:
            break
        
        # Try swapping: remove one, add one
        current_set = set(best_indices)
        current_sum = sum(inv_vals[i] for i in best_indices)
        
        for rem in list(best_indices):
            if time.time() >= deadline:
                break
            sum_without_rem = current_sum - inv_vals[rem]
            for add in range(n):
                if add in current_set and add != rem:
                    continue
                # If add is not in set, we can try adding it after removing rem
                # But we must ensure add != rem
                if add == rem:
                    continue
                new_sum = sum_without_rem + inv_vals[add]
                if new_sum < 1:
                    new_deficit = 1 - new_sum
                    if new_deficit < best_deficit:
                        # Update best
                        new_indices = [i for i in best_indices if i != rem]
                        if add not in new_indices:
                            new_indices.append(add)
                        best_indices = new_indices
                        best_deficit = new_deficit
                        improved = True
                        break
            if improved:
                break
    
    # Try a more systematic approach: random restarts with local search
    if time.time() < deadline:
        import random
        random.seed(42)
        
        for _ in range(100):
            if time.time() >= deadline:
                break
            
            # Random initialization
            size = random.randint(5, 25)
            indices = random.sample(range(n), size)
            
            # Greedy improvement: sort by value descending, try to add
            # Actually, let's just do local search from this random start
            current_indices = indices[:]
            current_sum = sum(inv_vals[i] for i in current_indices)
            
            if current_sum >= 1:
                # Remove largest elements until sum < 1
                sorted_indices = sorted(current_indices, key=lambda i: inv_vals[i], reverse=True)
                for i in sorted_indices:
                    if current_sum < 1:
                        break
                    current_indices.remove(i)
                    current_sum -= inv_vals[i]
            
            if not current_indices:
                continue
            
            cur_deficit = 1 - current_sum
            
            # Local search: add, remove, swap
            ls_improved = True
            while ls_improved and time.time() < deadline:
                ls_improved = False
                cur_set = set(current_indices)
                cur_sum = sum(inv_vals[i] for i in current_indices)
                cur_def = 1 - cur_sum
                
                # Try adding
                for i in range(n):
                    if i in cur_set:
                        continue
                    ns = cur_sum + inv_vals[i]
                    if ns < 1:
                        nd = 1 - ns
                        if nd < cur_def:
                            current_indices.append(i)
                            cur_def = nd
                            cur_sum = ns
                            ls_improved = True
                            break
                
                if time.time() >= deadline:
                    break
                
                # Try swapping
                if not ls_improved:
                    cur_set = set(current_indices)
                    for rem in current_indices:
                        if time.time() >= deadline:
                            break
                        s_wo = cur_sum - inv_vals[rem]
                        for add in range(n):
                            if add in cur_set and add != rem:
                                continue
                            if add == rem:
                                continue
                            ns = s_wo + inv_vals[add]
                            if ns < 1:
                                nd = 1 - ns
                                if nd < cur_def:
                                    new_idx = [x for x in current_indices if x != rem]
                                    if add not in new_idx:
                                        new_idx.append(add)
                                    current_indices = new_idx
                                    cur_def = nd
                                    cur_sum = ns
                                    ls_improved = True
                                    break
                        if ls_improved:
                            break
                
                if time.time() >= deadline:
                    break
            
            if cur_def < best_deficit:
                best_deficit = cur_def
                best_indices = current_indices[:]
    
    return best_indices
# EVOLVE-BLOCK-END
