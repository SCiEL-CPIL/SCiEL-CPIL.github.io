"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin from 60s limit
    
    # Precompute unit fractions
    unit_fracs = [Fraction(1, d) for d in denominators]
    
    # Greedy initial solution: add fractions in order if sum stays < 1
    selected = []
    total = Fraction(0)
    for i in range(n):
        if total + unit_fracs[i] < 1:
            selected.append(i)
            total += unit_fracs[i]
    
    best_deficit = 1 - total
    best_indices = selected[:]
    
    # Local search with 1-swap and 2-swap moves
    selected_set = set(selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1-swap: remove one from selected, add one from unselected
        for i_out in list(selected):
            for i_in in range(n):
                if i_in in selected_set:
                    continue
                
                new_total = total - unit_fracs[i_out] + unit_fracs[i_in]
                
                if new_total < 1 and new_total > total:
                    total = new_total
                    best_deficit = 1 - total
                    selected.remove(i_out)
                    selected.append(i_in)
                    selected.sort()
                    best_indices = selected[:]
                    selected_set.remove(i_out)
                    selected_set.add(i_in)
                    improved = True
                    break
            
            if improved:
                break
        
        if improved:
            continue
        
        # 2-swap: remove two from selected, add two from unselected
        if time.time() < deadline:
            sel_list = list(selected)
            unselected = [i for i in range(n) if i not in selected_set]
            
            for idx1 in range(len(sel_list)):
                if time.time() >= deadline:
                    break
                for idx2 in range(idx1 + 1, len(sel_list)):
                    if time.time() >= deadline:
                        break
                    i_out1 = sel_list[idx1]
                    i_out2 = sel_list[idx2]
                    base_total = total - unit_fracs[i_out1] - unit_fracs[i_out2]
                    
                    for i_in1 in unselected:
                        if time.time() >= deadline:
                            break
                        for i_in2 in unselected:
                            if i_in2 <= i_in1:
                                continue
                            
                            new_total = base_total + unit_fracs[i_in1] + unit_fracs[i_in2]
                            
                            if new_total < 1 and new_total > total:
                                total = new_total
                                best_deficit = 1 - total
                                selected = [i for i in selected if i != i_out1 and i != i_out2]
                                selected.append(i_in1)
                                selected.append(i_in2)
                                selected.sort()
                                best_indices = selected[:]
                                selected_set.discard(i_out1)
                                selected_set.discard(i_out2)
                                selected_set.add(i_in1)
                                selected_set.add(i_in2)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return best_indices
# EVOLVE-BLOCK-END
