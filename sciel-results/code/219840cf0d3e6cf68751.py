import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s budget for 90s limit, safety margin
    
    N = 256
    # Candidates are 2..256
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # Precompute lcm conflicts: conflict[i][j] = True if lcm(candidates[i], candidates[j]) <= 256
    # Use index-based for speed
    # candidates[i] = i + 2
    
    # Build conflict matrix as bitsets for speed
    # conflict_mask[i] is a set of indices j where lcm conflicts
    conflict = [set() for _ in range(n)]
    for i in range(n):
        a = candidates[i]
        for j in range(i+1, n):
            b = candidates[j]
            if math.lcm(a, b) <= N:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Greedy: process in order of descending 1/x (ascending x)
    # Since candidates are already in ascending order, 1/x is descending
    selected = []
    selected_set = set()
    
    for i in range(n):
        # Check if i conflicts with any selected
        compatible = True
        for s in selected:
            if s in conflict[i]:
                compatible = False
                break
        if compatible:
            selected.append(i)
            selected_set.add(i)
    
    def compute_sum(idx_list):
        return sum(1.0 / candidates[i] for i in idx_list)
    
    current_sum = compute_sum(selected)
    
    # Local search: try swaps
    # For each selected element, try replacing it with an unselected element
    # if the new element is compatible with all other selected elements
    # and the sum improves
    
    unselected = [i for i in range(n) if i not in selected_set]
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for si in range(len(selected)):
            old_idx = selected[si]
            # Try replacing old_idx with each unselected
            best_new = None
            best_gain = 0.0
            old_val = 1.0 / candidates[old_idx]
            
            for new_idx in unselected:
                # Check if new_idx is compatible with all selected except old_idx
                compatible = True
                for sj in range(len(selected)):
                    if sj == si:
                        continue
                    if selected[sj] in conflict[new_idx]:
                        compatible = False
                        break
                if not compatible:
                    continue
                new_val = 1.0 / candidates[new_idx]
                gain = new_val - old_val
                if gain > best_gain:
                    best_gain = gain
                    best_new = new_idx
            
            if best_new is not None and best_gain > 1e-15:
                # Perform the swap
                selected[si] = best_new
                selected_set.remove(old_idx)
                selected_set.add(best_new)
                unselected.remove(best_new)
                unselected.append(old_idx)
                current_sum += best_gain
                improved = True
    
    # Try adding more elements if possible (though greedy should have covered this)
    # Actually, greedy already tried all in order, so no more can be added without removing
    
    # Return the actual numbers, not indices
    result = [candidates[i] for i in selected]
    return result
