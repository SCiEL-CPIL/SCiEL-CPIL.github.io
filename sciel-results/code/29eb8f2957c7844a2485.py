import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 12
    m = 40
    
    # Use 6-element ground set for maximum density
    k = 6
    
    # Generate all subsets as frozensets
    all_subsets = [frozenset(i for i in range(k) if (mask >> i) & 1) for mask in range(1 << k)]
    n_sets = len(all_subsets)  # 64
    
    # Precompute comparability
    comparable = [set() for _ in range(n_sets)]
    for i in range(n_sets):
        for j in range(i+1, n_sets):
            if all_subsets[i] < all_subsets[j] or all_subsets[j] < all_subsets[i]:
                comparable[i].add(j)
                comparable[j].add(i)
    
    def count_comparables(selected):
        count = 0
        sel_list = list(selected)
        for i in range(len(sel_list)):
            for j in range(i+1, len(sel_list)):
                if sel_list[i] in comparable[sel_list[j]] or sel_list[j] in comparable[sel_list[i]]:
                    count += 1
        return count
    
    # Greedy selection
    selected = set([0])
    
    while len(selected) < m:
        best_idx = -1
        best_gain = -1
        for idx in range(n_sets):
            if idx in selected:
                continue
            gain = len(comparable[idx] & selected)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx == -1:
            break
        selected.add(best_idx)
    
    # Local search: try swapping one selected set with one unselected set
    if time.time() < deadline - 20:
        current_score = count_comparables(selected)
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            for idx_in in list(selected):
                for idx_out in range(n_sets):
                    if idx_out in selected:
                        continue
                    new_selected = set(selected)
                    new_selected.remove(idx_in)
                    new_selected.add(idx_out)
                    new_score = count_comparables(new_selected)
                    if new_score > current_score:
                        selected = new_selected
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
    
    # Convert to result
    result = [sorted(all_subsets[idx]) for idx in sorted(selected)]
    
    # Ensure exactly m sets
    if len(result) < m:
        used = set(frozenset(s) for s in result)
        for mask in range(1 << k):
            s = frozenset(i for i in range(k) if (mask >> i) & 1)
            if s not in used:
                result.append(sorted(s))
                used.add(s)
                if len(result) == m:
                    break
    
    # Final validation
    assert len(result) == m
    assert len(set(frozenset(s) for s in result)) == m
    result = [list(s) for s in result]
    
    return result
