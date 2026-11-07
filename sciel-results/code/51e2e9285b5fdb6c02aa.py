import itertools
import time

def construct():
    start_time = time.time()
    deadline = 85
    
    base_elements = [0, 1, 2, 3, 4]
    base_sets = []
    for r in range(6):
        for combo in itertools.combinations(base_elements, r):
            base_sets.append(frozenset(combo))
    
    all_projections = list(base_sets)
    
    def total_score(projs):
        base_ext = sum(2**len(p) - 1 for p in projs)
        ext_ext = 0
        for i in range(len(projs)):
            for j in range(i+1, len(projs)):
                if projs[i] < projs[j] or projs[j] < projs[i]:
                    ext_ext += 1
        return base_ext + ext_ext
    
    selected = []
    remaining = list(all_projections)
    for _ in range(8):
        if time.time() - start_time > deadline - 10:
            break
        best_proj = None
        best_score = -1
        for p in remaining:
            s = total_score(selected + [p])
            if s > best_score:
                best_score = s
                best_proj = p
        selected.append(best_proj)
        remaining.remove(best_proj)
    
    best_selected = list(selected)
    best_score = total_score(best_selected)
    
    for iteration in range(1000):
        if time.time() - start_time > deadline - 5:
            break
        improved = False
        for i in range(8):
            for p in all_projections:
                if p in best_selected:
                    continue
                new_sel = list(best_selected)
                new_sel[i] = p
                s = total_score(new_sel)
                if s > best_score:
                    best_score = s
                    best_selected = new_sel
                    improved = True
        if not improved:
            break
    
    extension_sets = [list(p) + [5] for p in best_selected]
    result = [list(s) for s in base_sets] + extension_sets
    
    return result
