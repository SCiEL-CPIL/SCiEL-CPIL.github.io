import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_m = 126
    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)] # Fallback: seed program
    
    def is_2_colorable(n, edges):
        colors = [-1] * n
        
        def propagate():
            changed = True
            while changed:
                changed = False
                for e in edges:
                    c0 = 0
                    c1 = 0
                    unassigned_idx = -1
                    for i, v in enumerate(e):
                        if colors[v] == 0:
                            c0 += 1
                        elif colors[v] == 1:
                            c1 += 1
                        else:
                            unassigned_idx = i
                            
                    if c0 == 5 or c1 == 5:
                        return False
                    if c0 == 4 and unassigned_idx != -1:
                        colors[e[unassigned_idx]] = 1
                        changed = True
                    elif c1 == 4 and unassigned_idx != -1:
                        colors[e[unassigned_idx]] = 0
                        changed = True
            return True

        def backtrack():
            v = -1
            for i in range(n):
                if colors[i] == -1:
                    v = i
                    break
            if v == -1:
                return True
            
            colors[v] = 0
            if propagate():
                if backtrack():
                    return True
            colors[v] = -1
            
            colors[v] = 1
            if propagate():
                if backtrack():
                    return True
            colors[v] = -1
            
            return False
        
        return backtrack()

    n_range = [10, 11, 12, 13]
    
    # We want to find the smallest m.
    # Start from 50.
    for m in range(50, 127):
        if m >= best_m:
            break
            
        for n in n_range:
            if time.time() > deadline:
                break
                
            max_edges = len(list(combinations(range(n), 5)))
            if m > max_edges:
                continue
                
            # Try 20 random instances
            for _ in range(20):
                if time.time() > deadline:
                    break
                all_edges = list(combinations(range(n), 5))
                random.shuffle(all_edges)
                edges = all_edges[:m]
                
                if not is_2_colorable(n, edges):
                    best_m = m
                    best_n = n
                    best_edges = [list(e) for e in edges]
                    break

    return {'n': best_n, 'edges': best_edges}
