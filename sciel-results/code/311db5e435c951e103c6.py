import random
from itertools import combinations
import time

def is_2_colorable_fast(edges, n):
    if not edges:
        return True

    colors = [-1] * n
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for e in edges:
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    if colors[v] == 0:
                        c0 += 1
                    elif colors[v] == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                if c0 == 5 or c1 == 5:
                    return False
                
                if len(unassigned) == 1:
                    u = unassigned[0]
                    if c0 == 4:
                        if colors[u] != -1:
                            if colors[u] != 1:
                                return False
                        else:
                            colors[u] = 1
                            changed = True
                    elif c1 == 4:
                        if colors[u] != -1:
                            if colors[u] != 0:
                                return False
                        else:
                            colors[u] = 0
                            changed = True
        return True

    def solve(idx):
        while idx < n and colors[idx] != -1:
            idx += 1
        
        if idx == n:
            return True
        
        # Try 0
        colors[idx] = 0
        if propagate():
            if solve(idx + 1):
                return True
        colors[idx] = -1
        
        # Try 1
        colors[idx] = 1
        if propagate():
            if solve(idx + 1):
                return True
        colors[idx] = -1
        
        return False

    return solve(0)

def construct():
    n = 9
    all_edges = [tuple(e) for e in combinations(range(n), 5)]
    
    best_edges = all_edges
    best_count = len(all_edges)
    
    start_time = time.time()
    limit = 140.0
    
    for restart in range(100):
        if time.time() - start_time > limit:
            break
            
        current_edges = set(all_edges)
        
        edges_list = list(all_edges)
        random.shuffle(edges_list)
        
        for e in edges_list:
            if time.time() - start_time > limit:
                break
                
            if e not in current_edges:
                continue
                
            current_edges.discard(e)
            
            if is_2_colorable_fast(current_edges, n):
                current_edges.add(e)
            
        if len(current_edges) < best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            
            if best_count <= 51:
                break

    return {'n': n, 'edges': [list(e) for e in best_edges]}
