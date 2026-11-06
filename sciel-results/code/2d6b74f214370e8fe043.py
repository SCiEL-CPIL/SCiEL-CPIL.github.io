import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    edges_tuple = [tuple(sorted(e)) for e in edges]
    colors = [-1] * n
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for e in edges_tuple:
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
                
                if c0 == 4 and len(unassigned) == 1:
                    colors[unassigned[0]] = 1
                    changed = True
                elif c1 == 4 and len(unassigned) == 1:
                    colors[unassigned[0]] = 0
                    changed = True
        return True

    def backtrack(idx):
        if idx == n:
            return True
        
        v = idx
        if colors[v] != -1:
            return backtrack(idx + 1)
        
        colors[v] = 0
        if propagate():
            if backtrack(idx + 1):
                return True
        colors[v] = -1
        
        colors[v] = 1
        if propagate():
            if backtrack(idx + 1):
                return True
        colors[v] = -1
        
        return False

    if not propagate():
        return False
    
    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_k = len(best_edges)
    
    n = 14
    all_edges = list(combinations(range(n), 5))
    
    # Start from a higher k to increase chance of finding a non-2-colorable graph
    # and then potentially find smaller ones if we have time?
    # No, we want the minimum.
    # Let's iterate k from 35 to 51.
    
    for k in range(35, 52):
        if time.time() > deadline:
            break
            
        found = False
        while not found and time.time() < deadline:
            edges = random.sample(all_edges, k)
            if not is_2_colorable(n, edges):
                found = True
                if k < best_k:
                    best_k = k
                    best_n = n
                    best_edges = [list(e) for e in edges]
                break
        
    return {'n': best_n, 'edges': best_edges}
