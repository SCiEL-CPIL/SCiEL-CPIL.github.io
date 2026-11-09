import time
import random
import math

def construct():
    n = 15
    start = time.time()
    timeout = 55.0
    
    # Precompute all simple paths of length 3
    paths = []
    for s in range(n):
        for u in range(n):
            if s == u: continue
            for v in range(n):
                if v == s or v == u: continue
                for t in range(n):
                    if t == s or t == u or t == v: continue
                    paths.append((s, u, v, t))
                    
    # Precompute canonical edge tuples for paths to avoid min/max in loop
    # path_edges[i] = (e1, e2, e3)
    path_edges = []
    for s, u, v, t in paths:
        e1 = (min(s,u), max(s,u))
        e2 = (min(u,v), max(u,v))
        e3 = (min(v,t), max(v,t))
        path_edges.append((e1, e2, e3))
        
    def calculate_violations(edges_set):
        deg = [0] * n
        for u, v in edges_set:
            deg[u] += 1
            deg[v] += 1
            
        violations = 0
        for i in range(len(paths)):
            e1, e2, e3 = path_edges[i]
            if e1 in edges_set and e2 in edges_set and e3 in edges_set:
                s, u, v, t = paths[i]
                if deg[s] == deg[t]:
                    violations += 1
        return violations

    # Initial state: K_{7,8}
    L = list(range(7))
    R = list(range(7, 15))
    initial_edges = set()
    for u in L:
        for v in R:
            initial_edges.add((u, v))
            
    current_edges = set(initial_edges)
    current_violations = calculate_violations(current_edges)
    
    best_valid_edges = set(current_edges)
    best_valid_count = len(current_edges)
    
    T = 5.0
    T_min = 0.001
    cooling = 0.9995
    
    all_edges_list = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() - start < timeout and T > T_min:
        idx = random.randint(0, len(all_edges_list) - 1)
        e = all_edges_list[idx]
        
        if e in current_edges:
            new_edges = current_edges - {e}
        else:
            new_edges = current_edges | {e}
        
        new_violations = calculate_violations(new_edges)
        
        if new_violations == 0:
            if len(new_edges) > best_valid_count:
                best_valid_count = len(new_edges)
                best_valid_edges = set(new_edges)
                
        old_energy = 1000 * current_violations - len(current_edges)
        new_energy = 1000 * new_violations - len(new_edges)
        
        delta = new_energy - old_energy
        
        if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
            current_edges = new_edges
            current_violations = new_violations
            
        T *= cooling
        
    return [[u, v] for u, v in best_valid_edges]
