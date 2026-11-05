import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 31
    q = 5
    
    # Generate PG(2,5) points
    vectors = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                vectors.append((x, y, z))
                
    points = []
    used = set()
    for v in vectors:
        if v in used:
            continue
        for k in range(1, q):
            mult = ((v[0]*k)%q, (v[1]*k)%q, (v[2]*k)%q)
            used.add(mult)
        points.append(v)
        
    assert len(points) == 31
    
    # Polarity graph: 90 edges
    edges = set()
    for i in range(n):
        for j in range(i+1, n):
            u = points[i]
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                edges.add((i, j))
                
    if len(edges) != 90:
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
        
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def count_c4():
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                if j not in adj[i]:
                    c = len(adj[i] & adj[j])
                    total += c * (c-1) // 2
        return total // 2
    
    # Add best 91st edge (minimize common neighbors)
    best_edge = None
    min_common = n
    for i in range(n):
        for j in range(i+1, n):
            if j not in adj[i]:
                c = len(adj[i] & adj[j])
                if c < min_common:
                    min_common = c
                    best_edge = (i, j)
    
    if best_edge:
        u, v = best_edge
        edges.add(best_edge)
        adj[u].add(v)
        adj[v].add(u)
    
    current_c4 = count_c4()
    
    # Spectral-guided local search with limited trials
    random.seed(789)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        best_swap = None
        best_new_c4 = current_c4
        
        edge_list = list(edges)
        non_edges = [(i,j) for i in range(n) for j in range(i+1, n) if j not in adj[i]]
        
        max_trials = 50
        if len(edge_list) * len(non_edges) > max_trials:
            trials = random.sample([(e, ne) for e in edge_list for ne in non_edges], max_trials)
        else:
            trials = [(e, ne) for e in edge_list for ne in non_edges]
        
        for rem_edge, add_edge in trials:
            if time.time() > deadline - 15:
                break
                
            ru, rv = rem_edge
            au, av = add_edge
            
            if (au, av) == (ru, rv) or (au, av) == (rv, ru):
                continue
            if av in adj[au]:
                continue
                
            adj[ru].discard(rv)
            adj[rv].discard(ru)
            adj[au].add(av)
            adj[av].add(au)
            
            new_c4 = count_c4()
            
            adj[ru].add(rv)
            adj[rv].add(ru)
            adj[au].discard(av)
            adj[av].discard(au)
            
            if new_c4 < best_new_c4:
                best_new_c4 = new_c4
                best_swap = (rem_edge, add_edge)
        
        if best_swap is not None and best_new_c4 < current_c4:
            rem_edge, add_edge = best_swap
            ru, rv = rem_edge
            au, av = add_edge
            
            edges.discard(rem_edge)
            edges.discard((rv, ru))
            adj[ru].discard(rv)
            adj[rv].discard(ru)
            
            edges.add(add_edge)
            adj[au].add(av)
            adj[av].add(au)
            
            current_c4 = best_new_c4
            improved = True
    
    result = [[u, v] for u, v in edges]
    
    if len(result) != 91:
        while len(result) < 91:
            for i in range(n):
                for j in range(i+1, n):
                    if (i,j) not in edges:
                        result.append([i,j])
                        edges.add((i,j))
                        break
                if len(result) >= 91: break
            if len(result) >= 91: break
        result = result[:91]
    
    return result[:91]
