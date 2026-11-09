import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin

    # 1. Construct the known 35-vertex graph
    def build_35():
        vertices = list(combinations(range(7), 3))
        n = len(vertices)
        idx_map = {v: i for i, v in enumerate(vertices)}
        adj = np.zeros((n, n), dtype=bool)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                u = vertices[i]
                v = vertices[j]
                inter_size = len(set(u).intersection(v))
                if inter_size != 1:
                    adj[i, j] = True
                    adj[j, i] = True
                    edges.append([i, j])
        return n, edges, adj

    n35, edges35, adj35 = build_35()
    
    # Verification of 35-vertex graph (fast)
    def is_valid(n, edges):
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edges:
            adj[u, v] = True
            adj[v, u] = True
        
        # Check K4
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i, j]: continue
                for k in range(j + 1, n):
                    if not adj[i, k] or not adj[j, k]: continue
                    for l in range(k + 1, n):
                        if adj[i, l] and adj[j, l] and adj[k, l]:
                            return False
                            
        # Check Independent Set of 6
        comp = ~adj
        np.fill_diagonal(comp, False)
        
        for i in range(n):
            for j in range(i + 1, n):
                if not comp[i, j]: continue
                for k in range(j + 1, n):
                    if not comp[i, k] or not comp[j, k]: continue
                    for l in range(k + 1, n):
                        if not comp[i, l] or not comp[j, l] or not comp[k, l]: continue
                        for m in range(l + 1, n):
                            if comp[i, m] and comp[j, m] and comp[k, m] and comp[l, m]:
                                for p in range(m + 1, n):
                                    if comp[i, p] and comp[j, p] and comp[k, p] and comp[l, p] and comp[m, p]:
                                        return False
        return True

    if is_valid(n35, edges35):
        best_result = {'n': n35, 'edges': edges35}
    else:
        # Fallback to smaller if 35 fails (should not)
        best_result = {'n': 1, 'edges': []}
        return best_result

    # 2. Try to find 36-vertex graph
    # Start with 35-graph + universal vertex
    n36 = 36
    adj36 = np.zeros((n36, n36), dtype=bool)
    adj36[:n35, :n35] = adj35
    # Connect new vertex 35 to all
    adj36[35, :] = True
    adj36[:, 35] = True
    np.fill_diagonal(adj36, False)

    def count_k4(adj):
        n = adj.shape[0]
        count = 0
        # Optimize: only check triangles
        # For each edge (u,v), count common neighbors w.
        # For each such w, count common neighbors of u,v,w? No.
        # Simpler: iterate over all 4-combinations? Too slow.
        # Use: K4 count = sum over all triangles (u,v,w) of (common neighbors of u,v,w) / 4?
        # Let's just use the nested loop for n=36, it's fast enough.
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i, j]: continue
                for k in range(j + 1, n):
                    if not adj[i, k] or not adj[j, k]: continue
                    for l in range(k + 1, n):
                        if adj[i, l] and adj[j, l] and adj[k, l]:
                            count += 1
        return count

    def count_ind4(adj):
        # Count independent sets of size 4
        # Equivalent to K4 in complement
        comp = ~adj
        np.fill_diagonal(comp, False)
        return count_k4(comp)

    def get_edges(adj):
        n = adj.shape[0]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return edges

    # Local Search
    best_score = float('inf')
    best_adj = adj36.copy()
    
    # Initial score
    current_adj = adj36.copy()
    current_score = count_k4(current_adj) + count_ind4(current_adj)
    
    # If initial is perfect, return
    if current_score == 0:
         # Verify exactly
         edges = get_edges(current_adj)
         if is_valid(n36, edges):
             return {'n': n36, 'edges': edges}

    best_adj = current_adj.copy()
    best_score = current_score

    t_start = time.time()
    
    # Simple Hill Climbing with Restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        current_adj = best_adj.copy()
        current_score = best_score
        
        # Simulated Annealing / Hill Climbing
        T = 1.0
        for step in range(500):
            if time.time() > deadline - 5:
                break
            
            # Pick random edge
            u = random.randint(0, n36 - 1)
            v = random.randint(0, n36 - 1)
            if u == v:
                continue
            
            # Flip
            current_adj[u, v] = not current_adj[u, v]
            current_adj[v, u] = current_adj[u, v]
            
            new_score = count_k4(current_adj) + count_ind4(current_adj)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj.copy()
                    if best_score == 0:
                        break
            else:
                # Reject (or accept with prob)
                current_adj[u, v] = not current_adj[u, v]
                current_adj[v, u] = current_adj[u, v]
                
        # Check best found so far for exact validity
        if best_score == 0:
            edges = get_edges(best_adj)
            if is_valid(n36, edges):
                return {'n': n36, 'edges': edges}

    # If 36 failed, return 35
    return best_result
