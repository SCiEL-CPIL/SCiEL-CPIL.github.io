import itertools
import sys

def construct():
    # Grötzsch graph (11 vertices, 20 edges)
    n = 16
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        
    c5 = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    for u, v in c5:
        add_edge(u, v)
        
    copy_edges = [
        (5, 1), (5, 4),
        (6, 0), (6, 2),
        (7, 1), (7, 3),
        (8, 2), (8, 4),
        (9, 3), (9, 0)
    ]
    for u, v in copy_edges:
        add_edge(u, v)
        
    for v in range(5, 10):
        add_edge(10, v)
        
    best_edges = -1
    best_adj = None
    import time
    start_time = time.time()
    timeout = 110.0 # Leave 10s margin
    
    def get_independent_sets(vertices, adj):
        indep_sets = []
        vs = list(vertices)
        m = len(vs)
        # Optimization: If m is large, we might want to limit, but m <= 15
        for mask in range(1 << m):
            subset = []
            is_indep = True
            for i in range(m):
                if mask & (1 << i):
                    subset.append(vs[i])
            for i in range(len(subset)):
                for j in range(i+1, len(subset)):
                    if subset[j] in adj[subset[i]]:
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
                indep_sets.append(subset)
        return indep_sets

    def backtrack(current_v, adj_copy):
        nonlocal best_edges, best_adj
        nonlocal start_time, timeout
        
        if time.time() - start_time > timeout:
            return
            
        if current_v == 16:
            edge_count = 0
            for u in range(16):
                edge_count += len(adj_copy[u])
            edge_count //= 2
            
            if edge_count > best_edges:
                best_edges = edge_count
                best_adj = [set(s) for s in adj_copy]
            return
        
        existing_vertices = list(range(current_v))
        indep_sets = get_independent_sets(existing_vertices, adj_copy)
        
        # Sort by size descending to find good solutions faster (for potential pruning if we had one)
        indep_sets.sort(key=len, reverse=True)
        
        # Optional: Limit search if too many sets
        # For now, try all.
        
        for iset in indep_sets:
            new_adj = [set(s) for s in adj_copy]
            new_adj[current_v] = set()
            
            for u in iset:
                new_adj[u].add(current_v)
                new_adj[current_v].add(u)
                
            backtrack(current_v + 1, new_adj)
            
            if time.time() - start_time > timeout:
                return

    initial_adj = [set(s) for s in adj]
    for i in range(11, 16):
        initial_adj[i] = set()
        
    backtrack(11, initial_adj)
    
    final_edges = []
    if best_adj:
        for u in range(16):
            for v in best_adj[u]:
                if u < v:
                    final_edges.append([u, v])
    else:
        for u in range(16):
            for v in adj[u]:
                if u < v:
                    final_edges.append([u, v])
                    
    return final_edges
