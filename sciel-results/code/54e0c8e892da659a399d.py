import time
import random
import itertools

def has_c5_fast(adj, n):
    """Check if graph has a C5 subgraph using adjacency list."""
    # Check all 5-cycles: for each starting vertex, try all 4-tuples of neighbors
    # A 5-cycle: v0-v1-v2-v3-v4-v0
    # Fix v0, then v1 in adj[v0], v2 in adj[v1] \ {v0}, v3 in adj[v2] \ {v0,v1}, v4 in adj[v3] \ {v0,v1,v2} and v4 in adj[v0]
    for v0 in range(n):
        for v1 in adj[v0]:
            for v2 in adj[v1]:
                if v2 == v0:
                    continue
                for v3 in adj[v2]:
                    if v3 == v0 or v3 == v1:
                        continue
                    for v4 in adj[v3]:
                        if v4 == v0 or v4 == v1 or v4 == v2:
                            continue
                        if v4 in adj[v0] and v4 != v1:
                            return True
    return False

def alpha_size_fast(adj, n):
    """Find max independent set size using branch and bound."""
    # Use a greedy upper bound: for each vertex, 1 + alpha of non-neighbors
    # Simple: try all subsets for small n, or use heuristic
    if n <= 20:
        max_alpha = 0
        for mask in range(1 << n):
            is_independent = True
            for i in range(n):
                if mask & (1 << i):
                    for j in range(i+1, n):
                        if mask & (1 << j):
                            if j in adj[i]:
                                is_independent = False
                                break
                    if not is_independent:
                        break
            if is_independent:
                cnt = bin(mask).count('1')
                if cnt > max_alpha:
                    max_alpha = cnt
        return max_alpha
    else:
        # Greedy heuristic for larger n
        # Use a simple greedy: repeatedly pick vertex with min degree, remove it and its neighbors
        # This gives a lower bound; for upper bound, use LP relaxation or just return a guess
        # For our purposes, we'll use a simple branch and bound
        remaining = set(range(n))
        best = 0
        def branch(current, current_size):
            nonlocal best
            if not remaining:
                best = max(best, current_size)
                return
            # Upper bound: current_size + len(remaining)
            if current_size + len(remaining) <= best:
                return
            # Pick a vertex
            v = next(iter(remaining))
            # Option 1: include v
            remaining.remove(v)
            neighbors_to_remove = set(adj[v]) & remaining
            remaining -= neighbors_to_remove
            branch(current + 1, current_size + 1)
            # Restore
            remaining.add(v)
            remaining |= neighbors_to_remove
            # Option 2: exclude v
            remaining.remove(v)
            branch(current, current_size)
            remaining.add(v)
        
        branch(0, 0)
        return best

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Start with the known 32-vertex solution: 8 disjoint K4s
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    best_n = 32
    best_edges = edges[:]
    
    # Try to find a 33-vertex C5-free graph with alpha <= 8
    # Strategy: random search with p=0.3
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        for trial in range(500):
            if time.time() > deadline - 10:
                break
            n_try = 33
            adj = [set() for _ in range(n_try)]
            edge_list = []
            for u in range(n_try):
                for v in range(u+1, n_try):
                    if random.random() < 0.3:
                        adj[u].add(v)
                        adj[v].add(u)
                        edge_list.append([u, v])
            
            # Check C5
            if has_c5_fast(adj, n_try):
                continue
            
            # Check alpha
            alpha = alpha_size_fast(adj, n_try)
            if alpha <= 8:
                best_n = n_try
                best_edges = edge_list
                break
    
    return {'n': best_n, 'edges': best_edges}
