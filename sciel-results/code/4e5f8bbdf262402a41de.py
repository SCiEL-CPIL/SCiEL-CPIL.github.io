import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    # Pre-compute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Use integer bitmasks for adjacency
    # adj[i] is an integer where bit j is 1 if edge (i,j) exists
    
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid_adj(adj):
        for i in range(n):
            for j in range(i + 1, n):
                if popcount(adj[i] & adj[j]) > 2:
                    return False
        return True
        
    def can_add(adj, u, v):
        if (adj[u] >> v) & 1:
            return False
        # Check pair (u, v)
        if popcount(adj[u] & adj[v]) > 2:
            return False
        # Check pairs (u, w) for w in N(v)
        nv = adj[v]
        w = 0
        while w < n:
            if (nv >> w) & 1 and w != u:
                # New common neighbors of (u, w) will include v
                # Old common neighbors: adj[u] & adj[w]
                # If v is in adj[w] (which it is, since w in N(v)), 
                # and v is NOT in adj[u] (before adding), 
                # then new count = popcount(adj[u] & adj[w]) + 1
                # So we need popcount(adj[u] & adj[w]) <= 1
                if popcount(adj[u] & adj[w]) > 1:
                    return False
            w += 1
        # Check pairs (v, w) for w in N(u)
        nu = adj[u]
        w = 0
        while w < n:
            if (nu >> w) & 1 and w != v:
                if popcount(adj[v] & adj[w]) > 1:
                    return False
            w += 1
        return True

    def get_edge_count(adj):
        count = 0
        for i in range(n):
            count += popcount(adj[i])
        return count // 2

    best_edges = 0
    best_adj = None
    
    # Run multiple restarts or a long single run? 
    # A single long run with good swaps is usually sufficient for n=20.
    # Let's do a few restarts to be safe, or just one long run with aggressive swapping.
    
    for restart in range(10):
        if time.time() > deadline - 10:
            break
            
        # Initialize with a random sparse graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.3:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Ensure validity
        while not is_valid_adj(adj):
            # Find a violating pair and remove an edge
            removed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if popcount(adj[i] & adj[j]) > 2:
                        # Remove a random edge incident to i or j that is part of the common neighborhood?
                        # Simpler: just remove a random edge from the graph until valid
                        # This is inefficient. Better: remove an edge that causes the violation.
                        # The violation is caused by 3 common neighbors. 
                        # Let's just remove a random edge from the graph.
                        existing = []
                        for a in range(n):
                            for b in range(a+1, n):
                                if (adj[a] >> b) & 1:
                                    existing.append((a,b))
                        if existing:
                            u, v = random.choice(existing)
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            removed = True
                            break
                if removed:
                    break
            if not removed:
                break # Should not happen if graph is not empty, but safety
        
        current_edges = get_edge_count(adj)
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
        
        # Local Search
        T = 1.0
        no_improvement = 0
        
        while time.time() < deadline - 5:
            if no_improvement > 1000:
                break # Stuck, restart or accept
                
            # Try adding an edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and not ((adj[u] >> v) & 1):
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                    no_improvement = 0
                else:
                    no_improvement += 1
            else:
                # Try swapping
                # Find a random edge to remove
                existing = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            existing.append((i, j))
                if not existing:
                    break
                
                rem_u, rem_v = random.choice(existing)
                # Try to find an edge to add
                added = False
                for _ in range(20):
                    add_u = random.randint(0, n - 1)
                    add_v = random.randint(0, n - 1)
                    if add_u == add_v:
                        continue
                    if (add_u, add_v) == (rem_u, rem_v) or (add_u, add_v) == (rem_v, rem_u):
                        continue
                    if (adj[add_u] >> add_v) & 1:
                        continue
                    
                    # Remove rem edge
                    adj[rem_u] &= ~(1 << rem_v)
                    adj[rem_v] &= ~(1 << rem_u)
                    
                    if can_add(adj, add_u, add_v):
                        adj[add_u] |= (1 << add_v)
                        adj[add_v] |= (1 << add_u)
                        added = True
                        break
                    else:
                        # Revert
                        adj[rem_u] |= (1 << rem_v)
                        adj[rem_v] |= (1 << rem_u)
                
                if added:
                    no_improvement = 0
                else:
                    no_improvement += 1
            
            T *= 0.999
            if T < 0.1:
                T = 0.1

    # Extract edges from best_adj
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
