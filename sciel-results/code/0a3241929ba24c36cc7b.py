import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 16
    subsets = list(combinations(range(n), 8))
    
    # Precompute for each subset, the list of pairs of local indices that must be edges?
    # No, we check against the graph.
    
    def is_q3(subset, adj):
        # Check if induced subgraph on subset is Q3
        # 1. Edge count 12, Degree 3, Bipartite
        
        local_adj = [0] * 8
        edge_count = 0
        degrees = [0] * 8
        
        for i in range(8):
            v_i = subset[i]
            for j in range(i+1, 8):
                v_j = subset[j]
                if (adj[v_i] >> v_j) & 1:
                    local_adj[i] |= (1 << j)
                    local_adj[j] |= (1 << i)
                    edge_count += 1
                    degrees[i] += 1
                    degrees[j] += 1
                    
        if edge_count != 12:
            return False
            
        for d in degrees:
            if d != 3:
                return False
                
        # Check Bipartite
        # Color 0..7
        color = [-1] * 8
        color[0] = 0
        stack = [0]
        while stack:
            u = stack.pop()
            for v in range(8):
                if (local_adj[u] >> v) & 1:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        stack.append(v)
                    elif color[v] == color[u]:
                        return False
                        
        return True

    def get_q3s(adj):
        """Returns a list of edges (u,v) that are part of at least one Q3."""
        # Actually, we want to find one Q3 or all Q3s to determine critical edges.
        # For greedy deletion, we want to find the "most critical" edge.
        # So we should find ALL Q3s and count edge frequencies.
        
        q3_edges_count = {}
        for subset in subsets:
            if is_q3(subset, adj):
                # This subset is a Q3.
                # Add its edges to the counter
                # The edges are the pairs in the subset that exist in adj.
                # We can just iterate the subset pairs again?
                # Or we can return the subset and process later.
                # To save time, let's just collect the subsets that are Q3s.
                pass
                
        # This implementation is getting complex.
        # Simpler: Just find ONE Q3 and remove a random edge from it.
        # This is "Randomized Greedy Deletion".
        
        for subset in subsets:
            if is_q3(subset, adj):
                # Found a Q3.
                # Return one edge from it.
                # Let's return the first edge found in the subset.
                for i in range(8):
                    for j in range(i+1, 8):
                        v_i = subset[i]
                        v_j = subset[j]
                        if (adj[v_i] >> v_j) & 1:
                            return (v_i, v_j)
        return None

    best_adj = None
    best_count = 0
    
    trials = 0
    while time.time() < deadline - 5:
        trials += 1
        random.seed(trials)
        
        # Start with Complete Graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        count = 120
        
        # Remove edges until Q3-free
        while True:
            if time.time() > deadline - 2:
                break
                
            edge_to_remove = get_q3s(adj)
            if edge_to_remove is None:
                break
                
            u, v = edge_to_remove
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            count -= 1
            
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            
        if trials > 200:
            break

    if best_adj is None:
        return []
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges

construct()
