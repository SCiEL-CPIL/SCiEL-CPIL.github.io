"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    n = 16
    # Start with Clebsch graph
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            xor = u ^ v
            bits = bin(xor).count('1')
            if bits == 1 or bits == 4:
                edges.append([u, v])
    
    # Convert to set for fast lookup
    edge_set = set((u, v) for u, v in edges)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # List of all possible non-edges
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    non_edges = [(u, v) for u, v in all_pairs if (u, v) not in edge_set]
    
    def is_triangle_free(u, v, adj):
        # Check if adding edge (u, v) creates a triangle
        # Triangle exists if there is a common neighbor w
        common = adj[u] & adj[v]
        return len(common) == 0
        
    def is_3_colorable(adj):
        # Backtracking 3-coloring
        colors = [-1] * n
        # Order vertices by degree descending for better pruning
        order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            for c in range(3):
                # Check if color c is allowed
                valid = True
                for w in adj[v]:
                    if colors[w] == c:
                        valid = False
                        break
                if valid:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
            
        return backtrack(0)
        
    # Greedy improvement
    # We try to add edges one by one.
    # To avoid getting stuck, we might shuffle non_edges or try multiple passes.
    # Given time limits, a single pass with restarts is safe.
    
    best_edges = edges[:]
    best_count = len(edges)
    
    # Try to add edges greedily
    # We iterate through non_edges. If we can add one, we add it and restart the loop
    # to see if more can be added.
    
    # Optimization: Pre-compute common neighbors for all pairs?
    # For n=16, it's small enough to compute on the fly.
    
    # We will do a few random restarts of the greedy process if we get stuck?
    # Or just one deterministic pass.
    
    # To maximize, we should prioritize edges that don't block many other edges?
    # Simple heuristic: just try all.
    
    for _pass in range(10): # Limit passes to prevent infinite loops or time outs
        progress = False
        for u, v in non_edges:
            if (u, v) in edge_set:
                continue
            if is_triangle_free(u, v, adj):
                # Tentatively add
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                edges.append([u, v])
                
                if not is_3_colorable(adj):
                    # Valid improvement
                    progress = True
                    # Remove from non_edges? No, just mark as in edge_set
                    # We continue scanning
                    pass
                else:
                    # Revert
                    adj[u].remove(v)
                    adj[v].remove(u)
                    edge_set.remove((u, v))
                    edges.pop()
        
        if not progress:
            break
            
    return edges
# EVOLVE-BLOCK-END
