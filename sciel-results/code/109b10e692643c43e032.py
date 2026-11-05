"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _has_triangle(adj, n):
    """Check if graph has a triangle using adjacency matrix."""
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                for k in range(j + 1, n):
                    if adj[i][k] and adj[j][k]:
                        return True
    return False

def _max_independent_set_size(adj, n):
    """Compute alpha(G) exactly using branch and bound."""
    # Use bitmasks for efficiency
    # adj_mask[i] is a bitmask of neighbors of i
    adj_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and adj[i][j]:
                adj_mask[i] |= (1 << j)
    
    # Non-neighbors mask (for independent set: we want vertices with no edges between them)
    # independent set = clique in complement
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = ((1 << n) - 1) ^ (1 << i) ^ adj_mask[i]
    
    best = 0
    def dfs(candidates, current_size):
        nonlocal best
        if current_size + bin(candidates).count('1') <= best:
            return
        if candidates == 0:
            if current_size > best:
                best = current_size
            return
        # Branch and bound: pick a vertex
        # Greedy: try to extend
        v = (candidates & -candidates).bit_length() - 1
        # Include v
        dfs(candidates & comp_mask[v], current_size + 1)
        # Exclude v
        dfs(candidates & ~(1 << v), current_size)
    
    all_mask = (1 << n) - 1
    dfs(all_mask, 0)
    return best

def _verify(adj, n):
    """Verify no triangle and alpha < 10."""
    if _has_triangle(adj, n):
        return False
    alpha = _max_independent_set_size(adj, n)
    if alpha >= 10:
        return False
    return True

def _build_adj(edges, n):
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with the known valid 26-vertex Andrasfai graph
    n0 = 26
    edges0 = [[i, j] for i in range(n0) for j in range(i + 1, n0) if (j - i) % n0 % 3 == 1]
    best_n = n0
    best_edges = edges0
    
    # Try to extend to larger n using local search
    # Strategy: start with n=26 valid graph, try to add vertices and adjust edges
    
    for target_n in range(27, 41):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize: copy the 26-vertex graph, add new vertices
            adj = [[False] * target_n for _ in range(target_n)]
            for u, v in edges0:
                adj[u][v] = True
                adj[v][u] = True
            
            # New vertices start with random connections to old vertices
            random.seed(attempt * 1000 + target_n)
            for new_v in range(n0, target_n):
                for old_v in range(n0):
                    if random.random() < 0.3:
                        adj[new_v][old_v] = True
                        adj[old_v][new_v] = True
            
            # Local search: flip edges to eliminate triangles and reduce alpha
            improved = True
            max_iter = 200
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Check validity
                if _verify(adj, target_n):
                    found = True
                    break
                
                # Find a triangle and break it
                if _has_triangle(adj, target_n):
                    # Find first triangle
                    tri_found = False
                    for i in range(target_n):
                        if tri_found:
                            break
                        for j in range(i + 1, target_n):
                            if tri_found or not adj[i][j]:
                                continue
                            for k in range(j + 1, target_n):
                                if adj[i][k] and adj[j][k]:
                                    # Triangle (i,j,k): remove one edge
                                    # Remove the edge that appears most often in triangles (heuristic: just pick first)
                                    adj[i][j] = False
                                    adj[j][i] = False
                                    tri_found = True
                                    break
                else:
                    # No triangle, but alpha >= 10: try to add edges to reduce alpha
                    # Find an independent set of size 10 and add an edge within it
                    # For simplicity, try random edge additions
                    for _ in range(10):
                        u = random.randint(0, target_n - 1)
                        v = random.randint(0, target_n - 1)
                        if u != v and not adj[u][v]:
                            adj[u][v] = True
                            adj[v][u] = True
                            if _has_triangle(adj, target_n):
                                # Revert if it creates a triangle
                                adj[u][v] = False
                                adj[v][u] = False
                            else:
                                break
            
            if found:
                # Extract edges
                new_edges = []
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if adj[u][v]:
                            new_edges.append([u, v])
                best_n = target_n
                best_edges = new_edges
                break
        
        if not found and target_n > 30:
            # If we can't extend further, stop trying larger n
            break
    
    # Final verification
    adj_final = _build_adj(best_edges, best_n)
    if not _verify(adj_final, best_n):
        # Fallback to known valid graph
        best_n = n0
        best_edges = edges0
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
