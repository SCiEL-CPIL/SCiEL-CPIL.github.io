import time
import random

def _is_triangle_free(adj):
    n = len(adj)
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                # Check if u and v have a common neighbor
                common = 0
                for w in range(n):
                    if w != u and w != v:
                        if adj[u][w] and adj[v][w]:
                            return False
    return True

def _is_3_colorable(adj):
    n = len(adj)
    # Bitmask backtracking
    # colors: 0, 1, 2
    # adj_mask[i] is a bitmask of neighbors of i
    
    adj_mask = [0] * n
    for u in range(n):
        mask = 0
        for v in range(n):
            if adj[u][v]:
                mask |= (1 << v)
        adj_mask[u] = mask

    colors = [-1] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        
        # Try to assign a color to vertex idx
        # Determine available colors
        used_colors = 0
        for neighbor_mask in [adj_mask[idx]]:
             # We need to check neighbors that are already colored
             pass
             
        # Simpler: check all neighbors
        used = 0
        for v in range(n):
            if adj[idx][v] and colors[v] != -1:
                used |= (1 << colors[v])
                
        for c in range(3):
            if not (used & (1 << c)):
                colors[idx] = c
                if backtrack(idx + 1):
                    return True
                colors[idx] = -1
        return False

    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave margin for safety
    
    n = 16
    
    # Initialize with Clebsch graph
    adj = [[0]*n for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            xor = u ^ v
            dist = bin(xor).count('1')
            if dist == 1 or dist == 4:
                adj[u][v] = 1
                adj[v][u] = 1
                edges.append([u, v])
                
    current_edges = len(edges)
    
    # List of all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v]:
                all_edges.append([u, v])
                
    # Greedy improvement
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        # Shuffle to break symmetry? Or just iterate
        # For deterministic result, iterate in fixed order
        # But maybe random order helps escape local minima?
        # Let's try fixed order first for speed and determinism
        
        for e in all_edges:
            if time.time() > deadline:
                break
            u, v = e
            if adj[u][v]:
                continue
                
            # Check triangle-free
            # u and v must not have common neighbors
            is_tf = True
            for w in range(n):
                if w != u and w != v:
                    if adj[u][w] and adj[v][w]:
                        is_tf = False
                        break
                        # break inner, but need to break outer?
                        # is_tf is False, so skip
                        break
            if not is_tf:
                continue
                
            # Check 3-colorability
            # Add edge temporarily
            adj[u][v] = 1
            adj[v][u] = 1
            
            if not _is_3_colorable(adj):
                # Valid improvement
                edges.append(e)
                current_edges += 1
                improved = True
                # Remove e from all_edges? Or just mark as used
                # We can leave it in all_edges, adj check will skip it
            else:
                # Revert
                adj[u][v] = 0
                adj[v][u] = 0
                
    return edges
