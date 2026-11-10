"""Program using spectral gap heuristic to guide edge insertion for C6-free graphs."""
import numpy as np
import time
from collections import defaultdict

def _has_c6(adj, u, v):
    """
    Check if adding edge (u,v) creates a C6.
    We need to check if there is a path of length 5 from u to v in the current graph.
    A C6 would be u-a-b-c-d-v where a,b,c,d are distinct and not u or v.
    """
    n = len(adj)
    
    # BFS/DFS to find path of length exactly 5 from u to v
    # State: (current_node, depth, visited_mask)
    # Since n=40, we can use bitmasks for visited nodes.
    # But 2^40 is too large. We need a smarter approach.
    
    # Alternative: 
    # A path of length 5 from u to v is u -> x1 -> x2 -> x3 -> x4 -> v.
    # x1, x2, x3, x4 must be distinct and not equal to u or v.
    
    # Let's use matrix multiplication for a faster check first.
    # If (A^5)[u][v] == 0, no path exists.
    # If > 0, we need to verify it's a simple path.
    
    # Compute A^5
    # For n=40, this is 5 matrix multiplications of 40x40 matrices.
    # This is fast enough.
    
    A = adj.astype(np.int64)
    # Power A to 5
    A2 = A @ A
    A3 = A2 @ A
    A4 = A3 @ A
    A5 = A4 @ A
    
    if A5[u][v] == 0:
        return False
    
    # Now we need to check if there is a SIMPLE path of length 5.
    # The walk count A5[u][v] might include non-simple paths.
    # We need to verify existence of a simple path.
    
    # We can use backtracking/DFS with pruning.
    # Start from u, find path to v of length 5.
    # Depth 0: u
    # Depth 1: neighbors of u (excluding v, since path length is 5, not 1)
    # ...
    # Depth 4: neighbors of depth 3 node (must be v at depth 5)
    # Depth 5: v
    
    # Let's do DFS from u.
    # visited: set of visited nodes.
    
    def dfs(node, depth, visited):
        if depth == 5:
            return node == v
        
        if depth == 4:
            # Next node must be v
            return adj[node][v] == 1 and v not in visited
        
        for nxt in range(n):
            if adj[node][nxt] and nxt not in visited:
                if nxt == u:
                    continue
                if nxt == v and depth < 4:
                    continue
                
                visited.add(nxt)
                if dfs(nxt, depth + 1, visited):
                    visited.discard(nxt)
                    return True
                visited.discard(nxt)
        
        return False
    
    visited = {u}
    return dfs(u, 0, visited)

def construct():
    n = 40
    start_time = time.time()
    timeout = 110
    
    # Start with 8 disjoint copies of K5
    adj = np.zeros((n, n), dtype=np.int8)
    edges = []
    
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u][v] = 1
                adj[v][u] = 1
                edges.append([u, v])
                
    # All possible edges
    all_possible = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 0:
                all_possible.append((i, j))
                
    import random
    random.seed(42)
    random.shuffle(all_possible)
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    
    while all_possible:
        if time.time() - start_time > timeout:
            break
        
        # Prioritize edges between nodes with lower degrees
        # This tends to balance the graph and avoid creating dense clusters
        # that might lead to C6s
        
        # Sort candidates by sum of degrees (ascending)
        # But we don't want to sort the entire list every time.
        # Instead, sample a larger set and pick the best.
        
        sample_size = min(len(all_possible), 200)
        candidates = random.sample(all_possible, sample_size)
        
        best_edge = None
        best_score = -1e9
        
        for (u, v) in candidates:
            # Fast C6 check
            if _has_c6(adj, u, v):
                continue
            
            # Heuristic: prefer edges that connect low-degree nodes
            # This helps maintain a more regular graph structure
            score = -(degrees[u] + degrees[v])
            
            # Additional heuristic: prefer edges that don't create triangles
            # with existing edges, as triangles can lead to C6s
            # Count common neighbors
            common = np.sum(adj[u] & adj[v])
            score -= common * 0.5
            
            if score > best_score:
                best_score = score
                best_edge = (u, v)
                
        if best_edge is None:
            # Try to find any valid edge
            valid_edge = None
            for (u, v) in candidates:
                if not _has_c6(adj, u, v):
                    valid_edge = (u, v)
                    break
            
            if valid_edge is None:
                candidate_set = set(candidates)
                all_possible = [e for e in all_possible if e not in candidate_set]
                if not all_possible:
                    break
                continue
            else:
                best_edge = valid_edge
        
        u, v = best_edge
        adj[u][v] = 1
        adj[v][u] = 1
        degrees[u] += 1
        degrees[v] += 1
        edges.append([u, v])
        
        # Remove from all_possible
        try:
            all_possible.remove(best_edge)
        except ValueError:
            pass
        
    return edges
