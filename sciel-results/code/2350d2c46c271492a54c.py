"""Program using spectral gap heuristic to guide edge insertion for C6-free graphs."""
import numpy as np
import time
from collections import defaultdict

def _has_c6(adj, u, v):
    """Check if adding edge (u,v) would create a C6 in the graph with adjacency matrix adj."""
    # A C6 through edge (u,v) means there's a path of length 5 from u to v
    # that doesn't use edge (u,v) directly
    # We can check by BFS/DFS for paths of length exactly 5 from u to v
    # But more efficiently: check if there's a path of length 5 from u to v
    
    # For efficiency, use matrix powers or BFS with depth limit
    # Path of length 5 from u to v: u - a - b - c - d - v
    # We need to find such paths where none of a,b,c,d is u or v (simple path)
    
    # Simple approach: BFS from u up to depth 5, track paths
    # But this is expensive. Let's use a smarter check.
    
    # Alternative: check if there exists a vertex w such that dist(u,w) + dist(w,v) = 5
    # and the paths can be made vertex-disjoint (except at w)
    
    # For simplicity and correctness, let's do a bounded BFS
    # Find all vertices reachable from u in exactly k steps for k=1..5
    
    n = len(adj)
    
    # dist_u[i] = set of vertices reachable from u in exactly i steps
    # We want to check if v is reachable from u in exactly 5 steps via a simple path
    
    # Use DFS with backtracking for paths of length 5
    # This is O(n^5) worst case but with pruning should be okay for n=40
    
    # Better: use matrix multiplication approach
    # A^5[u][v] counts walks of length 5, but we need simple paths
    
    # Let's just do a careful BFS for paths of length exactly 5
    # Start from u, go 5 steps, see if we can reach v
    
    # visited[u] = True at depth 0
    # At each depth, we can only visit unvisited vertices
    
    # This is essentially checking if there's a simple path of length 5 from u to v
    # which is equivalent to checking if v is in the set of vertices reachable from u
    # by a simple path of length 5
    
    # For n=40, this might be slow if done naively. Let's use a heuristic:
    # Check if there's a path of length 5 using BFS with parent tracking
    
    # Actually, let's use a different approach: 
    # A C6 exists through edge (u,v) if there's a path of length 5 from u to v
    # that doesn't use (u,v). We can check this by temporarily removing (u,v)
    # and checking if dist(u,v) <= 5 in the remaining graph, and specifically
    # if there's a path of exactly length 5.
    
    # For efficiency, let's check: is there a vertex w at distance 2 from u and 
    # distance 3 from v (or vice versa) in the graph without (u,v)?
    
    # Simpler: just do a bounded DFS
    def _path_exists_length5(start, end, adj_matrix, n):
        """Check if there's a simple path of length exactly 5 from start to end."""
        # DFS with depth limit
        visited = [False] * n
        visited[start] = True
        
        def dfs(current, depth, path):
            if depth == 5:
                return current == end
            for nxt in range(n):
                if not visited[nxt] and adj_matrix[current][nxt]:
                    visited[nxt] = True
                    path.append(nxt)
                    if dfs(nxt, depth + 1, path):
                        return True
                    path.pop()
                    visited[nxt] = False
            return False
        
        return dfs(start, 0, [start])
    
    return _path_exists_length5(u, v, adj, n)


def _spectral_gap(adj):
    """Compute the spectral gap (lambda_1 - lambda_2) of the adjacency matrix."""
    try:
        eigvals = np.linalg.eigvalsh(adj)
        eigvals = np.sort(eigvals)[::-1]  # descending order
        if len(eigvals) >= 2:
            return eigvals[0] - eigvals[1]
        return 0.0
    except:
        return 0.0


def construct():
    """Construct a C6-free graph on 40 vertices maximizing edges using spectral gap heuristic."""
    n = 40
    start_time = time.time()
    timeout = 110  # Leave some buffer for the 120s limit
    
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
    
    # Generate all possible edges not yet in the graph
    all_possible = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 0:
                all_possible.append((i, j))
    
    # Shuffle for diversity
    import random
    random.seed(42)
    random.shuffle(all_possible)
    
    # Iteratively add edges that improve spectral gap while maintaining C6-freeness
    while all_possible:
        if time.time() - start_time > timeout:
            break
        
        best_edge = None
        best_gap_improvement = -1e9
        current_gap = _spectral_gap(adj)
        
        # Sample a subset of candidate edges to evaluate (for speed)
        sample_size = min(len(all_possible), 50)
        candidates = random.sample(all_possible, sample_size)
        
        for (u, v) in candidates:
            # Check if adding this edge would create a C6
            if _has_c6(adj, u, v):
                continue
            
            # Compute new spectral gap
            adj[u][v] = 1
            adj[v][u] = 1
            new_gap = _spectral_gap(adj)
            adj[u][v] = 0
            adj[v][u] = 0
            
            improvement = new_gap - current_gap
            if improvement > best_gap_improvement:
                best_gap_improvement = improvement
                best_edge = (u, v)
        
        if best_edge is None:
            # No edge in the sample improves the gap, try next batch
            all_possible = [e for e in all_possible if e not in set(candidates)]
            if not all_possible:
                break
            continue
        
        # Add the best edge
        u, v = best_edge
        adj[u][v] = 1
        adj[v][u] = 1
        edges.append([u, v])
        all_possible.remove(best_edge)
    
    return edges
