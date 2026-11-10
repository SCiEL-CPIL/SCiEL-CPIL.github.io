"""Program using spectral gap heuristic to guide edge insertion for C6-free graphs."""
import numpy as np
import time
from collections import defaultdict

def _has_c6_fast(adj, u, v):
    """
    Check if adding edge (u,v) creates a C6.
    1. Fast check: Are there walks of length 5 from u to v?
    2. If yes, do exact simple path check.
    """
    n = len(adj)
    # Walk check
    # v_k[i] = number of walks of length k from u to i
    vec = np.zeros(n)
    vec[u] = 1
    for _ in range(5):
        # vec = A @ vec
        # Since adj is sparse-ish, we can do this efficiently
        # But N=40, dense matrix mult is fine
        vec = adj.dot(vec)
    
    if vec[v] == 0:
        return False
    
    # Exact check: Simple path of length 5
    # u - a - b - c - d - v
    # Use DFS
    visited = [False] * n
    visited[u] = True
    
    def dfs(current, depth):
        if depth == 5:
            return current == v
        for nxt in range(n):
            if not visited[nxt] and adj[current][nxt]:
                # Exclude u and v from intermediate steps?
                # The path is u-a-b-c-d-v.
                # a,b,c,d must be distinct and not u or v.
                # My DFS starts at u, depth 0.
                # Next is a (depth 1).
                # ...
                # d (depth 4).
                # Next is v (depth 5).
                
                # We must ensure we don't visit u or v before the end.
                if nxt == u:
                    continue
                if nxt == v and depth < 4:
                    continue
                    
                visited[nxt] = True
                if dfs(nxt, depth + 1):
                    return True
                visited[nxt] = False
        return False
    
    return dfs(u, 0)

def _spectral_gap(adj):
    """Compute lambda_1 - lambda_2."""
    try:
        eigvals = np.linalg.eigvalsh(adj)
        eigvals = np.sort(eigvals)[::-1]
        if len(eigvals) >= 2:
            return eigvals[0] - eigvals[1]
        return 0.0
    except:
        return 0.0

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
    
    current_gap = _spectral_gap(adj)
    
    while all_possible:
        if time.time() - start_time > timeout:
            break
        
        # Sample candidates
        sample_size = min(len(all_possible), 100)
        candidates = random.sample(all_possible, sample_size)
        
        best_edge = None
        best_score = -1e9
        
        for (u, v) in candidates:
            # Fast C6 check
            if _has_c6_fast(adj, u, v):
                continue
                
            # Spectral heuristic
            adj[u][v] = 1
            adj[v][u] = 1
            new_gap = _spectral_gap(adj)
            adj[u][v] = 0
            adj[v][u] = 0
            
            score = new_gap - current_gap
            if score > best_score:
                best_score = score
                best_edge = (u, v)
                
        if best_edge is None:
            # No candidate in sample is valid
            # Remove sampled candidates from all_possible?
            # Or just break?
            # If we break, we might miss edges.
            # Let's try to find ANY valid edge in the sample, ignoring score.
            valid_edge = None
            for (u, v) in candidates:
                if not _has_c6_fast(adj, u, v):
                    valid_edge = (u, v)
                    break
            
            if valid_edge is None:
                # No valid edge in sample. Remove them.
                candidate_set = set(candidates)
                all_possible = [e for e in all_possible if e not in candidate_set]
                if not all_possible:
                    break
                continue
            else:
                best_edge = valid_edge
                # Update current gap for this edge?
                # We didn't compute it, but we can just add it.
                # We should update current_gap.
                u, v = best_edge
                adj[u][v] = 1
                adj[v][u] = 1
                current_gap = _spectral_gap(adj)
                edges.append([u, v])
                all_possible.remove(best_edge)
                continue

        # Add best edge
        u, v = best_edge
        adj[u][v] = 1
        adj[v][u] = 1
        current_gap = _spectral_gap(adj)
        edges.append([u, v])
        all_possible.remove(best_edge)
        
    return edges
