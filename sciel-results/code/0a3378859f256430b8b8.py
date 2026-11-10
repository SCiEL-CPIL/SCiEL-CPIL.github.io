"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def _is_k23_free(adj):
    """Check if graph with adjacency matrix adj has no K2,3."""
    n = adj.shape[0]
    common = adj @ adj
    for u in range(n):
        for v in range(u+1, n):
            if common[u, v] >= 3:
                return False
    return True

def _count_edges(adj):
    return int(adj.sum() // 2)

def construct():
    n = 20
    # Step 1: Spectral/Circulant Construction
    # Vertices 0..19.
    # Symmetric difference sets S.
    # Directions: 1..9 (pairs {d, 20-d}) and 10 (self-inverse).
    # We iterate over all subsets of these 10 directions.
    
    best_adj = None
    best_edges = 0
    
    directions = list(range(1, 10)) + [10]
    
    # Iterate over all 2^10 = 1024 subsets
    for mask in range(1 << len(directions)):
        S = []
        for i, d in enumerate(directions):
            if (mask >> i) & 1:
                if d == 10:
                    S.append(10)
                else:
                    S.append(d)
                    S.append(20 - d)
        
        adj = np.zeros((n, n), dtype=int)
        for u in range(n):
            for d in S:
                v = (u + d) % n
                if u < v:
                    adj[u, v] = 1
                    adj[v, u] = 1
        
        if _is_k23_free(adj):
            e = _count_edges(adj)
            if e > best_edges:
                best_edges = e
                best_adj = adj.copy()
    
    # If no valid circulant graph found (unlikely), start with empty
    if best_adj is None:
        best_adj = np.zeros((n, n), dtype=int)
    
    # Step 2: Coordinate Ascent / Local Search Optimization
    # We try to add edges and remove edges to maximize edge count.
    # This phase breaks the circulant symmetry.
    
    adj = best_adj
    
    # Helper to check if adding edge (u,v) is valid
    def can_add_edge(adj, u, v):
        # Common neighbors of u and v
        common_uv = np.sum(adj[u, :] & adj[v, :])
        if common_uv >= 3:
            return False
        
        # Check impact on other pairs
        # Adding (u,v) increases common neighbors for pairs (u, w) where w~v
        # and pairs (v, w) where w~u.
        
        # For each w in N(v) (excluding u):
        # new common neighbors of u and w = old common neighbors of u and w + 1 (if v is in N(w))
        # Wait, v is in N(w) because w is in N(v) and graph is undirected.
        # So we need old_common(u, w) < 2.
        # Similarly for w in N(u) (excluding v): old_common(v, w) < 2.
        
        # Efficient check:
        # Let A be the current adjacency matrix.
        # We are adding A[u,v] = 1.
        # New A' = A + E_uv.
        # (A')^2 = A^2 + A E + E A + E^2.
        # We only care about entries (i,j) that might exceed 2.
        # Entries (u,v): (A')^2[u,v] = A^2[u,v] + A[u,u] + A[v,v] + 1? No.
        # (A^2)[u,v] is common neighbors.
        # Adding edge (u,v):
        # 1. Common neighbors of u and v: unchanged by the edge itself?
        #    No, common neighbors are $w$ such that $A[u,w]=1, A[v,w]=1$.
        #    Adding $A[u,v]=1$ doesn't change $A[u,w]$ or $A[v,w]$ for $w \neq u,v$.
        #    So common_uv is unchanged.
        
        # 2. Common neighbors of u and w (where w != v):
        #    $w$ is a common neighbor of u and x? No.
        #    We need to check if any pair (i,j) has common neighbors >= 3.
        #    Pairs affected:
        #    - (u, v): common neighbors unchanged.
        #    - (u, w) for w != v:
        #      Common neighbors of u and w in A' are $z$ such that $A'[u,z]=1, A'[w,z]=1$.
        #      $A'[u,z] = A[u,z] + \delta_{z,v}$.
        #      So if $z=v$, $A'[u,v]=1$. Is $A'[w,v]=1$? i.e., $w \sim v$?
        #      If $w \sim v$, then $v$ becomes a common neighbor of u and w.
        #      So common_count(u,w) increases by 1 if $w \sim v$.
        #      We need $common_count(u,w)_{new} < 3$.
        #      i.e., $common_count(u,w)_{old} + (1 \text{ if } w \sim v \text{ else } 0) < 3$.
        
        #    - (v, w) for w != u:
        #      Similarly, $u$ becomes a common neighbor of v and w if $w \sim u$.
        #      We need $common_count(v,w)_{old} + (1 \text{ if } w \sim u \text{ else } 0) < 3$.
        
        #    - (w1, w2) where $w1, w2 \notin \{u,v\}$:
        #      No change.
        
        # So we check:
        # 1. common_uv < 3
        # 2. For all w in N(v) \ {u}: common(u,w) < 2
        # 3. For all w in N(u) \ {v}: common(v,w) < 2
        
        common = adj @ adj
        
        if common[u, v] >= 3:
            return False
            
        # Check w in N(v)
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v, w]:
                if common[u, w] >= 2:
                    return False
            if adj[u, w]:
                if common[v, w] >= 2:
                    return False
                    
        return True

    # Greedy addition phase
    improved = True
    while improved:
        improved = False
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v] == 0:
                    if can_add_edge(adj, u, v):
                        adj[u, v] = 1
                        adj[v, u] = 1
                        improved = True
    
    # Swap optimization phase
    # Try removing an edge, then adding as many as possible.
    # If net gain > 0, keep it.
    
    changed = True
    while changed:
        changed = False
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v] == 1:
                    # Try removing (u,v)
                    adj_copy = adj.copy()
                    adj_copy[u, v] = 0
                    adj_copy[v, u] = 0
                    
                    # Try adding edges to adj_copy
                    added = 0
                    for x in range(n):
                        for y in range(x+1, n):
                            if adj_copy[x, y] == 0:
                                if can_add_edge(adj_copy, x, y):
                                    adj_copy[x, y] = 1
                                    adj_copy[y, x] = 1
                                    added += 1
                                    # Don't break, try to add all possible to maximize gain
                                    # But this might be slow? n=20, 190 pairs.
                                    # can_add_edge is O(n^2) roughly.
                                    # 190 * 400 = 76000 ops per swap check.
                                    # 190 swaps to check. 14 million ops. Fine.
                    
                    if added > 1:
                        adj = adj_copy
                        changed = True
                        break
            if changed:
                break

    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
