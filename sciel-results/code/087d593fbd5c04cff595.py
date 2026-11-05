"""Improved K3,3-free graph on 20 vertices via greedy edge addition with constraint check."""
# EVOLVE-BLOCK-START
import time
import random

def _check_pair_constraint(adj, u, v):
    """Check that u and v have at most 2 common neighbors."""
    # Count common neighbors
    count = 0
    nu = adj[u]
    nv = adj[v]
    # Iterate over smaller set
    if len(nu) < len(nv):
        small, big = nu, nv
    else:
        small, big = nv, nu
    for x in small:
        if x in big:
            count += 1
            if count > 2:
                return False
    return True

def _can_add_edge(adj, u, v):
    """Check if adding edge (u,v) preserves K3,3-free property."""
    # Adding edge (u,v) creates K3,3 iff there exist a,b,c in N(u)∩N(v)
    # such that the new edge closes a K3,3.
    # Actually: K3,3 with bipartition {u,v,x} and {a,b,c} requires
    # x adjacent to u,v and a,b,c; and a,b,c adjacent to u and v.
    # The new edge (u,v) matters when u,v are in the same part of the K3,3.
    # That requires: some x adjacent to both u and v, AND |N(u)∩N(v)∩{neighbors of x}| >= 3.
    # But since x is a common neighbor, we need 3 common neighbors of u,v that are also neighbors of x.
    # Simpler: after adding edge, check that no pair has >2 common neighbors.
    # The only pairs affected are (u,v) and pairs (u,w) and (v,w) for common neighbors w.
    # Actually adding edge (u,v) only changes common-neighbor counts for pairs involving u or v.
    # Pair (u,v): common neighbors are N(u)∩N(v) (unchanged by adding edge, since edge doesn't change adjacency to others)
    # Wait, adding edge (u,v) doesn't change N(u)∩N(v) for the purpose of other pairs.
    # Let me think again: N(u) gains v, N(v) gains u.
    # For pair (u, w) where w != v: common neighbors = (N(u)∪{v}) ∩ N(w) = (N(u)∩N(w)) ∪ ({v} if v in N(w))
    # So count increases by 1 if v is neighbor of w.
    # Similarly for pair (v, w).
    # For pair (u,v): common neighbors = N(u)∩N(v) (the edge u-v doesn't count as a common neighbor)
    # So we need to check:
    # 1. |N(u)∩N(v)| <= 2 (unchanged)
    # 2. For each w in N(v) (w != u): |(N(u)∩N(w))| + (1 if v in N(w) else 0) <= 2
    #    i.e., for w in N(v), w != u: |N(u)∩N(w)| + 1 <= 2, so |N(u)∩N(w)| <= 1
    # 3. For each w in N(u) (w != v): |N(v)∩N(w)| + 1 <= 2, so |N(v)∩N(w)| <= 1
    
    # Check condition 1
    if not _check_pair_constraint(adj, u, v):
        return False
    
    # Check condition 2: for w in N(v), w != u, need |N(u)∩N(w)| <= 1
    for w in adj[v]:
        if w == u:
            continue
        # Count common neighbors of u and w
        count = 0
        for x in adj[u]:
            if x == w:
                continue
            if x in adj[w]:
                count += 1
                if count > 1:
                    return False
    
    # Check condition 3: for w in N(u), w != v, need |N(v)∩N(w)| <= 1
    for w in adj[u]:
        if w == v:
            continue
        count = 0
        for x in adj[v]:
            if x == w:
                continue
            if x in adj[w]:
                count += 1
                if count > 1:
                    return False
    
    return True

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n = 20
    
    best_edges = []
    best_count = 0
    
    # Try multiple random orderings
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Random permutation of all edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        adj = [set() for _ in range(n)]
        edges = []
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if _can_add_edge(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    return best_edges
# EVOLVE-BLOCK-END
