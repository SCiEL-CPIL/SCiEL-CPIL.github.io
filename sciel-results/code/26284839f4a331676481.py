"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_adjacency(edges, n):
    """Build adjacency matrix from edge list."""
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    return adj

def _compute_score(adj, n):
    """Compute the maximum degree sum over all triangles."""
    deg = adj.sum(axis=1)
    # Find all triangles: for each edge (u,v), check common neighbors
    max_score = 0
    # Vectorized: triangles are (u,v,w) where u<v<w and all three edges exist
    # Use the adjacency matrix to find triangles
    # For efficiency, iterate over edges and find common neighbors
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                # Common neighbors of u and v
                common = np.where(adj[u, :1] == 1)[0]
                common = common[common > v]  # w > v to avoid duplicates
                for w in common:
                    if adj[v, w] and adj[u, w]:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_score:
                            max_score = s
    return max_score

def _compute_score_fast(adj, n):
    """Faster score computation using matrix operations."""
    deg = adj.sum(axis=1)
    max_score = 0
    # Triangles: (u,v,w) with u < v < w, all edges present
    # Use: for each pair (u,v) with edge, common neighbors w > v
    # Vectorized approach
    for u in range(n):
        neighbors_u = np.where(adj[u] == 1)[0]
        neighbors_u = neighbors_u[neighbors_u > u]
        for v in neighbors_u:
            # Common neighbors of u and v that are > v
            common = np.where((adj[u] & adj[v]) == 1)[0]
            common = common[common > v]
            for w in common:
                s = deg[u] + deg[v] + deg[w]
                if s > max_score:
                    max_score = s
    return max_score

def _get_all_edges(adj, n):
    """Get list of edges from adjacency matrix."""
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                edges.append([u, v])
    return edges

def _get_non_edges(adj, n):
    """Get list of non-edges from adjacency matrix."""
    non_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if not adj[u, v]:
                non_edges.append([u, v])
    return non_edges

def construct():
    n = 16
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start from K_{6,10} + perfect matching within the 6-side
    # Part A: vertices 0-5 (6 vertices), Part B: vertices 6-15 (10 vertices)
    # All cross edges: 6*10 = 60 edges
    # Plus 5 matching edges within A to get 65 total
    edges = []
    # Cross edges
    for u in range(6):
        for v in range(6, 16):
            edges.append([u, v])
    # Matching within A: (0,1), (2,3), (4,5) - that's 3 edges, need 5 more
    # Add edges (0,2), (0,3), (1,4), (1,5), (2,4), (3,5) - but that's too many
    # Let's just add 5 edges within A: (0,1), (0,2), (0,3), (1,4), (2,5)
    intra_a = [(0,1), (0,2), (0,3), (1,4), (2,5)]
    for u, v in intra_a:
        edges.append([u, v])
    
    adj = _build_adjacency(edges, n)
    current_score = _compute_score_fast(adj, n)
    best_adj = adj.copy()
    best_score = current_score
    
    # Local search: try edge swaps
    while time.time() < deadline:
        improved = False
        all_edges = _get_all_edges(adj, n)
        non_edges = _get_non_edges(adj, n)
        
        # Try each swap: remove edge e1, add edge e2
        for e1 in all_edges:
            if time.time() > deadline:
                break
            u1, v1 = e1
            # Temporarily remove e1
            adj[u1, v1] = 0
            adj[v1, u1] = 0
            
            for e2 in non_edges:
                u2, v2 = e2
                # Skip if same edge
                if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                    continue
                # Temporarily add e2
                adj[u2, v2] = 1
                adj[v2, u2] = 1
                
                new_score = _compute_score_fast(adj, n)
                
                # Undo
                adj[u2, v2] = 0
                adj[v2, u2] = 0
                
                if new_score < current_score:
                    # Accept the swap
                    adj[u1, v1] = 0
                    adj[v1, u1] = 0
                    adj[u2, v2] = 1
                    adj[v2, u2] = 1
                    current_score = new_score
                    improved = True
                    break
                else:
                    # Undo add
                    adj[u2, v2] = 0
                    adj[v2, u2] = 0
            
            # Restore e1 if not accepted
            if not improved:
                adj[u1, v1] = 1
                adj[v1, u1] = 1
            else:
                # e1 was removed, e2 was added
                # Update non_edges list would be needed, but we break anyway
                break
        
        if not improved:
            break
    
    # Return best result
    best_edges = _get_all_edges(best_adj, n)
    return {'n': 16, 'edges': best_edges}
# EVOLVE-BLOCK-END
