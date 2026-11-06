import time
import random
import numpy as np
from itertools import combinations

N = 20
DEADLINE_FRACTION = 0.90

def construct():
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION
    
    def _popcount(x):
        return bin(x).count('1')
        
    def _is_valid_add(adj, u, v):
        # Check if adding edge (u, v) keeps graph K3,3-free
        nv = adj[v]
        nu = adj[u]
        
        # Triples (u, x, y) where v is in N(x) and N(y)
        # x, y must be in N(v)
        neigh_v = [i for i in range(N) if (nv >> i) & 1]
        for x, y in combinations(neigh_v, 2):
            common = adj[u] & adj[x] & adj[y]
            if _popcount(common) >= 2:
                return False
                
        # Triples (v, x, y) where u is in N(x) and N(y)
        # x, y must be in N(u)
        neigh_u = [i for i in range(N) if (nu >> i) & 1]
        for x, y in combinations(neigh_u, 2):
            common = adj[v] & adj[x] & adj[y]
            if _popcount(common) >= 2:
                return False
                
        return True

    def _add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def _remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def _edge_count(adj):
        return sum(_popcount(a) for a in adj) // 2

    def _find_violation(adj):
        """Find a triple (u, v, w) with >= 3 common neighbors."""
        for u in range(N):
            for v in range(u+1, N):
                for w in range(v+1, N):
                    common = adj[u] & adj[v] & adj[w]
                    if _popcount(common) >= 3:
                        return (u, v, w)
        return None

    def _repair_graph(adj):
        """Remove edges to make graph K3,3-free."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            viol = _find_violation(adj)
            if viol is None:
                break
            u, v, w = viol
            # Remove one of the edges among u, v, w or their common neighbors?
            # The violation is that u, v, w share 3 neighbors.
            # Removing an edge (u, v) doesn't necessarily fix it if u, v, w still share 3 neighbors via other paths?
            # No, K3,3 requires the 9 edges between {u,v,w} and {x,y,z}.
            # If we remove edge (u, x), then u is not connected to x, so no K3,3 with u in that part.
            # So we should remove an edge connecting one of {u,v,w} to one of the common neighbors.
            common = adj[u] & adj[v] & adj[w]
            neighs = [i for i in range(N) if (common >> i) & 1]
            # Randomly pick one neighbor and one of the vertices to break the edge
            if neighs:
                x = random.choice(neighs)
                # Remove edge between x and one of u, v, w
                target = random.choice([u, v, w])
                _remove_edge(adj, target, x)
                improved = True
        return adj

    def _spectral_init():
        """Initialize graph using spectral heuristic."""
        # Generate random symmetric matrix
        M = np.random.rand(N, N)
        M = (M + M.T) / 2
        np.fill_diagonal(M, 0)
        
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(M)
        
        # Top eigenvector
        top_vec = eigvecs[:, -1]
        
        # Threshold at mean
        mean_val = np.mean(top_vec)
        
        # Create initial adjacency mask
        # Heuristic: Connect i, j if top_vec[i] * top_vec[j] > mean(top_vec[i] * top_vec[j])
        # Or simply if both are above mean?
        # Let's try: G_ij = 1 if top_vec[i] > mean_val and top_vec[j] > mean_val
        # This creates a clique on the "high" vertices.
        # To make it more random-like, let's use:
        # G_ij = 1 if top_vec[i] * top_vec[j] > np.mean(top_vec * top_vec)
        
        prod = np.outer(top_vec, top_vec)
        threshold = np.mean(prod)
        
        adj = [0] * N
        for i in range(N):
            for j in range(i+1, N):
                if prod[i, j] > threshold:
                    _add_edge(adj, i, j)
                    
        return adj

    def _local_search_add(adj):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            pairs = [(u, v) for u in range(N) for v in range(u+1, N) if not ((adj[u] >> v) & 1)]
            random.shuffle(pairs)
            for u, v in pairs:
                if _is_valid_add(adj, u, v):
                    _add_edge(adj, u, v)
                    improved = True
                    if time.time() > deadline:
                        break
        return adj

    def _local_search_swap(adj):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            existing = [(u, v) for u in range(N) for v in range(u+1, N) if (adj[u] >> v) & 1]
            missing = [(u, v) for u in range(N) for v in range(u+1, N) if not ((adj[u] >> v) & 1)]
            
            if not missing:
                break
                
            random.shuffle(existing)
            random.shuffle(missing)
            
            for u1, v1 in existing:
                if time.time() > deadline:
                    break
                for u2, v2 in missing:
                    if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                        continue
                    
                    _remove_edge(adj, u1, v1)
                    
                    if _is_valid_add(adj, u2, v2):
                        _add_edge(adj, u2, v2)
                        improved = True
                        existing = [(u, v) for u in range(N) for v in range(u+1, N) if (adj[u] >> v) & 1]
                        missing = [(u, v) for u in range(N) for v in range(u+1, N) if not ((adj[u] >> v) & 1)]
                        random.shuffle(existing)
                        random.shuffle(missing)
                        break
                    else:
                        _add_edge(adj, u1, v1)
                if improved:
                    break
        return adj

    best_adj = None
    best_count = 0
    
    # Multiple restarts with spectral initialization
    restarts = 0
    while time.time() < deadline and restarts < 200:
        restarts += 1
        
        # Initialize with spectral heuristic
        adj = _spectral_init()
        
        # Repair to make it K3,3-free
        adj = _repair_graph(adj)
        
        # Refine with local search
        adj = _local_search_add(adj)
        adj = _local_search_swap(adj)
        
        count = _edge_count(adj)
        if count > best_count:
            best_count = count
            best_adj = adj[:]

    if best_adj is None:
        return []

    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
