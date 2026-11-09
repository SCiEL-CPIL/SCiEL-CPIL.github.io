"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    
    n = 20
    # Total possible edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    m_total = len(all_pairs)  # 190
    
    # Map edge index to (u, v)
    edge_uv = {}
    for idx, (u, v) in enumerate(all_pairs):
        edge_uv[idx] = (u, v)
    
    # Precompute: for each pair of vertices (a, b), which edges have a or b as endpoint
    # Actually, for K3,3 check: three vertices a,b,c have 3 common neighbors
    # Common neighbors of {a,b,c} = intersection of N(a), N(b), N(c)
    # We need |N(a) ∩ N(b) ∩ N(c)| < 3 for all triples {a,b,c}
    
    # Faster check: for each triple (a,b,c), count common neighbors
    # Let's precompute for each vertex, its neighbor set (as a bitset or list)
    
    # Use adjacency matrix representation for speed
    # adj[u][v] = 1 if edge exists
    
    # Let's use a different approach: track for each pair (a,b), the set of common neighbors
    # common_nb[a][b] = set of vertices that are neighbors of both a and b
    # Then for triple (a,b,c): |common_nb[a][b] ∩ common_nb[a][c]| >= 3 means K3,3
    # Actually: common neighbors of a,b,c = common_nb[a][b] ∩ N(c)
    # Or: |N(a) ∩ N(b) ∩ N(c)| >= 3
    
    # Let's use numpy for speed
    
    def count_common_neighbors(adj, a, b, c):
        """Count common neighbors of vertices a, b, c using adjacency matrix."""
        # adj is n x n boolean matrix
        # Common neighbors = positions where adj[a], adj[b], adj[c] are all 1
        # Exclude a, b, c themselves (they're not neighbors of themselves in simple graph)
        mask = adj[a] & adj[b] & adj[c]
        # a, b, c are not neighbors of themselves, so adj[a][a]=0, etc.
        # But we should still exclude them to be safe
        mask[a] = False
        mask[b] = False
        mask[c] = False
        return np.sum(mask)
    
    def is_valid(adj):
        """Check if graph has no K3,3."""
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    if count_common_neighbors(adj, a, b, c) >= 3:
                        return False
        return True
    
    def count_violations(adj):
        """Count number of K3,3 violations (triples with >=3 common neighbors)."""
        violations = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    if count_common_neighbors(adj, a, b, c) >= 3:
                        violations += 1
        return violations
    
    # Simulated Annealing
    # Start with a random dense graph
    rng = np.random.default_rng(42)
    
    # Initial: random graph with edge probability ~0.5
    p_init = 0.5
    adj = rng.random((n, n)) < p_init
    np.fill_diagonal(adj, False)
    # Make symmetric
    adj = adj | adj.T
    
    current_edges = int(np.sum(adj) // 2)
    current_violations = count_violations(adj)
    
    best_adj = adj.copy()
    best_edges = current_edges if current_violations == 0 else -1
    
    # SA parameters
    T_start = 10.0
    T_end = 0.01
    alpha = 0.9995  # cooling rate
    
    # Number of iterations per temperature step
    # We need to be efficient. Let's estimate: 115 seconds available
    # Each iteration: flip one edge, check delta violations
    # Full check is O(n^3) = 8000 operations, but we can do incremental
    
    # For efficiency, let's do a simpler approach: 
    # Flip edge, then only check triples that include at least one endpoint of the edge
    # If edge (u,v) is added/removed, only triples containing u or v can change
    
    def get_affected_triples(u, v):
        """Get triples that include u or v."""
        triples = []
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    if a == u or b == u or c == u or a == v or b == v or c == v:
                        triples.append((a, b, c))
        return triples
    
    # Precompute affected triples for each vertex
    affected_by_vertex = {}
    for v in range(n):
        triples = []
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    if a == v or b == v or c == v:
                        triples.append((a, b, c))
        affected_by_vertex[v] = triples
    
    # For edge (u,v), affected triples = union of affected_by_vertex[u] and affected_by_vertex[v]
    # But this is still many triples. Let's just do full check but optimized.
    
    # Actually, let's use a different strategy: 
    # Use the fact that we only need to check if ANY triple has >=3 common neighbors
    # and we can stop early.
    
    # Let's just do a simple SA with full validity check, but limit iterations
    # based on time.
    
    T = T_start
    iteration = 0
    no_improve_count = 0
    
    while T > T_end:
        # Check time
        if time.time() > deadline:
            break
        
        # Pick a random edge to flip
        u = rng.integers(n)
        v = rng.integers(n)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        old_state = adj[u, v]
        
        # Flip the edge
        adj[u, v] = not old_state
        adj[v, u] = not old_state
        
        new_violations = count_violations(adj)
        new_edges = int(np.sum(adj) // 2)
        
        # Energy: we want to minimize violations and maximize edges
        # Energy = violations * 1000 - edges (so violations dominate)
        old_energy = current_violations * 1000 - current_edges
        new_energy = new_violations * 1000 - new_edges
        
        delta_E = new_energy - old_energy
        
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            # Accept
            current_violations = new_violations
            current_edges = new_edges
            
            if current_violations == 0 and current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Reject: revert
            adj[u, v] = old_state
            adj[v, u] = old_state
        
        T *= alpha
        iteration += 1
    
    # If we found a valid graph, return it
    if best_edges > 0:
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if best_adj[u, v]:
                    edges.append([u, v])
        return edges
    
    # Fallback: return the seed solution (K2,18)
    return [[a, b] for a in range(2) for b in range(2, 20)]

# EVOLVE-BLOCK-END
