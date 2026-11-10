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
    
    # Use a finite field GF(p) where p is a prime > 20
    # We'll try p = 29 (29 ≡ 1 mod 4, so -1 is a quadratic residue)
    p = 29
    
    # Precompute quadratic residues mod p
    qr = set()
    for i in range(1, p):
        qr.add((i * i) % p)
    # 0 is not a quadratic residue for our purposes (no self-loops)
    
    # Map 20 vertices to distinct elements in GF(p)
    # Start with a random permutation, then optimize via coordinate ascent
    np.random.seed(42)
    mapping = np.random.choice(p, size=n, replace=False)
    
    def build_adj_from_mapping(mapping):
        adj = np.zeros((n, n), dtype=int)
        for u in range(n):
            for v in range(u+1, n):
                diff = (mapping[v] - mapping[u]) % p
                if diff in qr:
                    adj[u, v] = adj[v, u] = 1
        return adj
    
    def max_common_neighbors(adj):
        common = adj @ adj
        max_cn = 0
        for u in range(n):
            for v in range(u+1, n):
                if common[u, v] > max_cn:
                    max_cn = common[u, v]
        return max_cn
    
    def is_valid(adj):
        return max_common_neighbors(adj) <= 2
    
    # Coordinate ascent: try swapping two vertices' positions in the field
    current_mapping = mapping.copy()
    current_adj = build_adj_from_mapping(current_mapping)
    current_edges = _count_edges(current_adj)
    
    if not is_valid(current_adj):
        # If invalid, we need to fix it. Remove edges that cause violations.
        # But first, let's try to find a valid mapping.
        # For now, start with a sparse graph and add edges.
        current_adj = np.zeros((n, n), dtype=int)
        current_edges = 0
    
    # Iteratively improve: try all swaps and keep the best valid one
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                # Try swapping mapping[i] and mapping[j]
                new_mapping = current_mapping.copy()
                new_mapping[i], new_mapping[j] = new_mapping[j], new_mapping[i]
                new_adj = build_adj_from_mapping(new_mapping)
                if is_valid(new_adj):
                    new_edges = _count_edges(new_adj)
                    if new_edges > current_edges:
                        current_mapping = new_mapping
                        current_adj = new_adj
                        current_edges = new_edges
                        improved = True
    
    # Now try to add more edges by modifying the edge rule
    # Try different quadratic residue sets or affine transformations
    # For now, let's also try local edge additions
    
    # Try adding edges that don't violate K2,3-free
    def try_add_edges(adj):
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u, v] == 0:
                        # Check if adding (u,v) creates a K2,3
                        # After adding (u,v), check all pairs
                        test_adj = adj.copy()
                        test_adj[u, v] = 1
                        test_adj[v, u] = 1
                        if is_valid(test_adj):
                            adj = test_adj
                            improved = True
        return adj
    
    current_adj = try_add_edges(current_adj)
    current_edges = _count_edges(current_adj)
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if current_adj[u, v]:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
