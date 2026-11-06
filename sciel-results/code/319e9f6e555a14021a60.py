"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations
import random

# EVOLVE-BLOCK-START
def _build_bitmask(n, edges):
    """Build adjacency as list of bitmasks for fast triangle detection."""
    mask = [0] * n
    for u, v in edges:
        mask[u] |= (1 << v)
        mask[v] |= (1 << u)
    return mask

def _compute_score_fast(n, mask, deg):
    """Compute max triangle degree sum using bitmask operations."""
    max_score = 0
    for i in range(n):
        mi = mask[i]
        # For each j > i adjacent to i
        m = mi & ~((1 << (i + 1)) - 1)  # only j > i
        while m:
            j_bit = m & (-m)
            j = j_bit.bit_length() - 1
            m ^= j_bit
            # Common neighbors of i and j with index > j
            common = mask[i] & mask[j] & ~((1 << (j + 1)) - 1)
            while common:
                k_bit = common & (-common)
                k = k_bit.bit_length() - 1
                common ^= k_bit
                s = deg[i] + deg[j] + deg[k]
                if s > max_score:
                    max_score = s
    return max_score

def _compute_score(n, adj):
    """Compute the max triangle degree sum (numpy version, kept for compatibility)."""
    deg = adj.sum(axis=1)
    max_score = 0
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                common = np.where(adj[i] & adj[j])[0]
                for k in common:
                    if k > j:
                        s = deg[i] + deg[j] + deg[k]
                        if s > max_score:
                            max_score = s
    return max_score

def _get_all_edges(n):
    return list(combinations(range(n), 2))

def _get_non_edges(n, edge_set):
    return [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]

def _local_search(n, edges, max_time=100):
    start_time = time.time()
    deadline = start_time + max_time
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    mask = _build_bitmask(n, edges)
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    current_score = _compute_score_fast(n, mask, deg)
    best_score = current_score
    best_edge_set = set(edge_set)
    
    # Precompute all possible non-edges for faster lookup
    all_pairs = set(tuple(sorted(p)) for p in combinations(range(n), 2))
    
    while time.time() < deadline:
        non_edges = all_pairs - edge_set
        if not non_edges:
            break
        
        improved = False
        # Try edge swaps: remove one edge, add another
        for e_remove in list(edge_set):
            if time.time() >= deadline:
                break
            u1, v1 = e_remove
            for e_add in non_edges:
                u2, v2 = e_add
                if (u2, v2) in edge_set:
                    continue
                if u2 == u1 and v2 == v1:
                    continue
                
                # Tentative swap
                mask[u1] &= ~(1 << v1)
                mask[v1] &= ~(1 << u1)
                mask[u2] |= (1 << v2)
                mask[v2] |= (1 << u2)
                deg[u1] -= 1
                deg[v1] -= 1
                deg[u2] += 1
                deg[v2] += 1
                
                new_score = _compute_score_fast(n, mask, deg)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edge_set = set(edge_set)
                    best_edge_set.discard(e_remove)
                    best_edge_set.add(tuple(sorted(e_add)))
                    current_score = new_score
                    improved = True
                    # Commit the swap
                    edge_set.discard(e_remove)
                    edge_set.add(tuple(sorted(e_add)))
                else:
                    # Revert
                    mask[u1] |= (1 << v1)
                    mask[v1] |= (1 << u1)
                    mask[u2] &= ~(1 << v2)
                    mask[v2] &= ~(1 << u2)
                    deg[u1] += 1
                    deg[v1] += 1
                    deg[u2] -= 1
                    deg[v2] -= 1
                
                if improved:
                    break
        
        if not improved:
            break
    
    # Convert back to edge list
    best_edges = [list(e) for e in best_edge_set]
    return best_edges, best_score

def _polish(n, edges, max_time=30):
    """Final polish pass with more aggressive local search."""
    start_time = time.time()
    deadline = start_time + max_time
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    mask = _build_bitmask(n, edges)
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    current_score = _compute_score_fast(n, mask, deg)
    best_score = current_score
    best_edge_set = set(edge_set)
    
    # Multi-start: try different random perturbations
    rng = random.Random(42)
    
    all_pairs = set(tuple(sorted(p)) for p in combinations(range(n), 2))
    
    while time.time() < deadline:
        non_edges = all_pairs - edge_set
        if not non_edges:
            break
        
        improved = False
        # Try all possible swaps
        shuffled_remove = list(edge_set)
        rng.shuffle(shuffled_remove)
        for e_remove in shuffled_remove:
            if time.time() >= deadline:
                break
            u1, v1 = e_remove
            shuffled_add = list(non_edges)
            rng.shuffle(shuffled_add)
            for e_add in shuffled_add:
                u2, v2 = e_add
                if (u2, v2) in edge_set:
                    continue
                if u2 == u1 and v2 == v1:
                    continue
                
                # Tentative swap
                mask[u1] &= ~(1 << v1)
                mask[v1] &= ~(1 << u1)
                mask[u2] |= (1 << v2)
                mask[v2] |= (1 << u2)
                deg[u1] -= 1
                deg[v1] -= 1
                deg[u2] += 1
                deg[v2] += 1
                
                new_score = _compute_score_fast(n, mask, deg)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edge_set = set(edge_set)
                    best_edge_set.discard(e_remove)
                    best_edge_set.add(tuple(sorted(e_add)))
                    current_score = new_score
                    improved = True
                    edge_set.discard(e_remove)
                    edge_set.add(tuple(sorted(e_add)))
                else:
                    mask[u1] |= (1 << v1)
                    mask[v1] |= (1 << u1)
                    mask[u2] &= ~(1 << v2)
                    mask[v2] &= ~(1 << u2)
                    deg[u1] += 1
                    deg[v1] += 1
                    deg[u2] -= 1
                    deg[v2] -= 1
                
                if improved:
                    break
        
        if not improved:
            break
    
    best_edges = [list(e) for e in best_edge_set]
    return best_edges, best_score

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    total_deadline = 145  # safety margin before 150s limit
    
    # Start from K_{6,10} + 5 internal edges in the 10-vertex part
    # Part A: vertices 0-5 (6 vertices), Part B: vertices 6-15 (10 vertices)
    # All cross edges: 6*10 = 60
    # Need 5 more edges within Part B
    # Choose 5 edges in Part B to minimize degree sums in triangles
    
    initial_edges = []
    # Cross edges
    for u in range(6):
        for v in range(6, 16):
            initial_edges.append([u, v])
    
    # 5 internal edges in Part B (vertices 6-15)
    # These create triangles with cross edges
    # Each internal edge (u,v) in B creates triangles with each vertex in A
    # deg(u) = 6 + (internal degree in B), deg(v) = 6 + (internal degree in B)
    # Triangle (a, u, v) where a in A: deg(a) + deg(u) + deg(v) = 10 + (6+du) + (6+dv)
    # = 22 + du + dv where du, dv are internal degrees
    
    # To minimize max triangle sum, we want to minimize max(du+dv) over internal edges
    # With 5 edges on 10 vertices, we can use a matching + 1 extra
    # Matching of 5: each vertex has degree 1, so du+dv = 2 for each edge
    # That gives triangle sum = 24
    # Alternative: star-like structures might be worse
    
    # Use a matching: (6,7), (8,9), (10,11), (12,13), (14,15)
    internal_edges = [(6,7), (8,9), (10,11), (12,13), (14,15)]
    for u, v in internal_edges:
        initial_edges.append([u, v])
    
    # Verify: 60 + 5 = 65 edges
    assert len(initial_edges) == 65
    
    # Run local search with most of the time
    elapsed = time.time() - start_time
    search_time = min(110, total_deadline - 15)
    best_edges, best_score = _local_search(n, initial_edges, max_time=search_time)
    
    # Polish pass with remaining time
    elapsed = time.time() - start_time
    polish_time = min(30, max(5, total_deadline - 15 - elapsed))
    best_edges, best_score = _polish(n, best_edges, max_time=polish_time)
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
