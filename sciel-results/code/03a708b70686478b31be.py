"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def _build_bitmask(n, edges):
    """Build adjacency as list of bitmasks for fast triangle detection."""
    mask = [0] * n
    for u, v in edges:
        mask[u] |= (1 << v)
        mask[v] |= (1 << u)
    return mask

def _compute_all_triangles(n, mask):
    """Return list of all triangles as (i, j, k) with i < j < k."""
    triangles = []
    for i in range(n):
        mi = mask[i]
        m = mi & ~((1 << (i + 1)) - 1)  # only j > i
        while m:
            j_bit = m & (-m)
            j = j_bit.bit_length() - 1
            m ^= j_bit
            common = mask[i] & mask[j] & ~((1 << (j + 1)) - 1)
            while common:
                k_bit = common & (-common)
                k = k_bit.bit_length() - 1
                common ^= k_bit
                triangles.append((i, j, k))
    return triangles

def _compute_score_from_triangles(triangles, deg):
    """Compute max triangle degree sum from a list of triangles."""
    max_score = 0
    for i, j, k in triangles:
        s = deg[i] + deg[j] + deg[k]
        if s > max_score:
            max_score = s
    return max_score

def _compute_score_fast(n, mask, deg):
    """Compute max triangle degree sum using bitmask operations."""
    triangles = _compute_all_triangles(n, mask)
    return _compute_score_from_triangles(triangles, deg)

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

def _get_affected_triangles(n, mask, u1, v1, u2, v2, existing_triangles):
    """
    Get triangles affected by removing edge (u1,v1) and adding edge (u2,v2).
    Returns (removed_triangles, added_triangles) as sets of frozensets.
    """
    removed = set()
    added = set()
    
    # Triangles containing edge (u1, v1) are removed
    for tri in existing_triangles:
        i, j, k = tri
        if (i == u1 and j == v1) or (i == v1 and j == u1) or \
           (i == u1 and k == v1) or (i == v1 and k == u1) or \
           (j == u1 and k == v1) or (j == v1 and k == u1):
            removed.add(frozenset(tri))
    
    # Triangles containing edge (u2, v2) are added (only if both other vertices connect to both u2 and v2)
    # For each common neighbor w of u2 and v2, (u2, v2, w) is a triangle
    common = mask[u2] & mask[v2]
    while common:
        w_bit = common & (-common)
        w = w_bit.bit_length() - 1
        common ^= w_bit
        tri = tuple(sorted((u2, v2, w)))
        added.add(frozenset(tri))
    
    return removed, added

def _local_search(n, edges, max_time=100):
    start_time = time.time()
    deadline = start_time + max_time
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    mask = _build_bitmask(n, edges)
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    triangles = _compute_all_triangles(n, mask)
    current_score = _compute_score_from_triangles(triangles, deg)
    best_score = current_score
    best_edge_set = set(edge_set)
    best_triangles = list(triangles)
    
    while time.time() < deadline:
        non_edges = _get_non_edges(n, edge_set)
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
                
                # Get affected triangles
                removed_tris, added_tris = _get_affected_triangles(n, mask, u1, v1, u2, v2, triangles)
                
                # Check if this swap is valid (no duplicate edges, etc.)
                # Tentatively update degrees
                old_deg_u1, old_deg_v1 = deg[u1], deg[v1]
                old_deg_u2, old_deg_v2 = deg[u2], deg[v2]
                deg[u1] -= 1
                deg[v1] -= 1
                deg[u2] += 1
                deg[v2] += 1
                
                # Compute new score considering only affected triangles
                # New triangles = old - removed + added
                # We need max over all triangles, but only affected ones change
                # Unaffected triangles have same degree sums as before
                
                # Find max among unaffected triangles
                # This is expensive; instead, just recompute from scratch if needed
                # For now, use a heuristic: check if the max could decrease
                
                # Actually, let's just do a quick check: what's the max degree sum in added triangles?
                max_added = 0
                for tri in added_tris:
                    i, j, k = tuple(sorted(tri))
                    s = deg[i] + deg[j] + deg[k]
                    if s > max_added:
                        max_added = s
                
                # What was the max among removed triangles (with old degrees)?
                max_removed_old = 0
                for tri in removed_tris:
                    i, j, k = tuple(sorted(tri))
                    s = deg[i] + deg[j] + deg[k] + (old_deg_u1 - deg[u1]) * (1 if i == u1 or j == u1 or k == u1 else 0) + \
                              (old_deg_v1 - deg[v1]) * (1 if i == v1 or j == v1 or k == v1 else 0)
                    if s > max_removed_old:
                        max_removed_old = s
                
                # If max_added < current_score and we're removing some high-score triangles, this might help
                # Simple heuristic: try the swap and compute full score
                # But that's expensive. Let's use a cheaper filter first.
                
                # Quick filter: if all added triangles have score < current_score, and we're removing
                # at least one triangle with score >= current_score, this swap is promising.
                promising = True
                if max_added >= current_score:
                    promising = False
                
                if not promising:
                    # Revert degrees
                    deg[u1] = old_deg_u1
                    deg[v1] = old_deg_v1
                    deg[u2] = old_deg_u2
                    deg[v2] = old_deg_v2
                    continue
                
                # Full computation for promising swaps
                # Apply the swap
                mask[u1] &= ~(1 << v1)
                mask[v1] &= ~(1 << u1)
                mask[u2] |= (1 << v2)
                mask[v2] |= (1 << u2)
                
                # Recompute triangles
                new_triangles = _compute_all_triangles(n, mask)
                new_score = _compute_score_from_triangles(new_triangles, deg)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edge_set = set(edge_set)
                    best_edge_set.discard(e_remove)
                    best_edge_set.add(tuple(sorted(e_add)))
                    current_score = new_score
                    improved = True
                    edge_set.discard(e_remove)
                    edge_set.add(tuple(sorted(e_add)))
                    triangles = new_triangles
                    best_triangles = list(triangles)
                else:
                    # Revert
                    mask[u1] |= (1 << v1)
                    mask[v1] |= (1 << u1)
                    mask[u2] &= ~(1 << v2)
                    mask[v2] &= ~(1 << u2)
                    deg[u1] = old_deg_u1
                    deg[v1] = old_deg_v1
                    deg[u2] = old_deg_u2
                    deg[v2] = old_deg_v2
                
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
    
    triangles = _compute_all_triangles(n, mask)
    current_score = _compute_score_from_triangles(triangles, deg)
    best_score = current_score
    best_edge_set = set(edge_set)
    best_triangles = list(triangles)
    
    import random
    rng = random.Random(42)
    
    while time.time() < deadline:
        non_edges = _get_non_edges(n, edge_set)
        if not non_edges:
            break
        
        improved = False
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
                
                # Quick filter
                removed_tris, added_tris = _get_affected_triangles(n, mask, u1, v1, u2, v2, triangles)
                
                max_added = 0
                for tri in added_tris:
                    i, j, k = tuple(sorted(tri))
                    s = deg[i] + deg[j] + deg[k]
                    if s > max_added:
                        max_added = s
                
                if max_added >= current_score:
                    continue
                
                # Apply swap
                old_deg_u1, old_deg_v1 = deg[u1], deg[v1]
                old_deg_u2, old_deg_v2 = deg[u2], deg[v2]
                deg[u1] -= 1
                deg[v1] -= 1
                deg[u2] += 1
                deg[v2] += 1
                
                mask[u1] &= ~(1 << v1)
                mask[v1] &= ~(1 << u1)
                mask[u2] |= (1 << v2)
                mask[v2] |= (1 << u2)
                
                new_triangles = _compute_all_triangles(n, mask)
                new_score = _compute_score_from_triangles(new_triangles, deg)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edge_set = set(edge_set)
                    best_edge_set.discard(e_remove)
                    best_edge_set.add(tuple(sorted(e_add)))
                    current_score = new_score
                    improved = True
                    edge_set.discard(e_remove)
                    edge_set.add(tuple(sorted(e_add)))
                    triangles = new_triangles
                    best_triangles = list(triangles)
                else:
                    mask[u1] |= (1 << v1)
                    mask[v1] |= (1 << u1)
                    mask[u2] &= ~(1 << v2)
                    mask[v2] &= ~(1 << u2)
                    deg[u1] = old_deg_u1
                    deg[v1] = old_deg_v1
                    deg[u2] = old_deg_u2
                    deg[v2] = old_deg_v2
                
                if improved:
                    break
        
        if not improved:
            break
    
    best_edges = [list(e) for e in best_edge_set]
    return best_edges, best_score

def construct():
    n = 16
    target_edges = 65
    total_deadline = 140  # safety margin before 150s limit
    
    # Start from K_{6,10} + 5 internal edges in the 10-vertex part
    initial_edges = []
    for u in range(6):
        for v in range(6, 16):
            initial_edges.append([u, v])
    
    internal_edges = [(6,7), (8,9), (10,11), (12,13), (14,15)]
    for u, v in internal_edges:
        initial_edges.append([u, v])
    
    assert len(initial_edges) == 65
    
    search_time = min(100, total_deadline - 10)
    best_edges, best_score = _local_search(n, initial_edges, max_time=search_time)
    
    polish_time = min(30, max(5, total_deadline - 10 - search_time))
    best_edges, best_score = _polish(n, best_edges, max_time=polish_time)
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
