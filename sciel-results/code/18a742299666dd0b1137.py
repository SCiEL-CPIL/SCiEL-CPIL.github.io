"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
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
        m = mi & ~((1 << (i + 1)) - 1)
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
    
    for tri in existing_triangles:
        i, j, k = tri
        if (i == u1 and j == v1) or (i == v1 and j == u1) or \
           (i == u1 and k == v1) or (i == v1 and k == u1) or \
           (j == u1 and k == v1) or (j == v1 and k == u1):
            removed.add(frozenset(tri))
    
    common = mask[u2] & mask[v2]
    while common:
        w_bit = common & (-common)
        w = w_bit.bit_length() - 1
        common ^= w_bit
        tri = tuple(sorted((u2, v2, w)))
        added.add(frozenset(tri))
    
    return removed, added

def _quick_filter(n, mask, deg, current_score, u1, v1, u2, v2, triangles):
    """
    Quick filter: return True if the swap might improve the score.
    Checks if all added triangles have degree sum < current_score
    and we're removing at least one triangle with degree sum >= current_score.
    """
    # Compute new degrees
    new_deg_u1 = deg[u1] - 1
    new_deg_v1 = deg[v1] - 1
    new_deg_u2 = deg[u2] + 1
    new_deg_v2 = deg[v2] + 1
    
    # Check added triangles
    common = mask[u2] & mask[v2]
    while common:
        w_bit = common & (-common)
        w = w_bit.bit_length() - 1
        common ^= w_bit
        # Triangle (u2, v2, w) with new degrees
        d_u2 = new_deg_u2
        d_v2 = new_deg_v2
        d_w = deg[w]
        s = d_u2 + d_v2 + d_w
        if s >= current_score:
            return False
    
    # Check if we're removing any triangle with score >= current_score
    for tri in triangles:
        i, j, k = tri
        if (i == u1 and j == v1) or (i == v1 and j == u1) or \
           (i == u1 and k == v1) or (i == v1 and k == u1) or \
           (j == u1 and k == v1) or (j == v1 and k == u1):
            # This triangle is removed
            d_i = deg[i] - (1 if i == u1 else 0) - (1 if i == v1 else 0)
            d_j = deg[j] - (1 if j == u1 else 0) - (1 if j == v1 else 0)
            d_k = deg[k] - (1 if k == u1 else 0) - (1 if k == v1 else 0)
            s = d_i + d_j + d_k
            if s >= current_score:
                return True
    
    return False

def _apply_swap(n, mask, deg, edge_set, triangles, u1, v1, u2, v2):
    """
    Apply a swap: remove edge (u1,v1), add edge (u2,v2).
    Returns (new_triangles, new_score, old_deg_u1, old_deg_v1, old_deg_u2, old_deg_v2).
    """
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
    
    edge_set.discard((u1, v1))
    edge_set.discard((v1, u1))
    edge_set.add((u2, v2))
    edge_set.add((v2, u2))
    
    new_triangles = _compute_all_triangles(n, mask)
    new_score = _compute_score_from_triangles(new_triangles, deg)
    
    return new_triangles, new_score, old_deg_u1, old_deg_v1, old_deg_u2, old_deg_v2

def _revert_swap(n, mask, deg, edge_set, triangles, u1, v1, u2, v2,
                 old_deg_u1, old_deg_v1, old_deg_u2, old_deg_v2):
    """Revert a swap."""
    deg[u1] = old_deg_u1
    deg[v1] = old_deg_v1
    deg[u2] = old_deg_u2
    deg[v2] = old_deg_v2
    
    mask[u1] |= (1 << v1)
    mask[v1] |= (1 << u1)
    mask[u2] &= ~(1 << v2)
    mask[v2] &= ~(1 << u2)
    
    edge_set.discard((u2, v2))
    edge_set.discard((v2, u2))
    edge_set.add((u1, v1))
    edge_set.add((v1, u1))

def _simulated_annealing(n, edges, max_time=60, seed=42):
    """
    Simulated annealing on edge-swap neighborhood.
    Accepts temporary score increases to escape local minima.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    rng = random.Random(seed)
    
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
    best_mask = list(mask)
    best_deg = list(deg)
    best_triangles = list(triangles)
    
    # Temperature schedule
    T_initial = 5.0
    T_final = 0.01
    cooling_rate = 0.9995
    
    T = T_initial
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        non_edges = _get_non_edges(n, edge_set)
        if not non_edges:
            break
        
        # Pick a random edge to remove
        e_remove = rng.choice(list(edge_set))
        u1, v1 = e_remove
        
        # Pick a random non-edge to add
        e_add = rng.choice(non_edges)
        u2, v2 = e_add
        
        if (u2, v2) in edge_set or (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
            continue
        
        # Apply swap
        new_triangles, new_score, old_d1, old_d2, old_d3, old_d4 = _apply_swap(
            n, mask, deg, edge_set, triangles, u1, v1, u2, v2
        )
        
        delta = new_score - current_score
        
        if delta <= 0:
            # Accept improvement
            current_score = new_score
            triangles = new_triangles
            no_improve_count = 0
            
            if new_score < best_score:
                best_score = new_score
                best_edge_set = set(edge_set)
                best_mask = list(mask)
                best_deg = list(deg)
                best_triangles = list(triangles)
        else:
            # Accept with probability exp(-delta / T)
            import math
            if T > 0 and rng.random() < math.exp(-delta / T):
                # Accept worsening
                current_score = new_score
                triangles = new_triangles
                no_improve_count += 1
            else:
                # Reject
                _revert_swap(n, mask, deg, edge_set, triangles, u1, v1, u2, v2,
                            old_d1, old_d2, old_d3, old_d4)
                no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_final:
            T = T_final
        
        # Reset temperature if stuck too long
        if no_improve_count > max_no_improve:
            T = T_initial * 0.5
            no_improve_count = 0
    
    best_edges = [list(e) for e in best_edge_set]
    return best_edges, best_score

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
                
                # Quick filter
                if not _quick_filter(n, mask, deg, current_score, u1, v1, u2, v2, triangles):
                    continue
                
                # Full computation for promising swaps
                new_triangles, new_score, old_d1, old_d2, old_d3, old_d4 = _apply_swap(
                    n, mask, deg, edge_set, triangles, u1, v1, u2, v2
                )
                
                if new_score < best_score:
                    best_score = new_score
                    best_edge_set = set(edge_set)
                    best_triangles = list(triangles)
                    current_score = new_score
                    triangles = new_triangles
                    improved = True
                else:
                    _revert_swap(n, mask, deg, edge_set, triangles, u1, v1, u2, v2,
                                old_d1, old_d2, old_d3, old_d4)
                
                if improved:
                    break
        
        if not improved:
            break
    
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
                if not _quick_filter(n, mask, deg, current_score, u1, v1, u2, v2, triangles):
                    continue
                
                # Apply swap
                new_triangles, new_score, old_d1, old_d2, old_d3, old_d4 = _apply_swap(
                    n, mask, deg, edge_set, triangles, u1, v1, u2, v2
                )
                
                if new_score < best_score:
                    best_score = new_score
                    best_edge_set = set(edge_set)
                    best_triangles = list(triangles)
                    current_score = new_score
                    triangles = new_triangles
                    improved = True
                else:
                    _revert_swap(n, mask, deg, edge_set, triangles, u1, v1, u2, v2,
                                old_d1, old_d2, old_d3, old_d4)
                
                if improved:
                    break
        
        if not improved:
            break
    
    best_edges = [list(e) for e in best_edge_set]
    return best_edges, best_score

def _diverse_start(n, seed=0):
    """Generate a diverse starting construction."""
    rng = random.Random(seed)
    
    # K_{6,10} + 5 internal edges in the 10-vertex part
    initial_edges = []
    for u in range(6):
        for v in range(6, 16):
            initial_edges.append([u, v])
    
    # Random internal edges
    internal_vertices = list(range(6, 16))
    possible_internal = list(combinations(internal_vertices, 2))
    rng.shuffle(possible_internal)
    internal_edges = possible_internal[:5]
    for u, v in internal_edges:
        initial_edges.append([u, v])
    
    assert len(initial_edges) == 65
    return initial_edges

def construct():
    n = 16
    target_edges = 65
    total_deadline = 140  # safety margin before 150s limit
    
    # Start from K_{6,10} + 5 internal edges in the 10-vertex part
    initial_edges = _diverse_start(n, seed=0)
    
    # Phase 1: Deterministic local search
    search_time = min(30, total_deadline - 80)
    best_edges, best_score = _local_search(n, initial_edges, max_time=search_time)
    
    # Phase 2: Simulated annealing
    sa_time = min(60, max(20, total_deadline - 40 - search_time))
    sa_edges, sa_score = _simulated_annealing(n, best_edges, max_time=sa_time, seed=42)
    if sa_score < best_score:
        best_edges, best_score = sa_edges, sa_score
    
    # Phase 3: Polish
    polish_time = min(30, max(5, total_deadline - search_time - sa_time - 10))
    best_edges, best_score = _polish(n, best_edges, max_time=polish_time)
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
