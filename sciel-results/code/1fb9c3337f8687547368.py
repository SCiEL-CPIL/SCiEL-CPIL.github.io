import time
import random
from collections import Counter

DEADLINE = 115  # Safety margin for 150s limit

def _compute_score(n, adj_masks):
    """
    Computes the maximum degree sum over all triangles in the graph.
    n: number of vertices
    adj_masks: list of int bitmasks representing adjacency
    """
    deg = [bin(mask).count('1') for mask in adj_masks]
    max_sum = 0
    max_triangles = []

    for i in range(n):
        mi = adj_masks[i]
        mask_j_greater = mi & ~((1 << (i + 1)) - 1)

        while mask_j_greater:
            lsb = mask_j_greater & -mask_j_greater
            j = lsb.bit_length() - 1
            mask_j_greater ^= lsb

            common = mi & adj_masks[j]
            mask_k = common
            while mask_k:
                lsb_k = mask_k & -mask_k
                k = lsb_k.bit_length() - 1
                mask_k ^= lsb_k
                s = deg[i] + deg[j] + deg[k]
                if s > max_sum:
                    max_sum = s
                    max_triangles = [(i, j, k)]
                elif s == max_sum:
                    max_triangles.append((i, j, k))

    return max_sum, max_triangles

def _find_worst_triangle(n, adj_masks, deg):
    """Find the triangle with maximum degree sum. Returns (score, (i,j,k)) or (0, None)."""
    worst_sum = 0
    worst_tri = None
    for i in range(n):
        mi = adj_masks[i]
        mask_j_greater = mi & ~((1 << (i + 1)) - 1)
        while mask_j_greater:
            lsb = mask_j_greater & -mask_j_greater
            j = lsb.bit_length() - 1
            mask_j_greater ^= lsb
            common = mi & adj_masks[j]
            mask_k = common
            while mask_k:
                lsb_k = mask_k & -mask_k
                k = lsb_k.bit_length() - 1
                mask_k ^= lsb_k
                s = deg[i] + deg[j] + deg[k]
                if s > worst_sum:
                    worst_sum = s
                    worst_tri = (i, j, k)
    return worst_sum, worst_tri

def _delta_score_remove_add(n, adj_masks, deg, rem_u, rem_v, add_u, add_v):
    """
    Fast evaluation: compute new max triangle degree sum after removing edge (rem_u,rem_v)
    and adding edge (add_u,add_v).
    Returns the new max score, or None if the swap is invalid (e.g., add edge already exists).
    """
    if (adj_masks[add_u] >> add_v) & 1:
        return None
    if (adj_masks[rem_u] >> rem_v) & 1 == 0:
        return None

    nd = list(deg)
    nd[rem_u] -= 1
    nd[rem_v] -= 1
    nd[add_u] += 1
    nd[add_v] += 1

    nm = list(adj_masks)
    nm[rem_u] &= ~(1 << rem_v)
    nm[rem_v] &= ~(1 << rem_u)
    nm[add_u] |= (1 << add_v)
    nm[add_v] |= (1 << add_u)

    max_sum = 0
    for i in range(n):
        mi = nm[i]
        mask_j_greater = mi & ~((1 << (i + 1)) - 1)
        while mask_j_greater:
            lsb = mask_j_greater & -mask_j_greater
            j = lsb.bit_length() - 1
            mask_j_greater ^= lsb
            common = mi & nm[j]
            mask_k = common
            while mask_k:
                lsb_k = mask_k & -mask_k
                k = lsb_k.bit_length() - 1
                mask_k ^= lsb_k
                s = nd[i] + nd[j] + nd[k]
                if s > max_sum:
                    max_sum = s
    return max_sum

def _smart_swap_search(n, adj_masks, deg, current_score, current_max_tris, current_edge_set, non_edge_set):
    """
    Degree-ordered edge-replacement heuristic:
    1. Identify the worst triangle(s)
    2. For each edge in the worst triangle, try to find a non-edge swap that
       eliminates the worst triangle and doesn't create new high-score triangles
    3. Prioritize swaps that reduce the degree of the highest-degree vertex
    
    Returns (rem_u, rem_v, add_u, add_v, new_score) or None if no improvement found.
    """
    if not current_max_tris:
        return None
    
    # Get unique worst triangles
    worst_tris = list(set(current_max_tris))
    
    # For each edge in the worst triangles, try swaps
    # Priority: edges involving highest-degree vertices first
    edge_candidates = []
    for tri in worst_tris:
        i, j, k = tri
        edges = [(i, j), (j, k), (i, k)]
        for e in edges:
            a, b = e
            # Score this edge by the degrees of its endpoints (higher = more urgent)
            edge_score = deg[a] + deg[b]
            edge_candidates.append((edge_score, a, b))
    
    # Sort by edge score descending (most urgent first)
    edge_candidates.sort(key=lambda x: -x[0])
    
    # Remove duplicates while preserving order
    seen_edges = set()
    unique_edge_candidates = []
    for score, a, b in edge_candidates:
        edge = (min(a, b), max(a, b))
        if edge not in seen_edges:
            seen_edges.add(edge)
            unique_edge_candidates.append((score, a, b))
    
    # For each candidate edge to remove, find a good edge to add
    non_edges_list = list(non_edge_set)
    
    for _, rem_u, rem_v in unique_edge_candidates:
        # Check if removing this edge eliminates all current worst triangles
        all_removed = True
        for tri in worst_tris:
            tu, tv, tw = tri
            if not ((rem_u == tu and rem_v == tv) or (rem_u == tu and rem_v == tw) or 
                    (rem_u == tv and rem_v == tu) or (rem_u == tv and rem_v == tw) or
                    (rem_u == tw and rem_v == tu) or (rem_u == tw and rem_v == tv)):
                all_removed = False
                break
        
        if not all_removed:
            continue
        
        # Now find a non-edge to add that doesn't create triangles with score >= current_score
        # Sort non-edges by potential impact (lower degree sum first)
        non_edges_sorted = []
        for add_u, add_v in non_edges_list:
            # Estimate the maximum triangle score this could create
            # New triangles: (add_u, add_v, w) where w is common neighbor
            # Score = deg[add_u]+1 + deg[add_v]+1 + deg[w] (approximately)
            # We want this to be < current_score
            max_common_deg = 0
            common = adj_masks[add_u] & adj_masks[add_v]
            mask_w = common
            while mask_w:
                lsb_w = mask_w & -mask_w
                w = lsb_w.bit_length() - 1
                mask_w ^= lsb_w
                if deg[w] > max_common_deg:
                    max_common_deg = deg[w]
            
            # Estimated max score from new triangles
            est_score = deg[add_u] + 1 + deg[add_v] + 1 + max_common_deg
            non_edges_sorted.append((est_score, add_u, add_v))
        
        # Sort by estimated score ascending (least risky first)
        non_edges_sorted.sort(key=lambda x: x[0])
        
        # Try the most promising non-edges
        for est_score, add_u, add_v in non_edges_sorted:
            # Quick check: if estimated score is already too high, skip
            if est_score >= current_score:
                continue
            
            # Full verification with delta score
            new_score = _delta_score_remove_add(n, adj_masks, deg, rem_u, rem_v, add_u, add_v)
            if new_score is not None and new_score < current_score:
                return (rem_u, rem_v, add_u, add_v, new_score)
    
    return None

def _construct_edges_from_masks(adj_masks, n):
    edges = []
    for i in range(n):
        mi = adj_masks[i]
        mask_j_greater = mi & ~((1 << (i + 1)) - 1)
        while mask_j_greater:
            lsb = mask_j_greater & -mask_j_greater
            j = lsb.bit_length() - 1
            edges.append([i, j])
            mask_j_greater ^= lsb
    return edges

def construct():
    start_time = time.time()
    n = 16
    target_edges = 65

    # Initial state: K_{8,8} + one internal edge (0,1)
    adj_masks = [0] * n
    for u in range(8):
        for v in range(8, 16):
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
    adj_masks[0] |= (1 << 1)
    adj_masks[1] |= (1 << 0)

    current_score, current_max_tris = _compute_score(n, adj_masks)
    best_score = current_score
    best_adj_masks = list(adj_masks)

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Use sets for O(1) membership and removal
    current_edge_set = set()
    non_edge_set = set()

    for e in all_edges:
        u, v = e
        if (adj_masks[u] >> v) & 1:
            current_edge_set.add(e)
        else:
            non_edge_set.add(e)

    random.seed(42)

    max_iterations = 100000
    iterations = 0
    stuck_count = 0
    deg = [bin(mask).count('1') for mask in adj_masks]

    while time.time() - start_time < DEADLINE and iterations < max_iterations:
        iterations += 1

        improved = False

        # Use the smart swap search heuristic
        swap_result = _smart_swap_search(n, adj_masks, deg, current_score, current_max_tris, current_edge_set, non_edge_set)
        
        if swap_result is not None:
            rem_u, rem_v, add_u, add_v, new_score = swap_result
            
            # Apply the swap
            adj_masks[rem_u] ^= (1 << rem_v)
            adj_masks[rem_v] ^= (1 << rem_u)
            adj_masks[add_u] ^= (1 << add_v)
            adj_masks[add_v] ^= (1 << add_u)

            rem_edge = (min(rem_u, rem_v), max(rem_u, rem_v))
            add_edge = (min(add_u, add_v), max(add_u, add_v))
            current_edge_set.discard(rem_edge)
            non_edge_set.add(rem_edge)
            non_edge_set.discard(add_edge)
            current_edge_set.add(add_edge)

            deg[rem_u] -= 1
            deg[rem_v] -= 1
            deg[add_u] += 1
            deg[add_v] += 1

            current_score = new_score
            current_score, current_max_tris = _compute_score(n, adj_masks)
            improved = True
            stuck_count = 0
            if current_score < best_score:
                best_score = current_score
                best_adj_masks = list(adj_masks)
        
        if not improved:
            # Try a random swap as exploration
            if non_edge_set and current_edge_set:
                rem_u, rem_v = random.choice(list(current_edge_set))
                add_u, add_v = random.choice(list(non_edge_set))

                rem_edge = (min(rem_u, rem_v), max(rem_u, rem_v))
                add_edge = (min(add_u, add_v), max(add_u, add_v))

                # Fast check: does it improve?
                new_score = _delta_score_remove_add(n, adj_masks, deg, rem_u, rem_v, add_u, add_v)
                if new_score is not None:
                    # Perform swap
                    adj_masks[rem_u] ^= (1 << rem_v)
                    adj_masks[rem_v] ^= (1 << rem_u)
                    current_edge_set.discard(rem_edge)
                    non_edge_set.add(rem_edge)

                    adj_masks[add_u] ^= (1 << add_v)
                    adj_masks[add_v] ^= (1 << add_u)
                    non_edge_set.discard(add_edge)
                    current_edge_set.add(add_edge)

                    deg[rem_u] -= 1
                    deg[rem_v] -= 1
                    deg[add_u] += 1
                    deg[add_v] += 1

                    if new_score < current_score:
                        current_score = new_score
                        current_score, current_max_tris = _compute_score(n, adj_masks) # Update max tris
                        if current_score < best_score:
                            best_score = current_score
                            best_adj_masks = list(adj_masks)
                        stuck_count = 0
                    else:
                        # Revert
                        adj_masks[rem_u] ^= (1 << rem_v)
                        adj_masks[rem_v] ^= (1 << rem_u)
                        non_edge_set.discard(rem_edge)
                        current_edge_set.add(rem_edge)

                        adj_masks[add_u] ^= (1 << add_v)
                        adj_masks[add_v] ^= (1 << add_u)
                        current_edge_set.discard(add_edge)
                        non_edge_set.add(add_edge)

                        deg[rem_u] += 1
                        deg[rem_v] += 1
                        deg[add_u] -= 1
                        deg[add_v] -= 1
                        stuck_count += 1

                # Perturbation if stuck
                if stuck_count > 30:
                    for _ in range(3):
                        if non_edge_set and current_edge_set:
                            r_u, r_v = random.choice(list(current_edge_set))
                            a_u, a_v = random.choice(list(non_edge_set))

                            r_edge = (min(r_u, r_v), max(r_u, r_v))
                            a_edge = (min(a_u, a_v), max(a_u, a_v))

                            adj_masks[r_u] ^= (1 << r_v)
                            adj_masks[r_v] ^= (1 << r_u)
                            current_edge_set.discard(r_edge)
                            non_edge_set.add(r_edge)

                            adj_masks[a_u] ^= (1 << a_v)
                            adj_masks[a_v] ^= (1 << a_u)
                            non_edge_set.discard(a_edge)
                            current_edge_set.add(a_edge)

                            deg[r_u] -= 1
                            deg[r_v] -= 1
                            deg[a_u] += 1
                            deg[a_v] += 1

                    current_score, current_max_tris = _compute_score(n, adj_masks)
                    stuck_count = 0

    best_edges = _construct_edges_from_masks(best_adj_masks, n)
    return {'n': n, 'edges': best_edges}
