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

        # Check if we can improve by removing all max triangles
        # We need to find a swap that removes all triangles in current_max_tris
        # and does not create any new triangle with score >= current_score
        
        # Heuristic: Focus on edges in the worst triangle
        if current_max_tris:
            # Pick the first worst triangle
            i, j, k = current_max_tris[0]
            tri_vertices = [(i, deg[i]), (j, deg[j]), (k, deg[k])]
            tri_vertices.sort(key=lambda x: -x[1])
            top_vertex = tri_vertices[0][0]

            edges_to_try = []
            other_edges = []
            for (a, b) in [(i, j), (j, k), (i, k)]:
                if a == top_vertex or b == top_vertex:
                    edges_to_try.append((a, b))
                else:
                    other_edges.append((a, b))
            edges_to_try.extend(other_edges)

            for rem_u, rem_v in edges_to_try:
                # Check if this removal eliminates all current_max_tris
                # A triangle is eliminated if it contains the removed edge
                # Note: current_max_tris is a list of (i,j,k)
                
                # Optimization: Only try if this edge is in ALL max triangles?
                # If not, removing it won't eliminate all max triangles, so score won't drop.
                # Unless the new max is lower than the old max, which requires all old max triangles to be gone.
                
                all_removed = True
                for t in current_max_tris:
                    tu, tv, tw = t
                    if not ((rem_u == tu and rem_v == tv) or (rem_u == tu and rem_v == tw) or 
                            (rem_u == tv and rem_v == tu) or (rem_u == tv and rem_v == tw) or
                            (rem_u == tw and rem_v == tu) or (rem_u == tw and rem_v == tv)):
                        all_removed = False
                        break
                
                if not all_removed:
                    continue

                # Now check if adding any non-edge creates a triangle with score >= current_score
                # We can sample some non-edges to see if this is promising
                # To save time, we just pick a few random non-edges or the most promising ones
                # Let's try a few random ones first
                
                # Get a list of non-edges (limit size for speed)
                # We want non-edges that don't create high degree sums
                # Approx: deg[u] + deg[v] + max_common_neighbor_deg < current_score
                
                found_improvement = False
                # Shuffle non_edges to randomize
                non_edges_list = list(non_edge_set)
                random.shuffle(non_edges_list)
                
                for add_u, add_v in non_edges_list[:50]:
                    # Check if adding (add_u, add_v) creates a triangle with score >= current_score
                    # New triangles are (add_u, add_v, w) where w is common neighbor
                    common = adj_masks[add_u] & adj_masks[add_v]
                    # Note: common neighbors in old graph are the same as in new graph
                    # because we only changed degrees of add_u, add_v and removed an edge elsewhere
                    # Wait, degrees change.
                    # deg_new[add_u] = deg[add_u] + 1
                    # deg_new[add_v] = deg[add_v] + 1
                    # deg_new[w] = deg[w] (if w is not rem_u/rem_v)
                    # If w is rem_u or rem_v, degree changes too.
                    
                    # Let's just do a fast check for the new triangles
                    # If any new triangle has score >= current_score, reject this add edge
                    safe = True
                    # We need to check all w in common
                    # Also, if rem_u/rem_v is in common, its degree decreases by 1
                    # If rem_u/rem_v is add_u/add_v, its degree increases by 1 (but add_u/add_v are distinct from rem_u/rem_v usually? No, they can overlap)
                    # Actually, if add_u == rem_u, then deg_new[add_u] = deg[rem_u] - 1 + 1 = deg[rem_u].
                    
                    # To be safe, let's just calculate the score for the new triangles
                    # This is O(degree) per candidate
                    mask_w = common
                    while mask_w:
                        lsb_w = mask_w & -mask_w
                        w = lsb_w.bit_length() - 1
                        mask_w ^= lsb_w
                        
                        d_u = deg[add_u]
                        d_v = deg[add_v]
                        d_w = deg[w]
                        
                        # Adjust for swap
                        if add_u == rem_u: d_u -= 1 # -1 from rem, +1 from add -> 0 net change? No.
                        # nd[rem_u] -= 1
                        # nd[add_u] += 1
                        # If add_u == rem_u: nd[add_u] = deg[add_u] - 1 + 1 = deg[add_u]
                        # If add_u != rem_u: nd[add_u] = deg[add_u] + 1
                        # Same for v, w.
                        
                        if add_u == rem_u:
                            d_u = deg[add_u]
                        else:
                            d_u = deg[add_u] + 1
                            
                        if add_v == rem_v:
                            d_v = deg[add_v]
                        else:
                            d_v = deg[add_v] + 1
                            
                        if w == rem_u:
                            d_w = deg[w] - 1
                        elif w == rem_v:
                            d_w = deg[w] - 1
                        else:
                            d_w = deg[w]
                            
                        s = d_u + d_v + d_w
                        if s >= current_score:
                            safe = False
                            break
                    
                    if safe:
                        # This swap is a candidate
                        # Apply it
                        adj_masks[rem_u] ^= (1 << rem_v)
                        adj_masks[rem_v] ^= (1 << rem_u)
                        adj_masks[add_u] ^= (1 << add_v)
                        adj_masks[add_v] ^= (1 << add_u)

                        rem_edge = (min(rem_u, rem_v), max(rem_u, rem_v))
                        current_edge_set.discard(rem_edge)
                        non_edge_set.add(rem_edge)
                        non_edge_set.discard((min(add_u, add_v), max(add_u, add_v)))
                        current_edge_set.add((min(add_u, add_v), max(add_u, add_v)))

                        # Update degrees
                        deg[rem_u] -= 1
                        deg[rem_v] -= 1
                        deg[add_u] += 1
                        deg[add_v] += 1

                        # Recompute score and max triangles
                        current_score, current_max_tris = _compute_score(n, adj_masks)
                        improved = True
                        found_improvement = True
                        if current_score < best_score:
                            best_score = current_score
                            best_adj_masks = list(adj_masks)
                        break

                if found_improvement:
                    break

        if not improved:
            # Try a random swap as exploration
            if non_edge_set and current_edge_set:
                rem_u, rem_v = random.choice(list(current_edge_set))
                add_u, add_v = random.choice(list(non_edge_set))

                rem_edge = (min(rem_u, rem_v), max(rem_u, rem_v))
                add_edge = (min(add_u, add_v), max(add_u, add_v))

                # Fast check: does it improve?
                # We can just do the swap and check score
                # But that's expensive.
                # Let's use the delta score function
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
