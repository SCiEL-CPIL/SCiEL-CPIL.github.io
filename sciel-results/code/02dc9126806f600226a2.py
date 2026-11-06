import time
import random

DEADLINE = 110  # Safety margin for 150s limit

def _compute_score(n, adj_masks):
    """
    Computes the maximum degree sum over all triangles in the graph.
    n: number of vertices
    adj_masks: list of int bitmasks representing adjacency
    """
    deg = [bin(mask).count('1') for mask in adj_masks]
    max_sum = 0

    # Iterate over all edges (i, j) with i < j
    for i in range(n):
        mi = adj_masks[i]
        # Only consider j > i
        mask_j_greater = mi & ~((1 << (i + 1)) - 1)

        while mask_j_greater:
            lsb = mask_j_greater & -mask_j_greater
            j = lsb.bit_length() - 1
            mask_j_greater ^= lsb

            # Common neighbors of i and j
            common = mi & adj_masks[j]

            # Iterate k in common
            max_deg_k = -1
            mask_k = common
            while mask_k:
                lsb_k = mask_k & -mask_k
                k = lsb_k.bit_length() - 1
                d = deg[k]
                if d > max_deg_k:
                    max_deg_k = d
                mask_k ^= lsb_k

            if max_deg_k >= 0:
                s = deg[i] + deg[j] + max_deg_k
                if s > max_sum:
                    max_sum = s

    return max_sum


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
    # Check validity
    if (adj_masks[add_u] >> add_v) & 1:
        return None
    if (adj_masks[rem_u] >> rem_v) & 1 == 0:
        return None

    # New degrees
    nd = list(deg)
    nd[rem_u] -= 1
    nd[rem_v] -= 1
    nd[add_u] += 1
    nd[add_v] += 1

    # New adjacency masks (copy)
    nm = list(adj_masks)
    nm[rem_u] &= ~(1 << rem_v)
    nm[rem_v] &= ~(1 << rem_u)
    nm[add_u] |= (1 << add_v)
    nm[add_v] |= (1 << add_u)

    # Compute new max triangle degree sum
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

    current_score = _compute_score(n, adj_masks)
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

    max_iterations = 200000
    iterations = 0
    stuck_count = 0
    deg = [bin(mask).count('1') for mask in adj_masks]
    
    # Convert non_edge_set to list for faster random access? 
    # random.choice on set is O(N) to convert to list? 
    # Actually, random.choice on a set converts to list internally in some versions or requires sequence.
    # Better to maintain a list of non-edges.
    non_edge_list = list(non_edge_set)
    current_edge_list = list(current_edge_set)

    while time.time() - start_time < DEADLINE and iterations < max_iterations:
        iterations += 1

        improved = False

        # Find worst triangle(s)
        worst_sum, worst_triangle = _find_worst_triangle(n, adj_masks, deg)

        if worst_triangle:
            i, j, k = worst_triangle
            # Degree-ordered: prioritize removing edges incident to highest-degree vertex
            tri_vertices = [(i, deg[i]), (j, deg[j]), (k, deg[k])]
            tri_vertices.sort(key=lambda x: -x[1])
            top_vertex = tri_vertices[0][0]

            # Edges incident to top_vertex first, then others
            edges_to_try = []
            other_edges = []
            for (a, b) in [(i, j), (j, k), (i, k)]:
                if a == top_vertex or b == top_vertex:
                    edges_to_try.append((a, b))
                else:
                    other_edges.append((a, b))
            edges_to_try.extend(other_edges)

            # For each removal candidate, try promising additions
            for rem_u, rem_v in edges_to_try:
                # Pick 5 random non-edges to try
                # This avoids sorting
                found_improvement = False
                # Shuffle a small subset? Or just pick random indices
                for _ in range(5):
                    if not non_edge_list:
                        break
                    idx = random.randint(0, len(non_edge_list) - 1)
                    add_u, add_v = non_edge_list[idx]
                    
                    add_edge = (min(add_u, add_v), max(add_u, add_v))
                    if add_edge in current_edge_set:
                        continue

                    # Fast delta evaluation
                    new_score = _delta_score_remove_add(n, adj_masks, deg, rem_u, rem_v, add_u, add_v)
                    if new_score is None:
                        continue

                    if new_score < current_score:
                        # Apply the swap
                        adj_masks[rem_u] ^= (1 << rem_v)
                        adj_masks[rem_v] ^= (1 << rem_u)
                        adj_masks[add_u] ^= (1 << add_v)
                        adj_masks[add_v] ^= (1 << add_u)

                        rem_edge = (min(rem_u, rem_v), max(rem_u, rem_v))
                        current_edge_set.discard(rem_edge)
                        non_edge_set.add(rem_edge)
                        non_edge_set.discard(add_edge)
                        current_edge_set.add(add_edge)
                        
                        # Update lists
                        # Find and remove rem_edge from current_edge_list
                        # This is O(N). 
                        # To avoid O(N), we can just rebuild lists occasionally or use a dict for index?
                        # For now, let's just update the set and rebuild list if it gets too messy?
                        # Actually, we can just append/removing from list is O(N).
                        # Let's try to keep lists in sync efficiently.
                        
                        # Remove rem_edge from current_edge_list
                        try:
                            ce_idx = current_edge_list.index(rem_edge)
                            current_edge_list.pop(ce_idx)
                        except ValueError:
                            pass
                            
                        # Add rem_edge to non_edge_list
                        non_edge_list.append(rem_edge)
                        
                        # Remove add_edge from non_edge_list
                        try:
                            ne_idx = non_edge_list.index(add_edge)
                            non_edge_list.pop(ne_idx)
                        except ValueError:
                            pass
                            
                        # Add add_edge to current_edge_list
                        current_edge_list.append(add_edge)

                        # Update degrees
                        deg[rem_u] -= 1
                        deg[rem_v] -= 1
                        deg[add_u] += 1
                        deg[add_v] += 1

                        current_score = new_score
                        improved = True
                        found_improvement = True
                        if new_score < best_score:
                            best_score = new_score
                            best_adj_masks = list(adj_masks)
                        break

                if found_improvement:
                    break

        if not improved:
            # Try a random swap as exploration
            if non_edge_list and current_edge_list:
                rem_u, rem_v = random.choice(current_edge_list)
                add_u, add_v = random.choice(non_edge_list)

                rem_edge = (min(rem_u, rem_v), max(rem_u, rem_v))
                add_edge = (min(add_u, add_v), max(add_u, add_v))

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
                    
                    # Update lists
                    try:
                        ce_idx = current_edge_list.index(rem_edge)
                        current_edge_list.pop(ce_idx)
                    except ValueError:
                        pass
                    non_edge_list.append(rem_edge)
                    
                    try:
                        ne_idx = non_edge_list.index(add_edge)
                        non_edge_list.pop(ne_idx)
                    except ValueError:
                        pass
                    current_edge_list.append(add_edge)

                    deg[rem_u] -= 1
                    deg[rem_v] -= 1
                    deg[add_u] += 1
                    deg[add_v] += 1

                    # Accept if better, or with small probability if equal (to explore)
                    if new_score < current_score or (new_score == current_score and random.random() < 0.1):
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
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
                        
                        # Revert lists
                        # This is getting messy. 
                        # Let's just rebuild lists from sets every 1000 iterations?
                        # Or just accept the O(N) cost of index/pop since N=120.
                        
                        try:
                            ne_idx = non_edge_list.index(rem_edge)
                            non_edge_list.pop(ne_idx)
                        except ValueError:
                            pass
                        current_edge_list.append(rem_edge)
                        
                        try:
                            ce_idx = current_edge_list.index(add_edge)
                            current_edge_list.pop(ce_idx)
                        except ValueError:
                            pass
                        non_edge_list.append(add_edge)

                        deg[rem_u] += 1
                        deg[rem_v] += 1
                        deg[add_u] -= 1
                        deg[add_v] -= 1
                        stuck_count += 1

                # Perturbation if stuck
                if stuck_count > 30:
                    for _ in range(3):
                        if non_edge_list and current_edge_list:
                            r_u, r_v = random.choice(current_edge_list)
                            a_u, a_v = random.choice(non_edge_list)

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
                            
                            # Update lists
                            try:
                                ce_idx = current_edge_list.index(r_edge)
                                current_edge_list.pop(ce_idx)
                            except ValueError:
                                pass
                            non_edge_list.append(r_edge)
                            
                            try:
                                ne_idx = non_edge_list.index(a_edge)
                                non_edge_list.pop(ne_idx)
                            except ValueError:
                                pass
                            current_edge_list.append(a_edge)

                            deg[r_u] -= 1
                            deg[r_v] -= 1
                            deg[a_u] += 1
                            deg[a_v] += 1

                    current_score = _compute_score(n, adj_masks)
                    stuck_count = 0

    best_edges = _construct_edges_from_masks(best_adj_masks, n)
    return {'n': n, 'edges': best_edges}
