import time
import random
import math
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
    # This is a good starting point because it's close to the Turan graph T_2(16)
    # which has no triangles, but we need 1 more edge, creating triangles.
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

    # Simulated Annealing parameters
    T = 5.0
    cooling_rate = 0.9999
    min_T = 0.1
    
    max_iterations = 200000
    iterations = 0
    
    deg = [bin(mask).count('1') for mask in adj_masks]

    while time.time() - start_time < DEADLINE and iterations < max_iterations:
        iterations += 1

        # 1. Generate a candidate move: Remove one edge, Add one non-edge
        # Strategy: Prefer removing edges from the worst triangles to lower the score
        # If no worst triangles (score 0), just pick random edges.
        
        if current_max_tris:
            # Pick a random worst triangle
            i, j, k = random.choice(current_max_tris)
            tri_edges = [(i, j), (j, k), (i, k)]
            rem_u, rem_v = random.choice(tri_edges)
        else:
            # If score is 0, we are in a perfect state (unlikely with 65 edges on 16 nodes? 
            # Actually, max edges in triangle-free graph on 16 nodes is 64. So we MUST have triangles.)
            # So current_max_tris should never be empty if score > 0.
            rem_u, rem_v = random.choice(list(current_edge_set))

        # Normalize edge for set operations
        rem_edge = (min(rem_u, rem_v), max(rem_u, rem_v))
        
        # Pick a random non-edge to add
        # To speed up, we don't shuffle the whole set, just pick a random index from a list view?
        # No, random.choice(list(non_edge_set)) is O(N). 
        # We can maintain a list for non_edges and update it, but that's complex.
        # For n=16, N is small (120 total edges). list(non_edge_set) is fast enough.
        
        non_edges_list = list(non_edge_set)
        if not non_edges_list:
            break
        add_u, add_v = random.choice(non_edges_list)
        add_edge = (min(add_u, add_v), max(add_u, add_v))

        # Perform Swap
        # Remove rem_edge
        adj_masks[rem_u] ^= (1 << rem_v)
        adj_masks[rem_v] ^= (1 << rem_u)
        deg[rem_u] -= 1
        deg[rem_v] -= 1
        
        current_edge_set.discard(rem_edge)
        non_edge_set.add(rem_edge)

        # Add add_edge
        adj_masks[add_u] ^= (1 << add_v)
        adj_masks[add_v] ^= (1 << add_u)
        deg[add_u] += 1
        deg[add_v] += 1

        non_edge_set.discard(add_edge)
        current_edge_set.add(add_edge)

        # Calculate New Score
        new_score, new_max_tris = _compute_score(n, adj_masks)
        delta = new_score - current_score

        # Acceptance Criterion
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_score = new_score
            current_max_tris = new_max_tris
            
            if current_score < best_score:
                best_score = current_score
                best_adj_masks = list(adj_masks)
                # Optional: Cool down faster when we find a new best?
                # T = max(min_T, T * 0.95) 
        else:
            # Revert
            # Remove add_edge
            adj_masks[add_u] ^= (1 << add_v)
            adj_masks[add_v] ^= (1 << add_u)
            deg[add_u] -= 1
            deg[add_v] -= 1
            non_edge_set.add(add_edge)
            current_edge_set.discard(add_edge)

            # Add rem_edge
            adj_masks[rem_u] ^= (1 << rem_v)
            adj_masks[rem_v] ^= (1 << rem_u)
            deg[rem_u] += 1
            deg[rem_v] += 1
            current_edge_set.add(rem_edge)
            non_edge_set.discard(rem_edge)

        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T

    best_edges = _construct_edges_from_masks(best_adj_masks, n)
    return {'n': n, 'edges': best_edges}
