import time
import random
import sys
import numpy as np
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin before the 120s limit

    n = 16
    num_edges_total = n * (n - 1) // 2  # 120

    # Map edge (u, v) to an index 0..119
    def edge_index(u, v):
        if u > v:
            u, v = v, u
        # u < v
        idx = 0
        for i in range(u):
            idx += (n - 1 - i)
        idx += (v - u - 1)
        return idx

    # Precompute edge indices for all pairs
    edge_idx_map = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx_map[(u, v)] = edge_index(u, v)

    # Inverse map: index -> (u, v)
    inv_edge_map = [None] * num_edges_total
    for (u, v), idx in edge_idx_map.items():
        inv_edge_map[idx] = (u, v)

    # Precompute local Q3 masks on 8 vertices (0..7)
    # Q3 edges in abstract space (using standard labeling 000..111)
    # 0:000, 1:001, 2:010, 3:011, 4:100, 5:101, 6:110, 7:111
    q3_abstract_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    # Local edge index mapping for K8
    def local_edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (7 - i)
        idx += (v - u - 1)
        return idx

    local_q3_masks = set()

    # Generate all unique local Q3 edge sets by permuting vertices
    for p in permutations(range(8)):
        edges_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            le = local_edge_index(u_val, v_val)
            edges_set.add(le)

        if len(edges_set) == 12:
            local_q3_masks.add(frozenset(edges_set))

    # Convert to list of bitmasks (integers) for fast operations
    local_q3_bitmasks = []
    for mask_set in local_q3_masks:
        mask_int = 0
        for e in mask_set:
            mask_int |= (1 << e)
        local_q3_bitmasks.append(mask_int)

    # Generate global Q3 masks
    # For each subset of 8 vertices from 0..15
    global_q3_masks = []
    all_subsets = list(combinations(range(n), 8))

    for subset in all_subsets:
        # Map local index 0..7 to global vertex
        l2g_edge = [0] * 28
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_index(u_loc, v_loc)
                u_glob = subset[u_loc]
                v_glob = subset[v_loc]
                ge = edge_idx_map[(min(u_glob, v_glob), max(u_glob, v_glob))]
                l2g_edge[le] = ge

        for local_mask in local_q3_bitmasks:
            global_mask = 0
            # Iterate through bits in local_mask
            m = local_mask
            while m:
                lsb = m & -m
                le = lsb.bit_length() - 1
                ge = l2g_edge[le]
                global_mask |= (1 << ge)
                m ^= lsb
            global_q3_masks.append(global_mask)

    num_q3s = len(global_q3_masks)

    # Greedy Set Cover on the complement
    # We want to remove edges to hit all Q3s.
    # active_q3[i] = True if Q3 i is not yet hit.
    # count[e] = number of active Q3s containing edge e.

    active = [True] * num_q3s
    counts = [0] * num_edges_total

    # Build lists: q3s_containing_edge[e] = list of indices of Q3s containing e
    q3s_containing_edge = [[] for _ in range(num_edges_total)]

    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb

    # Initialize counts
    for e in range(num_edges_total):
        counts[e] = len(q3s_containing_edge[e])

    removed_edges = set()
    current_edge_mask = (1 << num_edges_total) - 1  # All edges present initially

    # Greedy Loop
    # We need to efficiently find the edge with max count.
    # Since num_edges_total is 120, we can just scan.

    while True:
        if time.time() > deadline:
            break

        # Find edge with max count
        max_count = -1
        best_edge = -1
        for e in range(num_edges_total):
            if e in removed_edges:
                continue
            if counts[e] > max_count:
                max_count = counts[e]
                best_edge = e

        if best_edge == -1 or max_count == 0:
            break

        # Remove best_edge
        removed_edges.add(best_edge)
        current_edge_mask &= ~(1 << best_edge)

        # Update active Q3s and counts
        for q3_idx in q3s_containing_edge[best_edge]:
            if active[q3_idx]:
                active[q3_idx] = False
                # This Q3 is now hit.
                # We need to decrement counts for all other edges in this Q3.
                mask = global_q3_masks[q3_idx]
                m = mask
                while m:
                    lsb = m & -m
                    e_idx = lsb.bit_length() - 1
                    if e_idx != best_edge:
                        counts[e_idx] -= 1
                    m ^= lsb

    # The remaining edges are the solution
    result_edges = []
    for e in range(num_edges_total):
        if e not in removed_edges:
            u, v = inv_edge_map[e]
            result_edges.append([u, v])

    # --- Refinement Phase (Local Search) ---
    # Try to add back removed edges if possible.
    # This is a "repair" step to improve the greedy solution.
    # We iterate through removed edges and try to re-add them if they don't create a Q3.
    
    # We need a fast check for "does adding edge e create a Q3?"
    # This is equivalent to checking if there is a Q3 containing edge e in the current graph.
    # We can precompute for each edge e, the list of Q3s containing it.
    # Then check if any of those Q3s are "active" (i.e., all their edges are present).
    
    # However, maintaining the "active" state dynamically during re-addition is complex.
    # Instead, we can use a simple heuristic:
    # For each removed edge, check if it can be added without violating the constraint.
    # We can do this by checking all Q3s containing that edge.
    # A Q3 is violated if ALL its edges are present in the current graph.
    
    # Let's maintain the current edge set as a boolean array or set.
    current_edges_set = set()
    for e in range(num_edges_total):
        if e not in removed_edges:
            current_edges_set.add(e)
            
    # For faster lookup, use a bitset or boolean array
    edge_present = [False] * num_edges_total
    for e in current_edges_set:
        edge_present[e] = True

    # We will try to re-add edges in order of their original "desirability" (maybe degree or count)
    # But simple random or sorted order might work.
    # Let's sort removed edges by the number of Q3s they were involved in (higher is more "dangerous" but also maybe more "useful" if they can be added back safely? No, safer to try adding back edges that were removed early or have low impact).
    # Actually, let's just iterate through all removed edges and try to add them back if possible.
    
    removed_list = list(removed_edges)
    # Shuffle to avoid bias
    random.shuffle(removed_list)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for e in removed_list:
            if time.time() > deadline:
                break
            if e in current_edges_set:
                continue
            
            # Check if adding e creates a Q3
            # A Q3 is created if there exists a Q3 containing e such that all other 11 edges of that Q3 are already present.
            safe = True
            for q3_idx in q3s_containing_edge[e]:
                # Check if all edges in this Q3 are present (except e, which we are about to add)
                mask = global_q3_masks[q3_idx]
                # Remove e from mask
                mask_without_e = mask & ~(1 << e)
                
                # Check if all bits in mask_without_e are set in the current edge_present state
                # We can do this by iterating through bits or using bit operations
                # Since we don't have a global bitmask for the whole graph (120 bits is too big for simple int ops in Python without care),
                # we can just check the 11 edges.
                
                m = mask_without_e
                all_present = True
                while m:
                    lsb = m & -m
                    e_idx = lsb.bit_length() - 1
                    if not edge_present[e_idx]:
                        all_present = False
                        break
                    m ^= lsb
                
                if all_present:
                    safe = False
                    break
            
            if safe:
                # Add the edge
                current_edges_set.add(e)
                edge_present[e] = True
                improved = True
                # Remove from removed_list to avoid re-checking? Or just let it be skipped.
                # It's fine to leave it, the `if e in current_edges_set: continue` will skip it next time.

    # Construct final result
    result_edges = []
    for e in sorted(current_edges_set):
        u, v = inv_edge_map[e]
        result_edges.append([u, v])

    return result_edges

if __name__ == "__main__":
    res = construct()
    print(len(res))
