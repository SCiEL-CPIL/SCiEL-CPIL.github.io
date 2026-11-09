import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n = 16
    num_edges_total = n * (n - 1) // 2  # 120

    # --- Precomputation ---

    # Map edge (u, v) to an index 0..119
    def edge_index(u, v):
        if u > v:
            u, v = v, u
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

    # Local edge index mapping for K8 (28 edges)
    def local_edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (7 - i)
        idx += (v - u - 1)
        return idx

    # Q3 abstract edges (vertices 0..7)
    q3_abstract_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    # Precompute all local Q3 edge sets (as frozensets of local edge indices)
    # There are 8! = 40320 permutations, but many map to the same edge set.
    local_q3_edge_sets = set()
    for p in itertools.permutations(range(8)):
        edges_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            le = local_edge_index(u_val, v_val)
            edges_set.add(le)
        if len(edges_set) == 12:
            local_q3_edge_sets.add(frozenset(edges_set))

    # Convert to list of bitmasks (28 bits)
    local_q3_bitmasks = []
    for mask_set in local_q3_edge_sets:
        mask_int = 0
        for e in mask_set:
            mask_int |= (1 << e)
        local_q3_bitmasks.append(mask_int)
        
    # Precompute local Q3 edge lists for faster global mask construction
    local_q3_edge_lists = []
    for mask in local_q3_bitmasks:
        edges = []
        for i in range(28):
            if mask & (1 << i):
                edges.append(i)
        local_q3_edge_lists.append(edges)

    # Precompute global Q3 masks
    # For each subset of 8 vertices, map local edges to global edges
    global_q3_masks = []
    all_subsets = list(itertools.combinations(range(n), 8))
    
    # Precompute local edge (u,v) -> local_idx for K8
    local_edge_to_idx = {}
    for u in range(8):
        for v in range(u + 1, 8):
            local_edge_to_idx[(u, v)] = local_edge_index(u, v)

    # For each subset, build the global masks
    for subset in all_subsets:
        # Map local edge idx to global edge idx
        l2g = [0] * 28
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_to_idx[(u_loc, v_loc)]
                u_glob = subset[u_loc]
                v_glob = subset[v_loc]
                ge = edge_idx_map[(min(u_glob, v_glob), max(u_glob, v_glob))]
                l2g[le] = ge
                
        # For each local Q3, generate the global mask
        for edge_list in local_q3_edge_lists:
            global_mask = 0
            for le in edge_list:
                ge = l2g[le]
                global_mask |= (1 << ge)
            global_q3_masks.append(global_mask)
            
    num_q3s = len(global_q3_masks)
    
    # Build lists: q3s_containing_edge[e] = list of indices of Q3s containing e
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb

    if time.time() > deadline:
        # Fallback: return a simple valid graph (e.g., matching)
        return [[i, i+1] for i in range(0, 16, 2)]

    # --- Greedy Set Cover / Edge Removal ---
    
    def run_greedy_removal(seed):
        random.seed(seed)
        
        active = [True] * num_q3s
        counts = [0] * num_edges_total
        
        for e in range(num_edges_total):
            counts[e] = len(q3s_containing_edge[e])
            
        removed_edges = [False] * num_edges_total
        
        # Greedy Loop
        while True:
            if time.time() > deadline:
                break
                
            # Find edge with max count
            max_count = -1
            best_edge = -1
            for e in range(num_edges_total):
                if removed_edges[e]:
                    continue
                if counts[e] > max_count:
                    max_count = counts[e]
                    best_edge = e
                    
            if best_edge == -1 or max_count == 0:
                break
                
            # Remove best_edge
            removed_edges[best_edge] = True
            
            # Update active Q3s and counts
            for q3_idx in q3s_containing_edge[best_edge]:
                if active[q3_idx]:
                    active[q3_idx] = False
                    mask = global_q3_masks[q3_idx]
                    m = mask
                    while m:
                        lsb = m & -m
                        e_idx = lsb.bit_length() - 1
                        if e_idx != best_edge:
                            counts[e_idx] -= 1
                        m ^= lsb

        # Local search polish: try to re-add removed edges if the graph stays Q3-free
        # Build the current edge set as a bitmask
        current_mask = 0
        for e in range(num_edges_total):
            if not removed_edges[e]:
                current_mask |= (1 << e)
        
        # For each removed edge, check if adding it back creates a Q3
        # A Q3 containing edge e is "hit" if at least one of its 12 edges is missing from current_mask
        # If no Q3 containing e is fully present in current_mask, we can add e back.
        
        # Precompute for each edge e, the list of Q3s containing it.
        # We already have q3s_containing_edge.
        
        # To speed up, we can maintain a count of "active" Q3s for each edge?
        # Or just check directly.
        # Number of removed edges is ~40.
        # For each removed edge, iterate through its Q3s (maybe ~100-200?) and check if all 12 edges are in current_mask.
        
        # Let's do a few passes of re-adding
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Shuffle removed edges to randomize order
            removed_list = [e for e in range(num_edges_total) if removed_edges[e]]
            random.shuffle(removed_list)
            
            for e in removed_list:
                if time.time() > deadline:
                    break
                
                # Check if e can be added
                safe = True
                for q3_idx in q3s_containing_edge[e]:
                    # Check if this Q3 is fully present in current_mask
                    mask = global_q3_masks[q3_idx]
                    if (mask & current_mask) == mask:
                        safe = False
                        break
                
                if safe:
                    # Add edge e back
                    removed_edges[e] = False
                    current_mask |= (1 << e)
                    changed = True

        return removed_edges

    best_edges = None
    best_count = -1

    # Run multiple restarts
    seed = 42
    while time.time() < deadline:
        current_removed = run_greedy_removal(seed)
        current_count = sum(1 for e in range(num_edges_total) if not current_removed[e])
        
        if current_count > best_count:
            best_count = current_count
            best_edges = current_removed
            # If we've found a very high edge count, we might want to break early if we can't improve?
            # Max possible is 120. Q3-free max is likely much lower.
            # If best_count > 80, we are doing well.
            if best_count >= 85:
                # Try a few more to see if we can push it, but don't waste too much time
                pass
        seed += 1

    if best_edges is None:
        # Fallback
        return [[i, i+1] for i in range(0, 16, 2)]

    # Construct result
    result = []
    for e in range(num_edges_total):
        if not best_edges[e]:
            u, v = inv_edge_map[e]
            result.append([u, v])
            
    return result

# Test locally
if __name__ == "__main__":
    import sys
    edges = construct()
    print(f"Found {len(edges)} edges")
    # Verify no Q3
    # This is a separate verification step, not included in the timed construct() for production
    # but good to have for debugging.
