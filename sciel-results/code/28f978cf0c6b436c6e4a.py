import time
import sys
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n = 16
    num_edges_total = n * (n - 1) // 2  # 120

    # 1. Edge Indexing
    edge_idx_map = {}
    inv_edge_map = [None] * num_edges_total
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx_map[(u, v)] = idx
            inv_edge_map[idx] = (u, v)
            idx += 1

    # 2. Precompute Local Q3 Masks (on 0..7)
    q3_abstract_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    def local_edge_index(u, v):
        if u > v: u, v = v, u
        count = 0
        for i in range(u):
            count += (7 - i)
        count += (v - u - 1)
        return count

    local_q3_masks = set()
    for p in permutations(range(8)):
        mask = 0
        for u, v in q3_abstract_edges:
            lu = p[u]
            lv = p[v]
            le = local_edge_index(lu, lv)
            mask |= (1 << le)
        if mask.bit_count() == 12:
            local_q3_masks.add(mask)

    local_q3_list = list(local_q3_masks)

    # 3. Precompute Global Q3 Masks
    global_q3_masks = []
    all_subsets = list(combinations(range(n), 8))

    local_q3_edge_indices = []
    for m in local_q3_list:
        edges = []
        for i in range(28):
            if m & (1 << i):
                edges.append(i)
        local_q3_edge_indices.append(edges)

    for subset in all_subsets:
        l2g = [0] * 28
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_index(u_loc, v_loc)
                u_glob = subset[u_loc]
                v_glob = subset[v_loc]
                ge = edge_idx_map[(min(u_glob, v_glob), max(u_glob, v_glob))]
                l2g[le] = ge

        for edge_list in local_q3_edge_indices:
            g_mask = 0
            for le in edge_list:
                ge = l2g[le]
                g_mask |= (1 << ge)
            global_q3_masks.append(g_mask)

    num_q3s = len(global_q3_masks)

    if time.time() > deadline:
        return [[i, i + 1] for i in range(0, 16, 2)]

    # 4. Greedy Hitting Set
    active = [True] * num_q3s
    counts = [0] * num_edges_total
    q3s_containing_edge = [[] for _ in range(num_edges_total)]

    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb

    for e in range(num_edges_total):
        counts[e] = len(q3s_containing_edge[e])

    removed_edges = [False] * num_edges_total

    while True:
        if time.time() > deadline:
            break

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

        removed_edges[best_edge] = True

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

    # 5. Local Search: Add back edges
    q3_edge_lists = []
    for mask in global_q3_masks:
        edges = []
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            edges.append(e_idx)
            m ^= lsb
        q3_edge_lists.append(edges)

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for e in range(num_edges_total):
            if not removed_edges[e]:
                continue
            safe = True
            for q3_idx in q3s_containing_edge[e]:
                edges_in_q = q3_edge_lists[q3_idx]
                for other_e in edges_in_q:
                    if other_e == e:
                        continue
                    if removed_edges[other_e]:
                        safe = True
                        break
                else:
                    safe = False
                    break
            if safe:
                removed_edges[e] = False
                changed = True

    # 6. Simulated Annealing / Random Restart Local Search
    # We will try to improve the solution by randomly removing and re-adding edges.
    # Current state: removed_edges is True for edges in R.
    # We want to minimize |R| (maximize kept edges).
    
    def get_current_removed():
        return [e for e in range(num_edges_total) if removed_edges[e]]
    
    def is_valid_state(removals):
        # Check if removing `removals` hits all Q3s
        # This is expensive if done naively.
        # Instead, we maintain the state and check validity incrementally or via full check if needed.
        # For now, we rely on the local search to maintain validity.
        # If we do random moves, we need a fast validity check.
        # A move: Remove edge e from R (add to G). Check if any Q3 is formed.
        # If valid, accept. If invalid, reject (or accept with prob exp(-delta/T)).
        
        pass

    best_removed = get_current_removed()
    best_size = len(best_removed)
    
    # We will perform a local search that tries to swap edges in R with edges not in R.
    # Or simply try to remove edges from R one by one (already done in step 5).
    # To escape local minima, we can try adding an edge to R (making graph sparser) 
    # and then removing multiple edges from R.
    
    # Let's try a different approach: Randomized Greedy
    # Run the greedy algorithm multiple times with random tie-breaking and different orderings.
    
    # But the current solution is already quite good.
    # Let's try to improve by checking if we can swap a removed edge with a kept edge? No, that doesn't make sense.
    # We want to keep more edges.
    
    # Let's try to perturb the current solution:
    # Pick a random edge e in R. Remove it from R (add to G).
    # This might create Q3s.
    # Then, try to remove a set of edges from G to hit those Q3s, preferring edges that are already in R or "cheap" edges.
    
    # This is complex. Let's stick to the current robust solution but add a simple random restart if time permits.
    
    # Since we have time, let's try a few random restarts of the greedy process with different edge orderings.
    
    def greedy_hitting_set(order):
        active = [True] * num_q3s
        counts = [0] * num_edges_total
        rem = [False] * num_edges_total
        
        # Initialize counts
        for e in range(num_edges_total):
            counts[e] = len(q3s_containing_edge[e])
            
        # We need to process edges in the given order.
        # But the greedy choice depends on current counts.
        # Standard greedy: pick max count.
        # Randomized greedy: pick from top k counts?
        
        # Let's just shuffle the edge list for tie-breaking or initial priority?
        # Actually, the "marginal utility" idea suggests we should look at the structure.
        
        # For now, let's just return the current best if no time.
        pass

    # If we have time, try to optimize further with a simple local search that allows temporary invalidity (Simulated Annealing).
    # But implementing SA correctly with fast validity checks is tricky.
    
    # Let's just ensure the local search in step 5 is run multiple times with different starting orders?
    # No, step 5 is deterministic.
    
    # Let's try to randomize the "add back" order in step 5.
    # Re-run step 5 with random order of edges in R.
    
    # Save current best
    current_removed = get_current_removed()
    
    # Try 10 random permutations of the removed edges for the "add back" phase
    for _ in range(10):
        if time.time() > deadline:
            break
            
        # Restore current_removed state
        for e in range(num_edges_total):
            removed_edges[e] = (e in current_removed)
            
        # Shuffle the list of removed edges
        r_list = get_current_removed()
        random.shuffle(r_list)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for e in r_list:
                if not removed_edges[e]:
                    continue
                safe = True
                for q3_idx in q3s_containing_edge[e]:
                    edges_in_q = q3_edge_lists[q3_idx]
                    for other_e in edges_in_q:
                        if other_e == e:
                            continue
                        if removed_edges[other_e]:
                            safe = True
                            break
                    else:
                        safe = False
                        break
                if safe:
                    removed_edges[e] = False
                    changed = True
                    
        # Check if we improved
        new_removed = get_current_removed()
        if len(new_removed) < len(best_removed):
            best_removed = new_removed
            best_size = len(best_removed)
            current_removed = new_removed

    # Construct result from best_removed
    result_edges = []
    for e in range(num_edges_total):
        if e not in best_removed:
            u, v = inv_edge_map[e]
            result_edges.append([u, v])
            
    return result_edges

