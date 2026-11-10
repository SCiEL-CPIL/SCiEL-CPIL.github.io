import time
import itertools
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    total_edges = n * (n - 1) // 2  # 120
    
    # 1. Precompute global edge indices
    edge_index = [[-1] * n for _ in range(n)]
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[u][v] = idx
            edge_index[v][u] = idx
            edge_list.append((u, v))
            idx += 1
            
    # 2. Precompute local edge indices for K8
    local_edge_idx = [[-1] * 8 for _ in range(8)]
    l_idx = 0
    for i in range(8):
        for j in range(i + 1, 8):
            local_edge_idx[i][j] = l_idx
            local_edge_idx[j][i] = l_idx
            l_idx += 1
            
    # 3. Generate all Q3 masks in K8
    # A Q3 is K_{4,4} minus a perfect matching.
    # Bipartitions of 8 vertices into two sets of 4: C(8,4)/2 = 35
    # Perfect matchings between them: 4! = 24
    # Total masks: 35 * 24 = 840
    seen_masks = set()
    q3_masks = []
    for A in itertools.combinations(range(8), 4):
        A_set = set(A)
        B = tuple(v for v in range(8) if v not in A_set)
        # Avoid duplicate bipartitions (A,B) and (B,A)
        # Since we iterate combinations, A is the first 4. 
        # However, the set of vertices in A is fixed. 
        # The mask depends on which edges are present.
        # The edges of Q3 are all edges between A and B EXCEPT the matching.
        for perm in itertools.permutations(B):
            # perm[i] is the vertex in B matched to A[i]
            # The matching edges are (A[i], perm[i])
            # The Q3 edges are (a, b) for a in A, b in B such that b != matched(a)
            mask = 0
            for i in range(4):
                a = A[i]
                matched_b = perm[i]
                for j in range(4):
                    b = B[j]
                    if b != matched_b:
                        e_idx = local_edge_idx[a][b]
                        mask |= (1 << e_idx)
            if mask not in seen_masks:
                seen_masks.add(mask)
                q3_masks.append(mask)
                
    # 4. Precompute subset data
    all_subsets = list(itertools.combinations(range(n), 8))
    num_subsets = len(all_subsets)
    
    # For each subset, map local edge index to global edge index
    # local_to_global[i] = global_edge_index
    subset_local_to_global = []
    for s in all_subsets:
        l2g = []
        for i in range(8):
            for j in range(i + 1, 8):
                u_glob, v_glob = s[i], s[j]
                g_idx = edge_index[u_glob][v_glob]
                l2g.append(g_idx)
        subset_local_to_global.append(l2g)
        
    # For each global edge, list of subset indices that contain it
    edge_to_subsets = [[] for _ in range(total_edges)]
    for i, l2g in enumerate(subset_local_to_global):
        for g_idx in l2g:
            edge_to_subsets[g_idx].append(i)
            
    # Precompute for each local edge index (0..27), which Q3 masks contain it
    # This helps in checking if adding an edge creates a Q3
    q3_masks_by_local_edge = [[] for _ in range(28)]
    for q_mask in q3_masks:
        for l_e in range(28):
            if (q_mask >> l_e) & 1:
                q3_masks_by_local_edge[l_e].append(q_mask)

    def get_subset_mask(G, sub_idx):
        """Get the 28-bit mask of G restricted to the subset sub_idx"""
        l2g = subset_local_to_global[sub_idx]
        mask = 0
        for l in range(28):
            g_idx = l2g[l]
            if (G >> g_idx) & 1:
                mask |= (1 << l)
        return mask

    def is_q3_free_addition(G, e_idx):
        """Check if adding global edge e_idx to G creates a Q3"""
        for sub_idx in edge_to_subsets[e_idx]:
            l2g = subset_local_to_global[sub_idx]
            # Find local index of e_idx in this subset
            # Since l2g is a list of 28 global indices, we need to find e_idx
            # Optimization: Precompute global_to_local for each subset? 
            # Or just check if e_idx is in l2g. 
            # Since we know e_idx is in this subset (from edge_to_subsets), 
            # we can find its position.
            # For speed, we can precompute a dictionary or array for each subset.
            # Let's precompute global_to_local for all subsets now.
            pass # We'll do this in the main loop setup or precompute outside
            
        # Re-implementing with precomputed global_to_local
        return True # Placeholder, will be replaced

    # Precompute global_to_local for each subset
    # subset_global_to_local[i][g_idx] = l_idx or -1
    # This is 12870 * 120 integers. ~1.5M ints. Memory is fine.
    subset_global_to_local = []
    for i, l2g in enumerate(subset_local_to_global):
        g2l = [-1] * total_edges
        for l, g in enumerate(l2g):
            g2l[g] = l
        subset_global_to_local.append(g2l)

    def is_q3_free_addition_optimized(G, e_idx):
        for sub_idx in edge_to_subsets[e_idx]:
            g2l = subset_global_to_local[sub_idx]
            l_idx = g2l[e_idx]
            if l_idx == -1:
                continue
            
            # Get current mask
            l2g = subset_local_to_global[sub_idx]
            mask = 0
            for l in range(28):
                g = l2g[l]
                if (G >> g) & 1:
                    mask |= (1 << l)
            
            new_mask = mask | (1 << l_idx)
            
            # Check if any Q3 mask is a subset of new_mask
            # Only check Q3 masks that contain l_idx
            for q_mask in q3_masks_by_local_edge[l_idx]:
                if (new_mask & q_mask) == q_mask:
                    return False
        return True

    best_edge_count = 0
    best_G = 0
    
    # Multiple restarts
    while time.time() < deadline - 5.0:
        random.seed(int(time.time() * 1000) % 100000) # Change seed each restart
        
        G = (1 << total_edges) - 1
        edge_count = total_edges
        
        # Initialize bad subsets
        # All subsets are potentially bad initially
        bad_subsets = set(range(num_subsets))
        
        # Main loop: remove edges to eliminate Q3s
        while bad_subsets and time.time() < deadline - 10.0:
            # Pick a random bad subset
            sub_idx = random.choice(list(bad_subsets))
            
            # Get current mask
            mask = get_subset_mask(G, sub_idx)
            
            # Check if still bad
            still_bad = False
            violated_pattern = None
            for q_mask in q3_masks:
                if (mask & q_mask) == q_mask:
                    still_bad = True
                    violated_pattern = q_mask
                    break
            
            if not still_bad:
                bad_subsets.discard(sub_idx)
                continue
            
            # Find edges in the violated pattern that are in G
            pattern_edges_local = []
            for l in range(28):
                if (violated_pattern >> l) & 1:
                    g_idx = subset_local_to_global[sub_idx][l]
                    if (G >> g_idx) & 1:
                        pattern_edges_local.append(g_idx)
            
            if not pattern_edges_local:
                bad_subsets.discard(sub_idx)
                continue
            
            # Remove a random edge from the pattern
            e_to_remove = random.choice(pattern_edges_local)
            G &= ~(1 << e_to_remove)
            edge_count -= 1
            
            # Update bad_subsets
            # Only subsets containing e_to_remove might have changed status
            # However, removing an edge can only turn a "bad" subset into "good", 
            # never "good" into "bad". So we only need to re-check subsets 
            # that were bad and contain the removed edge.
            # If a subset was good, it remains good.
            
            affected = set(edge_to_subsets[e_to_remove])
            
            new_bad = set()
            for sidx in bad_subsets:
                if sidx in affected:
                    # Recheck
                    m = get_subset_mask(G, sidx)
                    is_bad = False
                    for q_mask in q3_masks:
                        if (m & q_mask) == q_mask:
                            is_bad = True
                            break
                    if is_bad:
                        new_bad.add(sidx)
                else:
                    # Not affected, so still bad
                    new_bad.add(sidx)
            
            bad_subsets = new_bad
            
        # Refinement: try to add back edges
        if time.time() < deadline - 5.0:
            # Iterate through all edges not in G
            # Order: random or sorted? Random might find different optima.
            missing_edges = [i for i in range(total_edges) if not ((G >> i) & 1)]
            random.shuffle(missing_edges)
            
            for e_idx in missing_edges:
                if time.time() >= deadline - 5.0:
                    break
                if is_q3_free_addition_optimized(G, e_idx):
                    G |= (1 << e_idx)
                    edge_count += 1
        
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_G = G

    # Convert best_G to edge list
    result = []
    for u, v in edge_list:
        e_idx = edge_index[u][v]
        if (best_G >> e_idx) & 1:
            result.append([u, v])
            
    return result

