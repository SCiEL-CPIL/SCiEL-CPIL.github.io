import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    total_edges = 120
    edge_index = [[-1]*n for _ in range(n)]
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u+1, n):
            edge_index[u][v] = idx
            edge_index[v][u] = idx
            edge_list.append((u, v))
            idx += 1
            
    # 1. Precompute all Q3 subgraphs in K16
    # A Q3 is defined by 8 vertices and a specific pairing/structure.
    # In K8, there are 168 Q3 subgraphs.
    # We generate them by picking 4 vertices for one part (A) and 4 for the other (B).
    # Then we need to map them such that it forms a cube.
    # Actually, simpler: 
    # Iterate all 8-subsets.
    # For each 8-subset, generate all Q3s inside it.
    
    # Precompute Q3 edge masks for a generic K8 (vertices 0..7)
    q3_masks_k8 = []
    # A Q3 is a bipartite graph (4,4) with degree 3.
    # Vertices 0,1,2,3 in A, 4,5,6,7 in B.
    # Edges: (0,4), (0,5), (0,6) -> 0 is not connected to 7.
    # This is not a cube. A cube is a 3-regular bipartite graph.
    # There are 168 such graphs.
    # We can generate them by taking all 3-regular bipartite graphs on (4,4).
    # Or simpler: 
    # Fix the structure of Q3.
    # Q3 vertices: 0..7. Edges:
    # 0-1, 0-2, 0-4
    # 1-3, 1-5
    # 2-6, 2-3
    # 3-7
    # 4-5, 4-6
    # 5-7
    # 6-7
    # Wait, standard Q3 edges:
    # 000-001 (0-1), 000-010 (0-2), 000-100 (0-4)
    # 001-011 (1-3), 001-101 (1-5)
    # 010-011 (2-3), 010-110 (2-6)
    # 011-111 (3-7)
    # 100-101 (4-5), 100-110 (4-6)
    # 101-111 (5-7)
    # 110-111 (6-7)
    # Total 12 edges.
    
    # Let's generate all isomorphic copies in K8.
    # We can just permute the 8 vertices of the standard Q3.
    # There are 8! = 40320 permutations.
    # Many will produce the same edge set.
    # Standard Q3 has automorphisms of size 48.
    # 40320 / 48 = 840 distinct edge sets?
    # No, 168 distinct Q3s in K8.
    # 40320 / 240?
    # Let's just generate them by iterating permutations and adding to a set.
    
    standard_q3_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    def get_q3_mask_for_perm(perm):
        # perm is a tuple of 8 vertices
        mask = 0
        # local edge index mapping for 0..7
        # We need to map local (u,v) to local edge index 0..27
        # Then map to global? No, this is for K8 local.
        # Let's just create a set of local edges (u,v) with u<v
        edges = set()
        for u, v in standard_q3_edges:
            pu = perm[u]
            pv = perm[v]
            if pu > pv: pu, pv = pv, pu
            edges.add((pu, pv))
        # Convert to bitmask of 28 edges
        # local_edge_idx[i][j]
        for i in range(8):
            for j in range(i+1, 8):
                if (i, j) in edges:
                    # Calculate bit index
                    # 0..7
                    # idx = i*7 - (i*(i+1))/2 + (j-i-1) ?
                    # Simpler: precompute local_edge_idx
                    pass
        return mask # Placeholder

    # Better: Precompute local edge indices for K8
    local_edge_idx = [[-1]*8 for _ in range(8)]
    l_idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_idx[i][j] = l_idx
            local_edge_idx[j][i] = l_idx
            l_idx += 1
            
    q3_masks_k8_set = set()
    for perm in itertools.permutations(range(8)):
        mask = 0
        for u, v in standard_q3_edges:
            pu = perm[u]
            pv = perm[v]
            e_idx = local_edge_idx[pu][pv]
            mask |= (1 << e_idx)
        q3_masks_k8_set.add(mask)
    q3_masks_k8 = list(q3_masks_k8_set)
    
    # Now map these to K16
    # For each 8-subset of 16, map local 0..7 to global vertices.
    all_q3_masks = []
    seen_masks = set()
    
    # Precompute mapping from local edge (i,j) in subset to global edge index
    # We will do this on the fly or precompute for all subsets?
    # 12870 subsets.
    # For each subset, 168 masks.
    # Total 2.16M masks.
    
    # To speed up, we can just store the global edge bitmask for each Q3.
    # We also need `q3s_with_edge[e]`.
    
    # Let's generate all Q3 masks for K16.
    # This might take a few seconds.
    for subset in itertools.combinations(range(n), 8):
        # Map local 0..7 to global vertices in subset
        # local i -> subset[i]
        for mask in q3_masks_k8:
            global_mask = 0
            valid = True
            # Iterate all 28 local edges
            for i in range(8):
                for j in range(i+1, 8):
                    if (mask >> local_edge_idx[i][j]) & 1:
                        u_glob = subset[i]
                        v_glob = subset[j]
                        g_idx = edge_index[u_glob][v_glob]
                        global_mask |= (1 << g_idx)
            if global_mask not in seen_masks:
                seen_masks.add(global_mask)
                all_q3_masks.append(global_mask)
                
    # Precompute q3s_with_edge
    q3s_with_edge = [[] for _ in range(total_edges)]
    for q3_mask in all_q3_masks:
        # Find edges in this mask
        m = q3_mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_with_edge[e_idx].append(q3_mask)
            m ^= lsb
            
    # Helper to check if graph G (bitmask) contains any Q3
    def contains_q3(G):
        for q3 in all_q3_masks:
            if (G & q3) == q3:
                return True
        return False
        
    # Helper to check if adding edge e to G creates a Q3
    def creates_q3(G, e_idx):
        new_G = G | (1 << e_idx)
        for q3 in q3s_with_edge[e_idx]:
            if (new_G & q3) == q3:
                return True
        return False

    best_G = 0
    best_count = 0
    
    # Strategy:
    # 1. Start with K16
    # 2. Remove edges to make it Q3-free
    # 3. Add edges back greedily
    # 4. Local search
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        G = (1 << total_edges) - 1
        removed_edges = []
        
        # Phase 1: Remove edges to eliminate Q3s
        # We need to find a Q3 in G.
        # To speed up, we can sample from all_q3_masks.
        # Or iterate.
        # Since G is dense, a Q3 is likely found quickly.
        
        while True:
            if time.time() > deadline:
                break
            # Find a Q3
            found = False
            # Random sample to avoid bias?
            # Or just iterate from random start?
            start_idx = random.randint(0, len(all_q3_masks) - 1)
            for i in range(len(all_q3_masks)):
                idx = (start_idx + i) % len(all_q3_masks)
                q3 = all_q3_masks[idx]
                if (G & q3) == q3:
                    # Found a Q3
                    # Pick an edge to remove
                    # Pick edge with highest degree in Q3? All are 3.
                    # Pick edge that is in most Q3s?
                    # Let's just pick a random edge from the Q3
                    m = q3
                    edges_in_q3 = []
                    while m:
                        lsb = m & -m
                        e_idx = lsb.bit_length() - 1
                        edges_in_q3.append(e_idx)
                        m ^= lsb
                    e_to_remove = random.choice(edges_in_q3)
                    G &= ~(1 << e_to_remove)
                    found = True
                    break
            if not found:
                break
                
        # Phase 2: Add edges back
        # Try to add every missing edge
        for e_idx in range(total_edges):
            if time.time() > deadline:
                break
            if not (G & (1 << e_idx)):
                if not creates_q3(G, e_idx):
                    G |= (1 << e_idx)
                    
        # Phase 3: Local Search (Swaps)
        # Try to improve by swapping
        # We can also try to add edges in random order
        missing_edges = [e for e in range(total_edges) if not (G & (1 << e))]
        present_edges = [e for e in range(total_edges) if (G & (1 << e))]
        
        while time.time() < deadline - 5:
            progress = False
            # Try adding missing edges again (order might matter)
            random.shuffle(missing_edges)
            for e_idx in missing_edges:
                if not (G & (1 << e_idx)):
                    if not creates_q3(G, e_idx):
                        G |= (1 << e_idx)
                        progress = True
                        missing_edges.remove(e_idx)
            
            if not progress:
                # Try swaps
                if missing_edges and present_edges:
                    e_add = random.choice(missing_edges)
                    e_rem = random.choice(present_edges)
                    if e_add != e_rem:
                        new_G = (G & ~(1 << e_rem)) | (1 << e_add)
                        # Check if new_G is Q3-free
                        # We only need to check Q3s containing e_add
                        # AND Q3s containing e_rem (which might be broken, so no new Q3 formed by removing e_rem)
                        # So only check Q3s containing e_add.
                        if not creates_q3(new_G, e_add):
                            G = new_G
                            present_edges.remove(e_rem)
                            missing_edges.remove(e_add)
                            missing_edges.append(e_rem)
                            progress = True
            if not progress:
                break
                
        count = G.bit_count()
        if count > best_count:
            best_count = count
            best_G = G
            
    result = []
    for e_idx in range(total_edges):
        if (best_G >> e_idx) & 1:
            u, v = edge_list[e_idx]
            result.append([u, v])
            
    return result
