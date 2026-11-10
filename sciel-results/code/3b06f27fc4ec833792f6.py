import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Precomputation
    # Vertices 0..15
    n = 16
    
    # Generate all 8-subsets
    subsets_8 = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets_8)
    
    # Map global edges to indices 0..119
    # Edge (u, v) with u < v
    # Index = u * 15 - u*(u+1)//2 + (v - u - 1) ? 
    # Simpler: just use a dictionary or a 16x16 matrix of indices.
    edge_index = [[-1]*n for _ in range(n)]
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u+1, n):
            edge_index[u][v] = idx
            edge_index[v][u] = idx
            edge_list.append((u, v))
            idx += 1
    total_edges = 120
    
    # For each 8-subset, we need:
    # 1. The 28 local edges mapped to global indices.
    # 2. The 840 Q3 masks (28-bit).
    # 3. For each local edge (0..27), the list of Q3 masks that contain it.
    
    # Precompute Q3 structure in K_8
    # Vertices 0..7
    # Partition into A (4) and B (4).
    # A is a 4-subset of 0..7. B is complement.
    # Perfect matching between A and B.
    # Q3 edges are all (a,b) for a in A, b in B, EXCEPT the matching edges.
    
    # Let's generate the 840 masks for a generic K_8
    # We need to be careful with vertex ordering.
    # The mask bits 0..27 correspond to local edges.
    # Let's define local edge index for (i, j) with i < j in 0..7.
    local_edge_idx = [[-1]*8 for _ in range(8)]
    l_idx = 0
    local_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_idx[i][j] = l_idx
            local_edge_idx[j][i] = l_idx
            local_edges.append((i, j))
            l_idx += 1
            
    q3_masks_for_k8 = []
    # Iterate over all 35 partitions (A, B)
    # To avoid duplicates, we can just iterate all 4-subsets A of 0..7.
    # There are 70. But (A, B) and (B, A) produce the same set of edges?
    # Yes, the edge set is symmetric.
    # So we can iterate 35 unique partitions.
    # Or just iterate all 70 and deduplicate masks?
    # Let's iterate all 4-subsets A of range(8).
    seen_masks = set()
    for A in itertools.combinations(range(8), 4):
        B = tuple(v for v in range(8) if v not in A)
        # For each perfect matching between A and B
        # Permutations of B assigned to A
        for perm in itertools.permutations(B):
            # perm[i] is the vertex in B matched to A[i]
            # Edges of Q3: (A[i], B[j]) for all i, j where perm[i] != B[j]
            # Wait, the matching removes (A[i], perm[i]).
            # So edges are (A[i], B[j]) for j != i? No.
            # Let's say matching pairs A[i] with M[i].
            # Then edges are (A[i], B[j]) for all i, j such that B[j] != M[i].
            
            mask = 0
            for i in range(4):
                a = A[i]
                m = perm[i]
                for j in range(4):
                    b = B[j]
                    if b != m:
                        # Edge (a, b) exists
                        # Get local edge index
                        e_idx = local_edge_idx[a][b]
                        mask |= (1 << e_idx)
            
            if mask not in seen_masks:
                seen_masks.add(mask)
                q3_masks_for_k8.append(mask)
                
    # Should be 840 masks
    # assert len(q3_masks_for_k8) == 840
    
    # For each global 8-subset, map local edges to global edges
    # And build the index: for each local edge, which Q3 masks contain it?
    
    subset_data = []
    for s in subsets_8:
        # s is a tuple of 8 global vertices
        # Map local 0..7 to global s[0]..s[7]
        # local_edge_global_indices: list of 28 global edge indices
        local_edge_global_indices = []
        for i in range(8):
            for j in range(i+1, 8):
                u_glob = s[i]
                v_glob = s[j]
                g_idx = edge_index[u_glob][v_glob]
                local_edge_global_indices.append(g_idx)
        
        # For each local edge (0..27), find Q3 masks that contain it
        # q3_by_local_edge: list of 28 lists
        q3_by_local_edge = [[] for _ in range(28)]
        for q_mask in q3_masks_for_k8:
            for l_e in range(28):
                if (q_mask >> l_e) & 1:
                    q3_by_local_edge[l_e].append(q_mask)
        
        subset_data.append({
            'verts': s,
            'local_to_global': local_edge_global_indices,
            'q3_by_local_edge': q3_by_local_edge
        })
    
    # Precompute for each global edge, which subsets contain it
    # edge_to_subsets: list of 120 lists of subset indices
    edge_to_subsets = [[] for _ in range(total_edges)]
    for i, data in enumerate(subset_data):
        verts = data['verts']
        # Find all pairs in verts
        for k in range(8):
            for m in range(k+1, 8):
                u = verts[k]
                v = verts[m]
                g_idx = edge_index[u][v]
                edge_to_subsets[g_idx].append(i)

    # Helper: Extract 28-bit mask of G on subset i
    # G is a 120-bit int
    def get_subset_mask(G, i):
        data = subset_data[i]
        l2g = data['local_to_global']
        mask = 0
        for l in range(28):
            g_idx = l2g[l]
            if (G >> g_idx) & 1:
                mask |= (1 << l)
        return mask

    # Helper: Check if adding edge e (global index) to G creates a Q3
    def is_q3_free_addition(G, e_idx):
        # G is 120-bit int without e_idx
        # We want to know if G | (1 << e_idx) contains a Q3
        # Any new Q3 must contain e_idx
        # Iterate over subsets containing e_idx
        for sub_idx in edge_to_subsets[e_idx]:
            data = subset_data[sub_idx]
            # Get current mask of G on this subset
            g_mask = get_subset_mask(G, sub_idx)
            # Find which local edge corresponds to e_idx
            # We can precompute this: for each subset and global edge, what is the local index?
            # Or just check if e_idx is in local_to_global
            # It is, because sub_idx is in edge_to_subsets[e_idx]
            # Find local index
            l2g = data['local_to_global']
            l_idx = l2g.index(e_idx) # This is O(28), slow. Precompute?
            
            # Precomputation for local index lookup:
            # In subset_data, we can store a dict or list mapping global edge to local index.
            # Let's do that in precomputation.
            
            # For now, let's assume we have it.
            # I will add this to precomputation below.
            
            # Get Q3 masks that contain this local edge
            q3_list = data['q3_by_local_edge'][l_idx]
            
            # Check if any q3_mask is subset of (g_mask | (1 << l_idx))
            # Note: g_mask does NOT have bit l_idx set (since e is not in G yet)
            # So new_mask = g_mask | (1 << l_idx)
            new_mask = g_mask | (1 << l_idx)
            
            for q_mask in q3_list:
                if (new_mask & q_mask) == q_mask:
                    return False # Q3 found
            
        return True

    # Update precomputation to include local index lookup
    for i, data in enumerate(subset_data):
        l2g = data['local_to_global']
        g2l = [-1]*total_edges
        for l, g in enumerate(l2g):
            g2l[g] = l
        data['global_to_local'] = g2l

    # Optimize is_q3_free_addition to use global_to_local
    def is_q3_free_addition_opt(G, e_idx):
        for sub_idx in edge_to_subsets[e_idx]:
            data = subset_data[sub_idx]
            g_mask = get_subset_mask(G, sub_idx)
            l_idx = data['global_to_local'][e_idx]
            q3_list = data['q3_by_local_edge'][l_idx]
            new_mask = g_mask | (1 << l_idx)
            
            for q_mask in q3_list:
                if (new_mask & q_mask) == q_mask:
                    return False
        return True

    # Local Search / Greedy Construction
    best_G = 0
    best_count = 0
    
    # We can try multiple random orders
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        G = 0
        edges = list(range(total_edges))
        random.shuffle(edges)
        
        for e_idx in edges:
            if time.time() > deadline:
                break
            if is_q3_free_addition_opt(G, e_idx):
                G |= (1 << e_idx)
        
        count = bin(G).count('1')
        if count > best_count:
            best_count = count
            best_G = G

    # Convert best_G to list of edges
    result = []
    for e_idx in range(total_edges):
        if (best_G >> e_idx) & 1:
            u, v = edge_list[e_idx]
            result.append([u, v])
            
    return result
