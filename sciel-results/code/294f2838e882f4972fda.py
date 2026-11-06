import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 145.0

    def get_subsets(n, k):
        # Returns list of masks and list of vertex lists
        masks = []
        verts = []
        # Use itertools for generation, convert to mask
        import itertools
        for comb in itertools.combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            masks.append(mask)
            verts.append(comb)
        return masks, verts

    def solve_sa(n, k_list, time_budget):
        # Precompute subsets for k in k_list
        all_subsets = [] # list of (mask, verts)
        subset_indices_by_vertex = [set() for _ in range(n)]
        
        # We need to track degrees for each subset
        # degs[i] is a numpy array of shape (k,)
        
        # Generate subsets
        subset_data = [] # (mask, verts, idx)
        for k in k_list:
            masks, verts_list = get_subsets(n, k)
            for i, (m, v) in enumerate(zip(masks, verts_list)):
                idx = len(subset_data)
                subset_data.append((m, v, idx))
                for vertex in v:
                    subset_indices_by_vertex[vertex].add(idx)
        
        num_subsets = len(subset_data)
        
        # Initialize degrees
        # degs[i] is array of degrees for vertices in subset i
        degs = np.zeros((num_subsets, max(k_list)), dtype=np.int8)
        
        # We need a mapping from (subset_idx, position_in_subset) to vertex?
        # Or just store the vertex IDs in a parallel array
        subset_verts = np.array([d[1] for d in subset_data]) # shape (num_subsets, max_k) padded?
        # Better: store list of lists for simplicity, but numpy is faster for updates?
        # Let's use list of arrays.
        
        # Re-initialize degs to be a list of arrays
        degs_list = []
        for m, v, idx in subset_data:
            degs_list.append(np.zeros(len(v), dtype=np.int8))
            
        # Helper to check regularity
        def is_regular(degs_arr):
            return np.all(degs_arr == degs_arr[0])
        
        # Initialize graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Compute initial degrees
        for idx, (m, v, _) in enumerate(subset_data):
            d_arr = degs_list[idx]
            for pos, vertex in enumerate(v):
                # Count neighbors of vertex in subset m
                d_arr[pos] = bin(adj[vertex] & m).count('1')
        
        # Initial energy
        current_energy = 0
        for idx in range(num_subsets):
            if is_regular(degs_list[idx]):
                current_energy += 1
        
        best_energy = current_energy
        best_adj = adj[:]
        
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        
        # Precompute affected subsets for each edge?
        # No, just iterate vertices u, v and union their subset indices.
        
        steps = 0
        while time.time() - start_time < start_time + time_budget:
            if T < min_T:
                # Restart or stop?
                T = 1.0
                # Reheat
                
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Get affected subsets
            affected = subset_indices_by_vertex[u] | subset_indices_by_vertex[v]
            
            delta_e = 0
            old_regulars = []
            new_regulars = []
            
            # We need to update degrees for u and v in these subsets
            # And check regularity
            
            # Store old states to rollback if rejected
            changes = [] # (idx, pos_u, old_deg_u, pos_v, old_deg_v)
            
            for idx in affected:
                m, v_list, _ = subset_data[idx]
                # Find positions of u and v in v_list
                # Since v_list is small, linear search is fine
                pos_u = -1
                pos_v = -1
                for p, vert in enumerate(v_list):
                    if vert == u:
                        pos_u = p
                    elif vert == v:
                        pos_v = p
                
                if pos_u == -1 or pos_v == -1:
                    continue # Should not happen if u,v in subset
                
                d_arr = degs_list[idx]
                old_d_u = d_arr[pos_u]
                old_d_v = d_arr[pos_v]
                
                old_reg = is_regular(d_arr)
                
                # Apply flip
                # If edge (u,v) exists, remove it. Else add it.
                if (adj[u] >> v) & 1:
                    # Remove edge
                    d_arr[pos_u] -= 1
                    d_arr[pos_v] -= 1
                else:
                    # Add edge
                    d_arr[pos_u] += 1
                    d_arr[pos_v] += 1
                
                new_reg = is_regular(d_arr)
                
                if old_reg and not new_reg:
                    delta_e -= 1
                elif not old_reg and new_reg:
                    delta_e += 1
                
                changes.append((idx, pos_u, old_d_u, pos_v, old_d_v))
            
            # Decide
            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                # Accept
                # Update adj
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_adj = adj[:]
            else:
                # Reject: Rollback degrees
                for idx, pos_u, old_d_u, pos_v, old_d_v in changes:
                    d_arr = degs_list[idx]
                    d_arr[pos_u] = old_d_u
                    d_arr[pos_v] = old_d_v
            
            T *= cooling
            steps += 1
            
            if best_energy == 0:
                # Found a graph with no regular 6/7 subsets
                # Verify for all k >= 6
                if verify_graph(adj, n):
                    return n, adj_to_edges(adj, n)
        
        # Return best found
        if verify_graph(best_adj, n):
            return n, adj_to_edges(best_adj, n)
            
        return None, None

    def verify_graph(adj, n):
        # Check all k >= 6
        for k in range(6, n + 1):
            masks, verts_list = get_subsets(n, k)
            for m, v in zip(masks, verts_list):
                # Check regularity
                degs = []
                for u in v:
                    degs.append(bin(adj[u] & m).count('1'))
                if len(set(degs)) == 1:
                    return False
        return True

    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # Try n=21
    result = solve_sa(21, [6, 7], 100.0)
    if result[0] is not None:
        return {'n': result[0], 'edges': result[1]}
        
    # Try n=20
    result = solve_sa(20, [6, 7], 40.0)
    if result[0] is not None:
        return {'n': result[0], 'edges': result[1]}
        
    # Fallback
    return {'n': 5, 'edges': []}

