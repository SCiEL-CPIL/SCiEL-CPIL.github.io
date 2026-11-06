import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit
    
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj_masks = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)
    
    def bits_to_list(mask):
        result = []
        temp = mask
        while temp:
            lsb = temp & -temp
            idx = (lsb.bit_length() - 1)
            result.append(idx)
            temp ^= lsb
        return result

    def has_k3_in_mask(adj, mask):
        """Check if induced subgraph on mask has a K3 (triangle)"""
        verts = bits_to_list(mask)
        for i in range(len(verts)):
            u = verts[i]
            neigh_in_mask = adj[u] & mask & ~(1 << u)
            v_list = bits_to_list(neigh_in_mask)
            for v in v_list:
                if adj[v] & neigh_in_mask & ~(1 << v) & ~(1 << u):
                    return True
        return False

    def has_i3_in_mask(adj, mask):
        """Check if induced subgraph on mask has an independent set of size 3"""
        verts = bits_to_list(mask)
        for i in range(len(verts)):
            u = verts[i]
            non_neigh_in_mask = mask & ~adj[u] & ~(1 << u)
            v_list = bits_to_list(non_neigh_in_mask)
            for v in v_list:
                if non_neigh_in_mask & ~adj[v] & ~(1 << v) & ~(1 << u):
                    return True
        return False

    def has_k4_in_mask(adj, mask):
        """Check if induced subgraph on mask has a K4"""
        verts = bits_to_list(mask)
        for u in verts:
            neigh = adj[u] & mask & ~(1 << u)
            if has_k3_in_mask(adj, neigh):
                return True
        return False

    def has_i4_in_mask(adj, mask):
        """Check if induced subgraph on mask has an independent set of size 4"""
        verts = bits_to_list(mask)
        for u in verts:
            non_neigh = mask & ~adj[u] & ~(1 << u)
            if has_i3_in_mask(adj, non_neigh):
                return True
        return False

    def is_valid(adj, n):
        """Check if the graph is (5,5)-free: no K5 and no independent 5-set"""
        all_mask = (1 << n) - 1
        for u in range(n):
            if has_k4_in_mask(adj, adj[u]):
                return False
            non_neigh = all_mask & ~adj[u] & ~(1 << u)
            if has_i4_in_mask(adj, non_neigh):
                return False
        return True

    # Verify seed
    if not is_valid(adj_masks, n):
        return n, []

    best_n = n
    best_adj = adj_masks[:]
    current_n = n
    current_adj = adj_masks[:]
    all_mask = (1 << current_n) - 1

    def find_conflict(adj, n, new_conn_mask):
        """
        Check if adding a new vertex (index n) with connections new_conn_mask 
        creates a K5 or I5. Returns (type, vertices) if found, else (None, None).
        type is 'K5' or 'I5'. vertices is a list of 5 vertices including n.
        """
        # Check for K5 involving the new vertex
        # A K5 involving n requires 4 neighbors of n that form a K4 in the existing graph.
        # We iterate over K4s in the neighborhood of n? No, that's expensive.
        # Instead, iterate over all K4s in the existing graph? No.
        
        # Efficient check:
        # For each vertex u in the neighborhood of n:
        #   Check if N(u) \cap N(n) contains a K3.
        
        neigh_n = new_conn_mask
        for u in bits_to_list(neigh_n):
            # Common neighbors of u and n
            common = adj[u] & neigh_n & ~(1 << u)
            if has_k3_in_mask(adj, common):
                # Found a K3 in common neighbors of u and n.
                # u + n + these 3 vertices form a K5.
                k3_verts = bits_to_list(common)
                # We need to pick 3 from k3_verts that form a triangle.
                # has_k3_in_mask returned True, so we know one exists.
                # For repair, we just need to know which vertices are involved.
                # Let's just return the vertices involved in the first found triangle.
                # Re-finding the specific triangle is redundant if we just want to flip an edge.
                # We can just flip an edge incident to n or u.
                return ('K5', [n, u] + k3_verts[:3]) # Approximate, just need indices
        
        # Check for I5 involving the new vertex
        # A I5 involving n requires 4 non-neighbors of n that form an I4 in the existing graph.
        non_neigh_n = all_mask & ~new_conn_mask
        for u in bits_to_list(non_neigh_n):
            # Common non-neighbors of u and n
            # Non-neighbors of u in the existing graph: ~adj[u] & all_mask & ~(1<<u)
            # We want vertices v such that v is not adj to u and v is not adj to n.
            non_neigh_u = all_mask & ~adj[u] & ~(1 << u)
            common_non = non_neigh_u & non_neigh_n & ~(1 << u)
            if has_i3_in_mask(adj, common_non):
                i3_verts = bits_to_list(common_non)
                return ('I5', [n, u] + i3_verts[:3])
        
        return (None, None)

    def try_add_vertex_local_search(adj, n, max_iters=1000):
        """
        Attempt to add a new vertex to adj (size n) using local search on the 
        new vertex's connections.
        Returns (new_n, new_adj) if successful, else None.
        """
        # Start with a random balanced connection set
        # Paley graph is self-complementary and vertex-transitive, so degrees are (q-1)/2 = 18.
        # Aim for degree ~ n/2.
        target_degree = n // 2
        conn = 0
        while conn.bit_count() < target_degree:
            v = random.randint(0, n - 1)
            conn |= (1 << v)
        
        # Local search
        for _ in range(max_iters):
            conflict_type, conflict_verts = find_conflict(adj, n, conn)
            if conflict_type is None:
                # Success! Construct new adjacency
                new_n = n + 1
                new_adj = adj + [0]
                for v in range(n):
                    if conn & (1 << v):
                        new_adj[v] |= (1 << new_n)
                        new_adj[new_n] |= (1 << v)
                return new_n, new_adj
            
            # Repair: Flip an edge in the conflict
            # If K5, we need to break the clique.
            # The conflict involves n and 4 other vertices.
            # We can flip the edge between n and one of the other vertices.
            # Or flip an edge between two of the other vertices (but that changes base graph, risky).
            # Let's only flip edges incident to n to keep base graph valid.
            
            if conflict_type == 'K5':
                # Conflict verts: [n, u, v1, v2, v3]
                # n is connected to all of them.
                # We pick one of the other vertices and disconnect n from it.
                # Or connect n to a non-neighbor? No, that might create more K5s.
                # Disconnecting is safer for K5.
                # Pick a random vertex from the conflict (excluding n)
                other_verts = [v for v in conflict_verts if v != n]
                v_to_flip = random.choice(other_verts)
                # If currently connected, disconnect.
                if conn & (1 << v_to_flip):
                    conn &= ~(1 << v_to_flip)
                else:
                    # Should not happen if it's a K5 conflict involving n, 
                    # but if we picked a vertex not in the specific triangle found...
                    # Actually find_conflict returns specific verts.
                    # Let's just ensure we flip a connection that is part of the conflict.
                    # For K5, n MUST be connected to the other 4.
                    # So we must disconnect one.
                    conn &= ~(1 << v_to_flip)
                    
            elif conflict_type == 'I5':
                # Conflict verts: [n, u, v1, v2, v3]
                # n is NOT connected to any of them.
                # We need to connect n to one of them.
                other_verts = [v for v in conflict_verts if v != n]
                v_to_flip = random.choice(other_verts)
                conn |= (1 << v_to_flip)
                
            # Occasional random perturbation to escape local minima
            if random.random() < 0.05:
                v = random.randint(0, n - 1)
                if random.random() < 0.5:
                    conn |= (1 << v)
                else:
                    conn &= ~(1 << v)

        return None

    # Main search loop
    # We try to extend the graph one vertex at a time.
    # If we get stuck, we might try to perturb the existing graph and restart.
    
    # Since we only have 37 -> 42 range, and 42 is the known max, 
    # and 38 is the immediate next step.
    
    attempts_per_vertex = 500
    max_vertices_to_try = 42 # R(5,5) is open between 43 and 46, so 42 is a safe target
    
    while time.time() < deadline and current_n < max_vertices_to_try:
        added = False
        for attempt in range(attempts_per_vertex):
            if time.time() > deadline:
                break
            
            # Try to add a vertex
            result = try_add_vertex_local_search(current_adj, current_n, max_iters=200)
            if result is not None:
                current_n, current_adj = result
                best_n = current_n
                best_adj = current_adj[:]
                all_mask = (1 << current_n) - 1
                added = True
                break
        
        if not added:
            # If we couldn't add a vertex, we might try to perturb the current graph 
            # to find a different structure that allows extension.
            # However, perturbing a valid graph might make it invalid.
            # We can try to flip edges in the current graph if they don't break validity.
            
            perturbed = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                i = random.randint(0, current_n - 1)
                j = random.randint(0, current_n - 1)
                if i == j: continue
                
                # Try flipping edge (i, j)
                new_adj = current_adj[:]
                if new_adj[i] & (1 << j):
                    # Remove edge
                    # Check if this creates a new K5 or I5
                    # This is expensive to check globally.
                    # We can check locally:
                    # Removing edge (i,j) might create a new I5? No, removing edges creates independent sets.
                    # Adding edges creates cliques.
                    
                    # Let's just try it and check validity of the whole graph?
                    # is_valid is O(n^5) roughly. For n=37, it's okay.
                    
                    new_adj[i] &= ~(1 << j)
                    new_adj[j] &= ~(1 << i)
                    
                    if is_valid(new_adj, current_n):
                        current_adj = new_adj
                        perturbed = True
                        break
                else:
                    # Add edge
                    new_adj[i] |= (1 << j)
                    new_adj[j] |= (1 << i)
                    
                    if is_valid(new_adj, current_n):
                        current_adj = new_adj
                        perturbed = True
                        break
            
            if not perturbed:
                # If we can't perturb and can't extend, we are stuck.
                # Return best found.
                break

    final_edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                final_edges.append((i, j))
                
    return best_n, final_edges
