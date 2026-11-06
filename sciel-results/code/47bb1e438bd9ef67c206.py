import itertools

def construct():
        n = 16
        # Adjacency matrix as list of integers (bitmasks)
        adj = [0] * n
        edges_set = set()
        for u in range(n):
            for v in range(u+1, n):
                edges_set.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Q3 structure
        q3_adj = {i: [] for i in range(8)}
        for i in range(8):
            for j in range(8):
                if i != j and (i ^ j) in (1, 2, 4):
                    q3_adj[i].append(j)
        
        # Pre-compute all 8-subsets? No, 12870 is fine to iterate.
        # But we can pre-compute the mask for each subset?
        # subset_masks = {subset: mask}
        
        def has_q3(adj, edges_set):
            # Iterate subsets
            for subset in itertools.combinations(range(n), 8):
                # Fast edge count using bitmasks
                mask = 0
                for u in subset:
                    mask |= (1 << u)
                
                # Count edges in induced subgraph
                # Sum of degrees within subset / 2
                edge_count = 0
                for u in subset:
                    edge_count += (adj[u] & mask).bit_count()
                edge_count //= 2
                
                if edge_count < 12:
                    continue
                
                # Backtracking
                mapping = {} # q_idx -> g_idx
                # Order Q3 vertices? Maybe by degree (all 3) or just 0..7
                # To improve pruning, we can try to map vertices that have many constraints first.
                # In Q3, all vertices are symmetric.
                
                def backtrack(idx):
                    if idx == 8:
                        return True
                    q_v = idx
                    # Try each graph vertex in subset
                    for g_v in subset:
                        if g_v in mapping.values():
                            continue
                        
                        # Check compatibility with already mapped neighbors
                        compatible = True
                        mapped_neighbors_count = 0
                        unmapped_neighbors_count = 0
                        # We need to check edges to mapped neighbors
                        for q_neighbor in q3_adj[q_v]:
                            if q_neighbor in mapping:
                                u, v = (mapping[q_neighbor], g_v)
                                if u > v: u, v = v, u
                                if (u, v) not in edges_set:
                                    compatible = False
                                    break
                                mapped_neighbors_count += 1
                            else:
                                unmapped_neighbors_count += 1
                        
                        if not compatible:
                            continue
                        
                        # Pruning: Check if there are enough available neighbors for unmapped Q3 neighbors
                        # The available neighbors are those in subset, not in mapping.values(), and connected to g_v
                        # We need at least unmapped_neighbors_count such vertices.
                        # Note: These vertices also need to be compatible with their own constraints later,
                        # but a simple count check is a good prune.
                        
                        # Get set of neighbors of g_v in subset that are not yet mapped
                        # adj[g_v] & mask gives all neighbors in subset
                        # We need to remove those already in mapping.values()
                        mapped_mask = 0
                        for gv in mapping.values():
                            mapped_mask |= (1 << gv)
                        
                        available_neighbors_mask = (adj[g_v] & mask) & ~mapped_mask
                        available_count = available_neighbors_mask.bit_count()
                        
                        if available_count < unmapped_neighbors_count:
                            continue
                        
                        # Additional prune: The available neighbors must be able to form the required structure?
                        # Too complex. Stick to count.
                        
                        mapping[q_v] = g_v
                        if backtrack(idx + 1):
                            return True
                        del mapping[q_v]
                    return False

                if backtrack(0):
                    q3_edges = []
                    for i in range(8):
                        for j in q3_adj[i]:
                            if i < j:
                                u, v = mapping[i], mapping[j]
                                if u > v: u, v = v, u
                                q3_edges.append((u, v))
                    return q3_edges
            return None

        # Greedy removal
        # Use a better heuristic for edge selection
        # Remove edge that is part of the most "dense" Q3?
        # Or simply remove the edge with the highest degree sum in the graph?
        # The current code removes the edge in the found Q3 with max degree sum.
        
        max_iterations = 500 # Increase if fast
        for _ in range(max_iterations):
            q3_edges = has_q3(adj, edges_set)
            if q3_edges is None:
                break
            
            # Heuristic: Remove edge with max (deg_u + deg_v)
            best_edge = None
            best_score = -1
            for u, v in q3_edges:
                deg_u = adj[u].bit_count()
                deg_v = adj[v].bit_count()
                score = deg_u + deg_v
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
            
            if best_edge:
                u, v = best_edge
                edges_set.remove(best_edge)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            
        return [list(e) for e in edges_set]
