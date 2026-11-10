import random
import time
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Bitmask helper
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    def can_add(u, v, adj_mask):
        """
        Check if adding edge (u, v) keeps the graph K3,3-free.
        A K3,3 is formed if there exist a1, a2 in N(u) and b1, b2 in N(v)
        such that a1, a2 are both connected to b1, b2.
        """
        if (adj_mask[u] >> v) & 1:
            return True
        
        N_u = adj_mask[u]
        N_v = adj_mask[v]
        
        # We need to check pairs (b1, b2) in N_v
        # For each pair, check if there are at least 2 vertices in N_u connected to both b1 and b2.
        
        # Extract list of neighbors of v
        nv_list = mask_to_list(N_v)
        
        for i in range(len(nv_list)):
            b1 = nv_list[i]
            for j in range(i + 1, len(nv_list)):
                b2 = nv_list[j]
                # Common neighbors of b1 and b2
                common_b1_b2 = adj_mask[b1] & adj_mask[b2]
                # Check how many of these are in N_u
                # If >= 2, we have a1, a2 in N_u connected to both b1, bb2
                # This forms a K2,2 between {a1, a2} and {b1, b2}.
                # Combined with edges (u, b1), (u, b2), (a1, v), (a2, v), this is a K3,3.
                # Note: u is connected to b1, b2. v is connected to a1, a2.
                # The K3,3 partitions are {u, a1, a2} and {v, b1, b2}.
                # Edges:
                # u-b1, u-b2 (exist)
                # a1-v, a2-v (exist because a1, a2 in N_u? NO. a1, a2 are in N_u, so u-a1, u-a2 exist.
                # Wait. The logic above was:
                # a1, a2 in N(u). So edges u-a1, u-a2 exist.
                # b1, b2 in N(v). So edges v-b1, v-b2 exist.
                # We need a1-b1, a1-b2, a2-b1, a2-b2.
                # My check: common_b1_b2 contains vertices connected to both b1 and b2.
                # If a1 is in common_b1_b2, then a1-b1 and a1-b2 exist.
                # So if |common_b1_b2 & N_u| >= 2, we found a1, a2.
                # This confirms the K3,3.
                if (common_b1_b2 & N_u).bit_count() >= 2:
                    return False
        return True

    def get_edge_count(adj_mask):
        return sum(mask.bit_count() for mask in adj_mask) // 2

    def polish(adj_mask, edges_set):
        """Local search to maximize edges."""
        improved = True
        no_improve_count = 0
        while improved and time.time() < deadline:
            improved = False
            no_improve_count += 1
            if no_improve_count > 5:
                break
            
            # 1. Try adding missing edges
            missing = [e for e in all_edges if e not in edges_set]
            random.shuffle(missing)
            for u, v in missing:
                if can_add(u, v, adj_mask):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    edges_set.add((u, v))
                    improved = True
            
            # 2. Try edge swaps (remove 1, add 1)
            # This allows escaping local optima where no single edge can be added.
            if time.time() < deadline - 2:
                edge_list = list(edges_set)
                random.shuffle(edge_list)
                for e1 in edge_list:
                    if time.time() >= deadline - 1:
                        break
                    u1, v1 = e1
                    # Remove e1
                    adj_mask[u1] &= ~(1 << v1)
                    adj_mask[v1] &= ~(1 << u1)
                    edges_set.discard(e1)
                    
                    # Try to add a different edge
                    missing = [e for e in all_edges if e not in edges_set]
                    random.shuffle(missing)
                    added = None
                    for e2 in missing:
                        if can_add(e2[0], e2[1], adj_mask):
                            added = e2
                            break
                    
                    if added:
                        u2, v2 = added
                        adj_mask[u2] |= (1 << v2)
                        adj_mask[v2] |= (1 << u2)
                        edges_set.add(added)
                        improved = True
                    else:
                        # Undo removal
                        adj_mask[u1] |= (1 << v1)
                        adj_mask[v1] |= (1 << u1)
                        edges_set.add(e1)

            # 3. Try 2-for-2 swaps (remove 2, add 2) - expensive, only if time permits
            if time.time() < deadline - 5 and no_improve_count == 1:
                edge_list = list(edges_set)
                random.shuffle(edge_list)
                # Sample a few pairs
                for _ in range(10):
                    if time.time() >= deadline - 2:
                        break
                    if len(edge_list) < 2:
                        break
                    e1, e2 = random.sample(edge_list, 2)
                    
                    # Remove e1, e2
                    u1, v1 = e1
                    u2, v2 = e2
                    adj_mask[u1] &= ~(1 << v1)
                    adj_mask[v1] &= ~(1 << u1)
                    adj_mask[u2] &= ~(1 << v2)
                    adj_mask[v2] &= ~(1 << u2)
                    edges_set.discard(e1)
                    edges_set.discard(e2)
                    
                    # Try to add 2 edges
                    missing = [e for e in all_edges if e not in edges_set]
                    random.shuffle(missing)
                    added = []
                    for e3 in missing:
                        if can_add(e3[0], e3[1], adj_mask):
                            adj_mask[e3[0]] |= (1 << e3[1])
                            adj_mask[e3[1]] |= (1 << e3[0])
                            edges_set.add(e3)
                            added.append(e3)
                        if len(added) == 2:
                            break
                    
                    if len(added) < 2:
                        # Undo
                        adj_mask[u1] |= (1 << v1)
                        adj_mask[v1] |= (1 << u1)
                        adj_mask[u2] |= (1 << v2)
                        adj_mask[v2] |= (1 << u2)
                        edges_set.add(e1)
                        edges_set.add(e2)
                        for e_add in added:
                            adj_mask[e_add[0]] &= ~(1 << e_add[1])
                            adj_mask[e_add[1]] &= ~(1 << e_add[0])
                            edges_set.discard(e_add)
                    else:
                        improved = True

        return adj_mask, edges_set

    best_edge_count = 0
    best_edges = []
    
    # Multi-start
    while time.time() < deadline:
        # Initialize with a random graph of moderate density
        # p=0.4 gives ~32 edges, p=0.6 gives ~58 edges
        # Start sparse to ensure K3,3-free, then add
        adj_mask = [0] * n
        edges_set = set()
        
        # Random greedy initialization
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if can_add(u, v, adj_mask):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges_set.add((u, v))
        
        # Polish
        adj_mask, edges_set = polish(adj_mask, edges_set)
        
        current_count = len(edges_set)
        if current_count > best_edge_count:
            best_edge_count = current_count
            best_edges = list(edges_set)
        
        # If we found a very good graph, we might want to do more polish on it
        if time.time() < deadline - 10:
             # Re-polish the best one? No, just keep searching.
             pass

    # Final polish on the best graph if time remains
    if time.time() < deadline - 5 and best_edges:
        adj_mask = [0] * n
        edges_set = set(best_edges)
        for u, v in best_edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        adj_mask, edges_set = polish(adj_mask, edges_set)
        if len(edges_set) > best_edge_count:
            best_edge_count = len(edges_set)
            best_edges = list(edges_set)

    result = [[u, v] for u, v in best_edges]
    return result
