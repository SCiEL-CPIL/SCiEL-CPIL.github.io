import random
import time
import sys
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin under 150s limit
    
    # Vertices 0..19
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid_addition(adj, u, v):
        """
        Check if adding edge (u, v) keeps the graph K2,3-free.
        """
        # 1. Check common neighbors of u and v
        common_uv = np.sum(adj[u] & adj[v])
        if common_uv > 2:
            return False
        
        # 2. Check neighbors of u
        neighbors_u = np.where(adj[u] == 1)[0]
        for w in neighbors_u:
            if w == v:
                continue
            common_wv = np.sum(adj[w] & adj[v])
            if common_wv > 1:
                return False
                
        # 3. Check neighbors of v
        neighbors_v = np.where(adj[v] == 1)[0]
        for w in neighbors_v:
            if w == u:
                continue
            common_uw = np.sum(adj[u] & adj[w])
            if common_uw > 1:
                return False
                
        return True

    def vectorized_can_add(adj, u, v):
        """
        Vectorized check for whether edge (u,v) can be added.
        Returns True if valid.
        """
        # Condition 1: |N(u) ∩ N(v)| <= 2
        if np.sum(adj[u] & adj[v]) > 2:
            return False
        
        # Condition 2: For all w in N(u), |N(w) ∩ N(v)| <= 1
        # N(u) are vertices where adj[u] == 1
        # We need to check for each w in N(u): sum(adj[w] & adj[v]) <= 1
        # This is equivalent to: max over w in N(u) of (adj[w] & adj[v]).sum() <= 1
        # Which means: for each w in N(u), the number of common neighbors with v is at most 1
        # We can compute all at once: for each w, count = sum(adj[w] & adj[v])
        # But we only care about w in N(u)
        
        # Get indices of N(u)
        nu_indices = np.where(adj[u] == 1)[0]
        if len(nu_indices) == 0:
            pass  # No neighbors, condition 2 vacuously true
        else:
            # For each w in N(u), compute common neighbors with v
            # adj[nu_indices] is shape (len(nu_indices), n)
            # adj[v] is shape (n,)
            # We want: for each row in adj[nu_indices], sum(row & adj[v]) <= 1
            common_counts = np.sum(adj[nu_indices] & adj[v], axis=1)
            if np.any(common_counts > 1):
                return False
        
        # Condition 3: For all w in N(v), |N(u) ∩ N(w)| <= 1
        nv_indices = np.where(adj[v] == 1)[0]
        if len(nv_indices) == 0:
            pass
        else:
            common_counts = np.sum(adj[u] & adj[nv_indices], axis=1)
            if np.any(common_counts > 1):
                return False
        
        return True

    def verify(edges):
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return None
            if adj[u, v]:
                return None
            adj[u, v] = 1
            adj[v, u] = 1
        for u in range(n):
            for v in range(u + 1, n):
                if np.sum(adj[u] & adj[v]) > 2:
                    return None
        return edges

    def greedy_build(seed):
        rng = random.Random(seed)
        adj = np.zeros((n, n), dtype=np.uint8)
        pairs = all_pairs[:]
        rng.shuffle(pairs)
        
        for u, v in pairs:
            if vectorized_can_add(adj, u, v):
                adj[u, v] = 1
                adj[v, u] = 1
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return edges

    def local_search(edges, seed):
        rng = random.Random(seed)
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
            
        current_count = len(edges)
        best_count = current_count
        best_adj = adj.copy()
        
        improved = True
        iters = 0
        max_iters = 100000
        
        while improved and iters < max_iters and time.time() < deadline - 5.0:
            improved = False
            iters += 1
            
            # 1. Try to add an edge (try all non-edges, pick first valid)
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
            rng.shuffle(non_edges)
            
            added = False
            for u, v in non_edges:
                if vectorized_can_add(adj, u, v):
                    adj[u, v] = 1
                    adj[v, u] = 1
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj.copy()
                    improved = True
                    added = True
                    break
            
            if added:
                continue
                
            # 2. Try 1-2 swap (remove 1, add 2)
            edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
            if not edge_list:
                break
            
            swaps_tried = 0
            max_swaps = 50
            while swaps_tried < max_swaps and time.time() < deadline - 5.0:
                swaps_tried += 1
                u_rem, v_rem = rng.choice(edge_list)
                
                adj_save = adj.copy()
                count_save = current_count
                
                adj[u_rem, v_rem] = 0
                adj[v_rem, u_rem] = 0
                current_count -= 1
                
                added_count = 0
                non_edges_swap = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
                rng.shuffle(non_edges_swap)
                
                for u_add, v_add in non_edges_swap[:80]:
                    if added_count == 2:
                        break
                    if vectorized_can_add(adj, u_add, v_add):
                        adj[u_add, v_add] = 1
                        adj[v_add, u_add] = 1
                        current_count += 1
                        added_count += 1
                        
                if current_count > count_save:
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj.copy()
                    improved = True
                    break
                else:
                    adj = adj_save
                    current_count = count_save
            
            # 3. Try 2-2 swap (remove 2, add 2) - more aggressive
            if not improved and time.time() < deadline - 5.0:
                edge_list_22 = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
                if len(edge_list_22) >= 2:
                    swaps_tried_22 = 0
                    max_swaps_22 = 30
                    while swaps_tried_22 < max_swaps_22 and time.time() < deadline - 5.0:
                        swaps_tried_22 += 1
                        # Pick 2 random edges to remove
                        rem_edges = rng.sample(edge_list_22, 2)
                        (u1_rem, v1_rem), (u2_rem, v2_rem) = rem_edges
                        
                        adj_save = adj.copy()
                        count_save = current_count
                        
                        # Remove both edges
                        adj[u1_rem, v1_rem] = 0
                        adj[v1_rem, u1_rem] = 0
                        adj[u2_rem, v2_rem] = 0
                        adj[v2_rem, u2_rem] = 0
                        current_count -= 2
                        
                        # Try to add 2 edges
                        added_count = 0
                        non_edges_22 = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
                        rng.shuffle(non_edges_22)
                        
                        for u_add, v_add in non_edges_22[:100]:
                            if added_count == 2:
                                break
                            if vectorized_can_add(adj, u_add, v_add):
                                adj[u_add, v_add] = 1
                                adj[v_add, u_add] = 1
                                current_count += 1
                                added_count += 1
                                
                        if current_count > count_save:
                            if current_count > best_count:
                                best_count = current_count
                                best_adj = adj.copy()
                            improved = True
                            break
                        else:
                            adj = adj_save
                            current_count = count_save
        
        # Final polish: systematic check for any addable edge
        if time.time() < deadline - 2.0:
            non_edges_final = [(u, v) for u in range(n) for v in range(u + 1, n) if not best_adj[u, v]]
            for u, v in non_edges_final:
                if vectorized_can_add(best_adj, u, v):
                    best_adj[u, v] = 1
                    best_adj[v, u] = 1
                    best_count += 1
            
            # Systematic 1-2 swap polish
            if time.time() < deadline - 1.0:
                edge_list_final = [(u, v) for u in range(n) for v in range(u + 1, n) if best_adj[u, v]]
                for u_rem, v_rem in edge_list_final:
                    if time.time() > deadline - 1.0:
                        break
                    adj_save = best_adj.copy()
                    count_save = best_count
                    best_adj[u_rem, v_rem] = 0
                    best_adj[v_rem, u_rem] = 0
                    best_count -= 1
                    
                    added_count = 0
                    non_edges_swap = [(u, v) for u in range(n) for v in range(u + 1, n) if not best_adj[u, v]]
                    for u_add, v_add in non_edges_swap:
                        if added_count == 2:
                            break
                        if vectorized_can_add(best_adj, u_add, v_add):
                            best_adj[u_add, v_add] = 1
                            best_adj[v_add, u_add] = 1
                            best_count += 1
                            added_count += 1
                    
                    if best_count <= count_save:
                        best_adj = adj_save
                        best_count = count_save
        
        # Return best found
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if best_adj[u, v]:
                    edges.append([u, v])
        return edges

    best_edges = None
    best_count = 0
    
    # Run multiple restarts with adaptive stopping
    num_restarts = 500
    stagnant_count = 0
    
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Greedy construction
        edges = greedy_build(i * 12345 + 6789)
        
        # Local search
        improved_edges = local_search(edges, i * 98765 + 4321)
        
        if len(improved_edges) > best_count:
            best_count = len(improved_edges)
            best_edges = improved_edges
            stagnant_count = 0
        else:
            stagnant_count += 1
        
        # Stop if no improvement for 30 consecutive restarts
        if stagnant_count >= 30:
            break

    if best_edges is None:
        return []
        
    # Final verification and cleanup
    final_edges = verify(best_edges)
    if final_edges is None:
        adj_check = np.zeros((n, n), dtype=np.uint8)
        for u, v in best_edges:
            adj_check[u, v] = 1
            adj_check[v, u] = 1
        result = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj_check[u, v]:
                    result.append([u, v])
        return result
    
    return final_edges

if __name__ == "__main__":
    result = construct()
    print(f"Edges: {len(result)}")
