import random
import time

def construct():
    n = 20
    start_time = time.time()
    max_time = 115.0  # Safety margin for 120s limit
    
    best_edges = []
    best_count = 0
    
    def get_all_pairs(n):
        return [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid_add(u, v, adj_mask, common):
        # Check if edge (u, v) can be added without violating K2,3-free constraint
        if (adj_mask[u] >> v) & 1:
            return False
            
        # Check neighbors of v
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != u and common[u][w] >= 2:
                return False
            m -= lsb
            
        # Check neighbors of u
        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != v and common[v][w] >= 2:
                return False
            m -= lsb
            
        return True

    def add_edge(u, v, adj_mask, common, edges):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.append([u, v])
        
        # Update common counts
        mask_v = adj_mask[v] & ~(1 << u)
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] += 1
            common[w][u] += 1
            m -= lsb
            
        mask_u = adj_mask[u] & ~(1 << v)
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] += 1
            common[w][v] += 1
            m -= lsb

    def remove_edge(u, v, adj_mask, common, edges):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        # Remove from edges list efficiently
        for i in range(len(edges) - 1, -1, -1):
            if (edges[i][0] == u and edges[i][1] == v) or (edges[i][0] == v and edges[i][1] == u):
                edges.pop(i)
                break
        
        # Update common counts (decrement)
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            m -= lsb
            
        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] -= 1
            common[w][v] -= 1
            m -= lsb

    def greedy_fill(adj_mask, common, edges, seed=None):
        all_pairs = get_all_pairs(n)
        if seed is not None:
            random.seed(seed)
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if is_valid_add(u, v, adj_mask, common):
                add_edge(u, v, adj_mask, common, edges)

    def local_search(e1, max_steps=12000):
        # Initialize state from e1
        adj = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        for u, v in e1:
            add_edge(u, v, adj, common, edges)
            
        current_count = len(edges)
        best_local_count = current_count
        best_local_edges = [e[:] for e in edges]
        
        all_pairs = get_all_pairs(n)
        
        no_improve_count = 0
        max_no_improve = 800
        
        for step in range(max_steps):
            if time.time() - start_time > max_time - 5:
                break
            if no_improve_count > max_no_improve:
                break
                
            # 1. Try to add a random edge (multiple attempts per step)
            added = False
            for _ in range(5):
                u, v = random.choice(all_pairs)
                if is_valid_add(u, v, adj, common):
                    add_edge(u, v, adj, common, edges)
                    current_count += 1
                    if current_count > best_local_count:
                        best_local_count = current_count
                        best_local_edges = [e[:] for e in edges]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    added = True
                    break
            
            if added:
                continue
            
            # 2. Try a swap: Remove one edge, add this one
            if edges:
                ru, rv = random.choice(edges)
                remove_edge(ru, rv, adj, common, edges)
                current_count -= 1
                
                # Try to add a random edge
                u, v = random.choice(all_pairs)
                if is_valid_add(u, v, adj, common):
                    add_edge(u, v, adj, common, edges)
                    current_count += 1
                    if current_count > best_local_count:
                        best_local_count = current_count
                        best_local_edges = [e[:] for e in edges]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # Revert: Add back ru, rv
                    add_edge(ru, rv, adj, common, edges)
                    current_count += 1
                    no_improve_count += 1
                continue
            
            # 3. Random kick: Remove 2 edges, add 1
            if len(edges) >= 2:
                e_idx1 = random.randint(0, len(edges)-1)
                e_idx2 = random.randint(0, len(edges)-1)
                if e_idx1 == e_idx2:
                    continue
                e1_remove = edges[e_idx1]
                e2_remove = edges[e_idx2]
                
                remove_edge(e1_remove[0], e1_remove[1], adj, common, edges)
                remove_edge(e2_remove[0], e2_remove[1], adj, common, edges)
                current_count -= 2
                
                u, v = random.choice(all_pairs)
                if is_valid_add(u, v, adj, common):
                    add_edge(u, v, adj, common, edges)
                    current_count += 1
                    if current_count > best_local_count:
                        best_local_count = current_count
                        best_local_edges = [e[:] for e in edges]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # Revert
                    add_edge(e1_remove[0], e1_remove[1], adj, common, edges)
                    add_edge(e2_remove[0], e2_remove[1], adj, common, edges)
                    current_count += 2
                    no_improve_count += 1

        return best_local_edges

    def polish_pass(edges_list):
        """Final polish: systematically try to add any missing edge"""
        adj = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        for u, v in edges_list:
            add_edge(u, v, adj, common, edges)
        
        improved = True
        while improved and time.time() - start_time < max_time - 3:
            improved = False
            all_pairs = get_all_pairs(n)
            for u, v in all_pairs:
                if is_valid_add(u, v, adj, common):
                    add_edge(u, v, adj, common, edges)
                    improved = True
        
        return edges

    def verify_edges(edge_list):
        if not edge_list:
            return True
        adj = [0] * n
        for u, v in edge_list:
            if u == v:
                return False
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for u in range(n):
            for v in range(u + 1, n):
                if bin(adj[u] & adj[v]).count('1') > 2:
                    return False
        return True

    # Main loop
    seed_counter = 0
    while time.time() - start_time < max_time - 15:
        seed_counter += 1
        adj1 = [0] * n
        common1 = [[0] * n for _ in range(n)]
        edges1 = []
        greedy_fill(adj1, common1, edges1, seed=seed_counter * 12345)
        
        res_edges = local_search(edges1)
        
        if len(res_edges) > best_count:
            best_count = len(res_edges)
            best_edges = [e[:] for e in res_edges]

    # Final polish on best result
    if best_edges and time.time() - start_time < max_time - 2:
        polished = polish_pass(best_edges)
        if len(polished) > best_count:
            best_edges = polished
            best_count = len(polished)

    # Final Verification
    if not best_edges:
        best_edges = [[0, 1]]
    else:
        if not verify_edges(best_edges):
            # Fallback to a simple valid graph if something went wrong
            best_edges = []
            adj = [0] * n
            common = [[0] * n for _ in range(n)]
            # Try to build a simple valid graph
            all_pairs = get_all_pairs(n)
            random.seed(42)
            random.shuffle(all_pairs)
            for u, v in all_pairs:
                if is_valid_add(u, v, adj, common):
                    add_edge(u, v, adj, common, best_edges)
            if not best_edges:
                best_edges = [[0, 1]]

    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
