import random
import time

def construct():
    n = 20
    start_time = time.time()
    max_time = 115.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    def get_all_pairs(n):
        return [(u, v) for u in range(n) for v in range(u + 1, n)]

    def is_valid_add_c4(u, v, adj_mask, common):
        if (adj_mask[u] >> v) & 1:
            return False
        if common[u][v] > 1:
            return False
        # Adding (u,v) to C4-free graph:
        # common[u][w] must be 0 for all w in N(v) \ {u}
        # common[v][w] must be 0 for all w in N(u) \ {v}
        
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != u and common[u][w] > 0:
                return False
            m -= lsb
            
        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != v and common[v][w] > 0:
                return False
            m -= lsb
            
        return True

    def add_edge_c4(u, v, adj_mask, common, edges):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.append([u, v])
        
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != u:
                common[u][w] += 1
                common[w][u] += 1
            m -= lsb
            
        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != v:
                common[v][w] += 1
                common[w][v] += 1
            m -= lsb

    def remove_edge_c4(u, v, adj_mask, common, edges):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edges.remove([u, v])
        
        # Decrement common
        # For w in CURRENT N(v) (which is OLD N(v) \ {u}), common[u][w] -= 1
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            m -= lsb
            
        # For w in CURRENT N(u) (which is OLD N(u) \ {v}), common[v][w] -= 1
        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] -= 1
            common[w][v] -= 1
            m -= lsb

    def greedy_fill_c4(adj_mask, common, edges):
        all_pairs = get_all_pairs(n)
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if is_valid_add_c4(u, v, adj_mask, common):
                add_edge_c4(u, v, adj_mask, common, edges)

    def local_search_two_c4(e1, e2):
        # Initialize state
        adj1 = [0] * n
        common1 = [[0] * n for _ in range(n)]
        edges1 = []
        for u, v in e1:
            add_edge_c4(u, v, adj1, common1, edges1)
            
        adj2 = [0] * n
        common2 = [[0] * n for _ in range(n)]
        edges2 = []
        for u, v in e2:
            add_edge_c4(u, v, adj2, common2, edges2)
            
        current_count = len(edges1) + len(edges2)
        best_local_count = current_count
        best_local_e1 = list(edges1)
        best_local_e2 = list(edges2)
        
        all_pairs = get_all_pairs(n)
        
        for _ in range(500): # Steps per restart
            # Try to add any valid edge to either G1 or G2
            # We prioritize adding to the graph with more potential or randomly
            shuffled_pairs = all_pairs[:]
            random.shuffle(shuffled_pairs)
            
            added_any = False
            for u, v in shuffled_pairs:
                # Try adding to G1
                if is_valid_add_c4(u, v, adj1, common1):
                    add_edge_c4(u, v, adj1, common1, edges1)
                    current_count += 1
                    if current_count > best_local_count:
                        best_local_count = current_count
                        best_local_e1 = list(edges1)
                        best_local_e2 = list(edges2)
                    added_any = True
                    break
                # Try adding to G2
                if is_valid_add_c4(u, v, adj2, common2):
                    add_edge_c4(u, v, adj2, common2, edges2)
                    current_count += 1
                    if current_count > best_local_count:
                        best_local_count = current_count
                        best_local_e1 = list(edges1)
                        best_local_e2 = list(edges2)
                    added_any = True
                    break
            
            if not added_any:
                # Local optimum reached
                # Kick: remove 1 or 2 edges from either G1 or G2
                kick_size = 1 if random.random() < 0.5 else 2
                
                # Select edges to remove randomly from the union
                # We need to pick edges from edges1 or edges2
                # To ensure we don't pick more than available, we create a list of (graph_id, u, v)
                available_moves = []
                for u, v in edges1:
                    available_moves.append((1, u, v))
                for u, v in edges2:
                    available_moves.append((2, u, v))
                    
                if not available_moves:
                    break
                
                random.shuffle(available_moves)
                to_remove = available_moves[:kick_size]
                
                for gid, u, v in to_remove:
                    if gid == 1:
                        remove_edge_c4(u, v, adj1, common1, edges1)
                    else:
                        remove_edge_c4(u, v, adj2, common2, edges2)
                    current_count -= 1
                    
                # Try to fill again
                for step in range(50):
                    shuffled_pairs = all_pairs[:]
                    random.shuffle(shuffled_pairs)
                    added = False
                    for u, v in shuffled_pairs:
                        if is_valid_add_c4(u, v, adj1, common1):
                            add_edge_c4(u, v, adj1, common1, edges1)
                            current_count += 1
                            if current_count > best_local_count:
                                best_local_count = current_count
                                best_local_e1 = list(edges1)
                                best_local_e2 = list(edges2)
                            added = True
                            break
                        if is_valid_add_c4(u, v, adj2, common2):
                            add_edge_c4(u, v, adj2, common2, edges2)
                            current_count += 1
                            if current_count > best_local_count:
                                best_local_count = current_count
                                best_local_e1 = list(edges1)
                                best_local_e2 = list(edges2)
                            added = True
                            break
                    if not added:
                        break

        return best_local_e1, best_local_e2

    # Main loop
    while time.time() - start_time < max_time:
        # 1. Greedy Initialization for two C4-free graphs
        adj1 = [0] * n
        common1 = [[0] * n for _ in range(n)]
        edges1 = []
        greedy_fill_c4(adj1, common1, edges1)
        
        adj2 = [0] * n
        common2 = [[0] * n for _ in range(n)]
        edges2 = []
        greedy_fill_c4(adj2, common2, edges2)
        
        # 2. Local Search
        res_e1, res_e2 = local_search_two_c4(edges1, edges2)
        
        all_res_edges = res_e1 + res_e2
        if len(all_res_edges) > best_count:
            best_count = len(all_res_edges)
            best_edges = all_res_edges
            
        # Check time
        if time.time() - start_time > max_time - 5:
            break

    # Final Verification
    if not best_edges:
        # Fallback
        best_edges = [[0, 1]]
    else:
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        valid = True
        for u in range(n):
            for v in range(u + 1, n):
                if bin(adj[u] & adj[v]).count('1') > 2:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
             # This should not happen if state maintenance is correct
             # But for safety, return a small valid graph
             return [[0, 1]]

    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
