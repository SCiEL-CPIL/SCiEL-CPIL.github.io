import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110
    
    best_edges = 0
    best_graph = None
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        
        adj_mask = [0] * n
        edges_list = []
        edge_count = 0
        
        for u, v in all_edges:
            valid = True
            
            # Check x in N(u)
            mask_u = adj_mask[u]
            m = mask_u
            while m:
                lsb = m & -m
                x = lsb.bit_length() - 1
                if (adj_mask[x] & adj_mask[v]).bit_count() >= 2:
                    valid = False
                    break
                m -= lsb
            
            if not valid:
                continue
                
            # Check y in N(v)
            mask_v = adj_mask[v]
            m = mask_v
            while m:
                lsb = m & -m
                y = lsb.bit_length() - 1
                if (adj_mask[u] & adj_mask[y]).bit_count() >= 2:
                    valid = False
                    break
                m -= lsb
            
            if valid:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edge_count += 1
                edges_list.append([u, v])
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_graph = edges_list

    return best_graph
