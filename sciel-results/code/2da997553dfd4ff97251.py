import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def try_add_edge(adj, u, v):
        N_u = adj[u]
        N_v = adj[v]
        
        cu = N_u
        while cu:
            lsb_c = cu & (-cu)
            cu ^= lsb_c
            c = lsb_c.bit_length() - 1
            
            cd = N_u & (~lsb_c)
            while cd:
                lsb_d = cd & (-cd)
                cd ^= lsb_d
                d = lsb_d.bit_length() - 1
                
                candidates = N_v & adj[c] & adj[d]
                mask = ~(1 << u) & ~(1 << v) & ~(1 << c) & ~(1 << d)
                candidates &= mask
                
                if candidates.bit_count() >= 2:
                    return False
        return True

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def systematic_local_search(adj):
        adj = adj[:]
        current_edges = count_edges(adj)
        
        while time.time() < deadline - 5.0:
            # Phase 1: Add edges
            added_any = False
            # Shuffle edges to avoid bias
            shuffled_edges = all_edges[:]
            random.shuffle(shuffled_edges)
            
            for u, v in shuffled_edges:
                if time.time() > deadline - 5.0:
                    break
                if not (adj[u] & (1 << v)):
                    if try_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        current_edges += 1
                        added_any = True
                        break 
            if added_any:
                continue
                
            # Phase 2: Swaps
            swap_found = False
            shuffled_edges = all_edges[:]
            random.shuffle(shuffled_edges)
            
            for a, b in shuffled_edges:
                if time.time() > deadline - 5.0:
                    break
                if not (adj[a] & (1 << b)):
                    continue
                    
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                
                found_add = False
                # Try to add any non-edge
                # We can iterate through all non-edges
                for c, d in all_edges:
                    if time.time() > deadline - 5.0:
                        break
                    if not (adj[c] & (1 << d)):
                        if try_add_edge(adj, c, d):
                            adj[c] |= (1 << d)
                            adj[d] |= (1 << c)
                            found_add = True
                            break
                
                if found_add:
                    swap_found = True
                    break
                else:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    
            if not swap_found:
                break
                
        return current_edges, adj

    best_edges = 0
    best_adj = None
    
    while time.time() < deadline - 10.0:
        adj = [0] * n
        random.shuffle(all_edges)
        for u, v in all_edges:
            if try_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        edges, adj = systematic_local_search(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    if best_adj is not None:
        final_adj = best_adj[:]
        final_edges = best_edges
        for u, v in all_edges:
            if time.time() > deadline - 2.0:
                break
            if not (final_adj[u] & (1 << v)):
                if try_add_edge(final_adj, u, v):
                    final_adj[u] |= (1 << v)
                    final_adj[v] |= (1 << u)
                    final_edges += 1
        if final_edges > best_edges:
            best_edges = final_edges
            best_adj = final_adj

    edges_list = []
    for u in range(n):
        v_mask = best_adj[u]
        while v_mask:
            lsb = v_mask & (-v_mask)
            v_mask ^= lsb
            v = lsb.bit_length() - 1
            if u < v:
                edges_list.append([u, v])
                
    return edges_list
