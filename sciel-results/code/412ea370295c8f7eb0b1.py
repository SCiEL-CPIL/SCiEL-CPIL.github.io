import time
import random

def construct():
    N = 40
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def check_add(u, v, adj):
        """
        Check if adding edge (u,v) creates a C6.
        Returns True if safe, False if creates C6.
        """
        nu = adj[u]
        nv = adj[v]
        
        a_mask = nu
        while a_mask:
            a_bit = a_mask & (-a_mask)
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            d_mask = nv
            while d_mask:
                d_bit = d_mask & (-d_mask)
                d = d_bit.bit_length() - 1
                d_mask ^= d_bit
                
                # Check if a and d share a neighbor b
                common = adj[a] & adj[d]
                common &= ~(1 << u)
                common &= ~(1 << v)
                if common:
                    return False
        return True

    def random_greedy_init(rng):
        adj = [0] * N
        edges = []
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        rng.shuffle(all_edges)
        
        for u, v in all_edges:
            if check_add(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        return edges, adj

    def local_search(edges, adj, rng, max_iters=10000):
        current_edges = set(edges)
        current_adj = list(adj)
        current_count = len(current_edges)
        best_count = current_count
        best_edges = set(current_edges)
        best_adj = list(current_adj)
        
        no_improve = 0
        max_no_improve = 2000
        
        for it in range(max_iters):
            if time.time() >= deadline:
                break
                
            # 1. Try to add safe edge
            safe_edge = None
            for u in range(N):
                for v in range(u + 1, N):
                    if (u, v) in current_edges:
                        continue
                    if check_add(u, v, current_adj):
                        safe_edge = (u, v)
                        break
                if safe_edge:
                    break
            
            if safe_edge:
                u, v = safe_edge
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges.add((u, v))
                current_count += 1
                no_improve = 0
                
                if current_count > best_count:
                    best_count = current_count
                    best_edges = set(current_edges)
                    best_adj = list(current_adj)
            else:
                # 2. Swap
                if current_count == 0:
                    break
                    
                e_remove = rng.choice(list(current_edges))
                u1, v1 = e_remove
                
                pre_swap_count = current_count
                
                saved_adj = list(current_adj)
                saved_edges = set(current_edges)
                
                current_adj[u1] &= ~(1 << v1)
                current_adj[v1] &= ~(1 << u1)
                current_edges.discard(e_remove)
                current_count -= 1
                
                found = 0
                # Scan for 2 safe edges
                for u in range(N):
                    for v in range(u + 1, N):
                        if (u, v) in current_edges:
                            continue
                        if check_add(u, v, current_adj):
                            current_adj[u] |= (1 << v)
                            current_adj[v] |= (1 << u)
                            current_edges.add((u, v))
                            current_count += 1
                            found += 1
                            if found >= 2:
                                break
                    if found >= 2:
                        break
                
                if found >= 2:
                    no_improve = 0
                    if current_count > best_count:
                        best_count = current_count
                        best_edges = set(current_edges)
                        best_adj = list(current_adj)
                else:
                    current_adj = saved_adj
                    current_edges = saved_edges
                    current_count = pre_swap_count
                    no_improve += 1
                    if no_improve > max_no_improve:
                        break
                        
        return best_edges, best_adj, best_count

    best_edges = []
    best_count = 0
    best_adj = [0] * N
    
    # Phase 1: Multiple Inits + Local Search
    for seed in range(100):
        if time.time() >= deadline:
            break
        rng = random.Random(seed * 12345 + 6789)
        edges, adj = random_greedy_init(rng)
        count = len(edges)
        
        if count > best_count:
            best_count = count
            best_edges = list(edges)
            best_adj = list(adj)
            
        if time.time() < deadline - 5:
            new_edges, new_adj, new_count = local_search(edges, adj, rng, max_iters=1000)
            if new_count > best_count:
                best_count = new_count
                best_edges = list(new_edges)
                best_adj = list(new_adj)

    # Phase 2: Deep Polish
    if best_edges and time.time() < deadline - 10:
        adj = [0] * N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        rng = random.Random(99999)
        new_edges, new_adj, new_count = local_search(best_edges, adj, rng, max_iters=50000)
        if new_count > best_count:
            best_count = new_count
            best_edges = list(new_edges)
            best_adj = list(new_adj)

    return [[u, v] for u, v in best_edges]
