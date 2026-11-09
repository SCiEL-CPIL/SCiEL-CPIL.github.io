import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n = 20
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_k33_free(adj):
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                common_ij = ai & adj[j]
                if common_ij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    if (common_ij & adj[k]).bit_count() >= 3:
                        return False
        return True

    def edge_count(adj):
        return sum(x.bit_count() for x in adj) // 2

    def get_edges(adj):
        return [[u, v] for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]

    def greedy_add_edges(adj, edge_order):
        for u, v in edge_order:
            if (adj[u] >> v) & 1:
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not is_k33_free(adj):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return adj

    def local_search(adj, max_iters=50):
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > deadline - 1.0:
                break
                
            missing = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] >> v) & 1]
            if missing:
                random.shuffle(missing)
                for u, v in missing:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if is_k33_free(adj):
                        improved = True
                        break
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
            
            if not improved:
                existing = [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]
                if existing and missing:
                    random.shuffle(existing)
                    random.shuffle(missing)
                    found_swap = False
                    attempts = 0
                    for u1, v1 in existing[:25]:
                        if found_swap or attempts > 150:
                            break
                        for u2, v2 in missing[:25]:
                            if found_swap or attempts > 150:
                                break
                            attempts += 1
                            if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                                continue
                            
                            adj[u1] &= ~(1 << v1)
                            adj[v1] &= ~(1 << u1)
                            adj[u2] |= (1 << v2)
                            adj[v2] |= (1 << u2)
                            
                            if is_k33_free(adj):
                                found_swap = True
                                improved = True
                                break
                            else:
                                adj[u1] |= (1 << v1)
                                adj[v1] |= (1 << u1)
                                adj[u2] &= ~(1 << v2)
                                adj[v2] &= ~(1 << u2)
                    
                    if found_swap:
                        break
                    
                    if not improved and len(existing) >= 2 and len(missing) >= 2:
                        for _ in range(80):
                            rem1, rem2 = random.sample(existing, 2)
                            add1, add2 = random.sample(missing, 2)
                            edges_set = {(min(a[0],a[1]), max(a[0],a[1])) for a in [rem1, rem2, add1, add2]}
                            if len(edges_set) != 4:
                                continue
                            
                            for u,v in [rem1, rem2]:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                            for u,v in [add1, add2]:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                            
                            if is_k33_free(adj):
                                improved = True
                                break
                            else:
                                for u,v in [rem1, rem2]:
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
                                for u,v in [add1, add2]:
                                    adj[u] &= ~(1 << v)
                                    adj[v] &= ~(1 << u)
                        
                        if not improved and len(existing) >= 3 and len(missing) >= 3:
                            for _ in range(30):
                                rem = random.sample(existing, 3)
                                add = random.sample(missing, 3)
                                edges_set = {(min(a[0],a[1]), max(a[0],a[1])) for a in rem + add}
                                if len(edges_set) != 6:
                                    continue
                                
                                for u,v in rem:
                                    adj[u] &= ~(1 << v)
                                    adj[v] &= ~(1 << u)
                                for u,v in add:
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
                                
                                if is_k33_free(adj):
                                    improved = True
                                    break
                                else:
                                    for u,v in rem:
                                        adj[u] |= (1 << v)
                                        adj[v] |= (1 << u)
                                    for u,v in add:
                                        adj[u] &= ~(1 << v)
                                        adj[v] &= ~(1 << u)

        return adj

    best_edges = 0
    best_adj = None

    while time.time() < deadline - 2.0:
        strategy = random.random()
        
        if strategy < 0.5:
            random_order = all_edges[:]
            random.shuffle(random_order)
            adj = [0] * n
            adj = greedy_add_edges(adj, random_order)
        else:
            adj = [0] * n
            for u, v in all_edges:
                if random.random() < 0.25:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            for u, v in all_edges:
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    if is_k33_free(adj):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        adj = local_search(adj, max_iters=12)
        
        ec = edge_count(adj)
        if ec > best_edges:
            best_edges = ec
            best_adj = adj[:]
        
        if best_edges > 75 and time.time() < deadline - 8.0:
            adj_copy = best_adj[:]
            adj_copy = local_search(adj_copy, max_iters=8)
            ec = edge_count(adj_copy)
            if ec > best_edges:
                best_edges = ec
                best_adj = adj_copy[:]

    if best_adj:
        return get_edges(best_adj)
    else:
        return []
