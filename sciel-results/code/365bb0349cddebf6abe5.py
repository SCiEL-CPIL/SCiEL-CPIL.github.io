import time
import random

def construct():
    n = 20
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    def bit_count(x):
        return bin(x).count('1')
        
    best_edges = 0
    best_adj = None
    
    for seed in range(100): # Try multiple restarts
        if time.time() - start_time > time_limit - 5:
            break
            
        random.seed(seed)
        adj = [0] * n
        
        def add_edge(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        def remove_edge(u, v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
        def has_edge(u, v):
            return (adj[u] >> v) & 1
            
        def count_edges():
            return sum(bit_count(adj[i]) for i in range(n)) // 2
            
        def is_valid_add(u, v):
            nu = adj[u]
            nv = adj[v]
            
            # Check 1: u in U, v in V
            # Pairs u1, u2 in N(v)
            nv_list = [i for i in range(n) if (nv >> i) & 1]
            for i in range(len(nv_list)):
                u1 = nv_list[i]
                for j in range(i+1, len(nv_list)):
                    u2 = nv_list[j]
                    c = adj[u1] & adj[u2] & nu
                    c &= ~(1 << v)
                    if bit_count(c) >= 2:
                        return False
                        
            # Check 2: u in V, v in U
            # Pairs v1, v2 in N(u)
            nu_list = [i for i in range(n) if (nu >> i) & 1]
            for i in range(len(nu_list)):
                v1 = nu_list[i]
                for j in range(i+1, len(nu_list)):
                    v2 = nu_list[j]
                    c = adj[v1] & adj[v2] & nv
                    c &= ~(1 << u)
                    if bit_count(c) >= 2:
                        return False
                        
            return True

        # Initialization: Greedy add
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if is_valid_add(u, v):
                add_edge(u, v)
                
        # Local Search
        stuck_count = 0
        while time.time() - start_time < time_limit - 5:
            improved = False
            
            # Try to add edges
            for u, v in edges:
                if not has_edge(u, v):
                    if is_valid_add(u, v):
                        add_edge(u, v)
                        improved = True
            
            if improved:
                stuck_count = 0
                if count_edges() > best_edges:
                    best_edges = count_edges()
                    best_adj = adj[:]
            else:
                stuck_count += 1
                if stuck_count > 100:
                    break
                    
                # Try a swap
                current_edges = [(u, v) for u, v in edges if has_edge(u, v)]
                if not current_edges:
                    break
                    
                e1 = random.choice(current_edges)
                u1, v1 = e1
                remove_edge(u1, v1)
                
                # Try to add a different edge
                added = False
                for u, v in edges:
                    if not has_edge(u, v) and (u, v) != (u1, v1) and (u, v) != (v1, u1):
                        if is_valid_add(u, v):
                            add_edge(u, v)
                            added = True
                            break
                            
                if not added:
                    add_edge(u1, v1)
                else:
                    # If we added an edge, we might be able to add more in next iteration
                    # But we maintain same edge count for now.
                    # Check if we improved global best? No, same count.
                    # But structure changed.
                    pass

    # Reconstruct best_adj
    if best_adj is None:
        return []
        
    res = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                res.append([i, j])
    return res
