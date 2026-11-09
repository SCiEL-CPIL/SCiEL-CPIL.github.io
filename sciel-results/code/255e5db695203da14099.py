import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110
    
    best_edges = 0
    best_adj = None
    
    def popcount(x):
        return x.bit_count()
        
    def edge_count(adj):
        return sum(popcount(a) for a in adj) // 2
        
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def can_add(adj, u, v):
        if adj[u] & (1 << v):
            return True # Already exists
            
        # Check pair u, v
        if popcount(adj[u] & adj[v]) > 2:
            return False
            
        # Check interactions with other nodes
        for w in range(n):
            if w == u or w == v:
                continue
            # If w is connected to v, adding v to u increases common neighbors of (u, w)
            if adj[v] & (1 << w):
                if popcount(adj[u] & adj[w]) > 1:
                    return False
            # If w is connected to u, adding u to v increases common neighbors of (v, w)
            if adj[u] & (1 << w):
                if popcount(adj[v] & adj[w]) > 1:
                    return False
        return True
        
    def try_greedy_add(adj):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] & (1 << v)):
                        if can_add(adj, u, v):
                            add_edge(adj, u, v)
                            changed = True
                            
    def repair(adj):
        while True:
            violation_found = False
            for i in range(n):
                for j in range(i+1, n):
                    common = adj[i] & adj[j]
                    if popcount(common) > 2:
                        violation_found = True
                        w_list = [k for k in range(n) if common & (1 << k)]
                        w = random.choice(w_list)
                        if random.random() < 0.5:
                            remove_edge(adj, i, w)
                        else:
                            remove_edge(adj, j, w)
                        break
                if violation_found:
                    break
            if not violation_found:
                break
                
    def local_search(adj):
        repair(adj)
        try_greedy_add(adj)
        
        current_count = edge_count(adj)
        
        for _ in range(100):
            if time.time() > deadline:
                break
                
            improvement = False
            target_u, target_v, target_w = -1, -1, -1
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] & (1 << v)):
                        common = adj[u] & adj[v]
                        if popcount(common) == 3:
                            w_list = [k for k in range(n) if common & (1 << k)]
                            w = random.choice(w_list)
                            target_u, target_v, target_w = u, v, w
                            break
                if target_u != -1:
                    break
                    
            if target_u == -1:
                edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u] & (1 << v):
                            edges.append((u, v))
                if not edges:
                    break
                u, v = random.choice(edges)
                adj_backup = adj.copy()
                remove_edge(adj, u, v)
                try_greedy_add(adj)
                new_count = edge_count(adj)
                if new_count > current_count:
                    current_count = new_count
                    improvement = True
                elif new_count == current_count and random.random() < 0.5:
                    current_count = new_count
                    improvement = True
                else:
                    adj = adj_backup
            else:
                u, v, w = target_u, target_v, target_w
                if adj[u] & (1 << w):
                    adj_backup = adj.copy()
                    remove_edge(adj, u, w)
                    add_edge(adj, u, v)
                    try_greedy_add(adj)
                    new_count = edge_count(adj)
                    if new_count > current_count:
                        current_count = new_count
                        improvement = True
                    elif new_count == current_count and random.random() < 0.5:
                        current_count = new_count
                        improvement = True
                    else:
                        adj = adj_backup

        return current_count, adj

    for seed in range(1000):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        p = random.uniform(0.3, 0.7)
        adj = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    add_edge(adj, u, v)
                    
        count, adj_final = local_search(adj)
        
        if count > best_edges:
            best_edges = count
            best_adj = adj_final
            
    edges = []
    if best_adj:
        for u in range(n):
            for v in range(u+1, n):
                if best_adj[u] & (1 << v):
                    edges.append([u, v])
    else:
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
            
    return edges
