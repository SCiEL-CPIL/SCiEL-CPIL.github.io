import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = 120.0  # Safety margin

    best_edges = 0
    best_adj = [0] * n

    def is_safe(u, v, adj):
        N_u = adj[u]
        N_v = adj[v]
        
        x1_mask = N_u
        while x1_mask:
            lsb = x1_mask & -x1_mask
            x1 = lsb.bit_length() - 1
            x1_mask ^= lsb
            
            x3_mask = N_v
            while x3_mask:
                lsb3 = x3_mask & -x3_mask
                x3 = lsb3.bit_length() - 1
                x3_mask ^= lsb3
                
                if x1 == x3:
                    continue
                
                if adj[x1] & adj[x3]:
                    return False
        return True

    def find_blocking_path(u, v, adj):
        N_u = adj[u]
        N_v = adj[v]
        
        x1_mask = N_u
        while x1_mask:
            lsb = x1_mask & -x1_mask
            x1 = lsb.bit_length() - 1
            x1_mask ^= lsb
            
            x3_mask = N_v
            while x3_mask:
                lsb3 = x3_mask & -x3_mask
                x3 = lsb3.bit_length() - 1
                x3_mask ^= lsb3
                
                if x1 == x3:
                    continue
                
                common = adj[x1] & adj[x3]
                if common:
                    lsb2 = common & -common
                    x2 = lsb2.bit_length() - 1
                    return (u, x1, x2, x3, v)
        return None

    def greedy_add(adj):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] >> v) & 1:
                        if is_safe(u, v, adj):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            changed = True

    def count_edges(adj):
        c = 0
        for u in range(n):
            c += bin(adj[u]).count('1')
        return c // 2

    def local_search(adj, time_limit):
        while time.time() < time_limit:
            blocked_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] >> v) & 1:
                        if not is_safe(u, v, adj):
                            blocked_edges.append((u, v))
            
            if not blocked_edges:
                break
            
            u, v = random.choice(blocked_edges)
            path = find_blocking_path(u, v, adj)
            if not path:
                break
            
            old_adj = adj[:]
            old_count = count_edges(adj)
            
            edge_idx = random.randint(0, 3)
            a, b = path[edge_idx], path[edge_idx + 1]
            
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            greedy_add(adj)
            
            new_count = count_edges(adj)
            
            if new_count > old_count:
                continue
            else:
                adj = old_adj
        
        return adj

    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        if restarts == 1:
            adj = [0] * n
            greedy_add(adj)
        else:
            # Perturb best
            adj = best_adj[:]
            # Remove 3-10 random edges
            k = random.randint(3, 10)
            for _ in range(k):
                while True:
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v and (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        break
            greedy_add(adj)
        
        adj = local_search(adj, deadline - 1)
        
        current_count = count_edges(adj)
        if current_count > best_edges:
            best_edges = current_count
            best_adj = adj[:]

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
