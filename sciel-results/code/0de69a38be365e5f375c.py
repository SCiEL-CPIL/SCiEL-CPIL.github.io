import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def can_add_edge(adj, u, v):
        return (adj[u] & adj[v]) == 0

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def calc_alpha(adj, n):
        best = 0
        def dfs(candidates, current_size):
            nonlocal best
            if candidates == 0:
                if current_size > best:
                    best = current_size
                return
            
            if current_size + bin(candidates).count("1") <= best:
                return

            v = (candidates & -candidates).bit_length() - 1
            
            new_candidates = candidates & ~adj[v] & ~(1 << v)
            dfs(new_candidates, current_size + 1)
            
            new_candidates = candidates & ~(1 << v)
            dfs(new_candidates, current_size)

        all_vertices = (1 << n) - 1
        dfs(all_vertices, 0)
        return best

    def run_local_search(n, time_limit):
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if can_add_edge(adj, u, v):
                add_edge(adj, u, v)
        
        current_alpha = calc_alpha(adj, n)
        best_adj = adj[:]
        best_alpha = current_alpha
        
        if current_alpha < 10:
            return best_adj, True
        
        t_start = time.time()
        
        while time.time() - t_start < time_limit:
            if time.time() > deadline:
                break
                
            if random.random() < 0.8:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v or (adj[u] & (1 << v)):
                    continue
                if not can_add_edge(adj, u, v):
                    continue
                
                add_edge(adj, u, v)
                new_alpha = calc_alpha(adj, n)
                
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    best_adj = adj[:]
                    best_alpha = current_alpha
                    if current_alpha < 10:
                        return best_adj, True
                else:
                    remove_edge(adj, u, v)
            else:
                found = False
                for _ in range(10):
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v and (adj[u] & (1 << v)):
                        found = True
                        break
                if not found:
                    continue
                
                remove_edge(adj, u, v)
                new_alpha = calc_alpha(adj, n)
                
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    best_adj = adj[:]
                    best_alpha = current_alpha
                    if current_alpha < 10:
                        return best_adj, True
                else:
                    if random.random() < 0.1:
                        pass
                    else:
                        add_edge(adj, u, v)
            
        return best_adj, best_alpha < 10

    for n in range(40, 26, -1):
        if time.time() > deadline - 5:
            break
        time_left = deadline - time.time()
        if time_left < 2:
            break
            
        # Try 3 restarts for each n
        found = False
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            search_time = min(3, (deadline - time.time()) / 3)
            if search_time < 0.5:
                break
            adj, valid = run_local_search(n, search_time)
            if valid:
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}

    n = 26
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
