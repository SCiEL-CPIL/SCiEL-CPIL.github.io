import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    adj = [0] * n
    edge_set = set()
    
    def popcount(x):
        return bin(x).count('1')
        
    def can_add(u, v):
        if (u, v) in edge_set:
            return False
        common = adj[u] & adj[v]
        mask = common
        while mask:
            lsb = mask & -mask
            w = lsb.bit_length() - 1
            mask ^= lsb
            
            if popcount(adj[v] & adj[w]) > 1:
                return False
            if popcount(adj[u] & adj[w]) > 1:
                return False
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v))
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edge_set.remove((u, v))

    def count_edges():
        return len(edge_set)

    best_edges = 0
    best_adj = [0] * n
    
    def save_state():
        nonlocal best_edges, best_adj
        curr = count_edges()
        if curr > best_edges:
            best_edges = curr
            best_adj = adj[:]

    def greedy_add():
        edges_to_try = list(all_edges)
        random.shuffle(edges_to_try)
        for u, v in edges_to_try:
            if (u, v) not in edge_set:
                if can_add(u, v):
                    add_edge(u, v)
                    
    def local_search():
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in edge_set:
                        if can_add(u, v):
                            add_edge(u, v)
                            changed = True
                            
    for _ in range(50000):
        if time.time() > deadline:
            break
            
        adj = [0] * n
        edge_set = set()
        
        greedy_add()
        local_search()
        save_state()
        
        if time.time() > deadline - 5:
            break
            
        if count_edges() > 0:
            k = random.randint(2, 6)
            edges_list = list(edge_set)
            if len(edges_list) >= k:
                edges_to_remove = random.sample(edges_list, k)
                for u, v in edges_to_remove:
                    remove_edge(u, v)
                
                local_search()
                save_state()

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
