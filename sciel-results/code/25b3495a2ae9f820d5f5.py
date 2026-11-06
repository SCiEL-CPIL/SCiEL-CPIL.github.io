import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n = 40
    
    def time_ok():
        return time.time() < deadline

    def bits(mask):
        while mask:
            bit = mask & (-mask)
            yield bit.bit_length() - 1
            mask ^= bit

    def make_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def would_create_c6(adj, u, v):
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)
        
        mask_a = Nu
        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a
            
            Na_base = adj[a] & ~(1 << u) & ~(1 << v)
            
            mask_d = Nv & ~(1 << a)
            while mask_d:
                bit_d = mask_d & (-mask_d)
                d = bit_d.bit_length() - 1
                mask_d ^= bit_d
                
                B = Na_base & ~(1 << d)
                if B == 0:
                    continue
                
                C = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if C == 0:
                    continue
                
                mask_b = B
                while mask_b:
                    bit_b = mask_b & (-mask_b)
                    b = bit_b.bit_length() - 1
                    mask_b ^= bit_b
                    
                    if adj[b] & C & ~(1 << b):
                        return True
        
        return False

    def greedy_add(adj, edge_set, edges, candidate_edges):
        added_any = True
        while added_any and time_ok():
            added_any = False
            remaining = []
            for u, v in candidate_edges:
                if (min(u,v), max(u,v)) in edge_set:
                    continue
                if would_create_c6(adj, u, v):
                    remaining.append((u, v))
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((min(u,v), max(u,v)))
                    edges.append([u, v])
                    added_any = True
            candidate_edges = remaining

    def local_search(adj, edge_set, edges):
        remaining = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edge_set]
        random.shuffle(remaining)
        greedy_add(adj, edge_set, edges, remaining)
        return len(edges)

    def random_initial_graph(density=0.25):
        edges = []
        edge_set = set()
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < density:
                    edges.append([u, v])
                    edge_set.add((u, v))
        adj = make_adj(edges)
        return adj, edge_set, edges

    best_edges = []
    best_count = 0
    
    # Use fixed seed for reproducibility if needed, but random is fine
    # random.seed(42)
    
    while time_ok():
        adj, edge_set, edges = random_initial_graph(density=0.25)
        
        local_search(adj, edge_set, edges)
        
        current_count = len(edges)
        if current_count > best_count:
            best_count = current_count
            best_edges = edges[:]
            
        # ILS
        for step in range(30):
            if not time_ok():
                break
            
            k = random.randint(5, 10)
            if len(edges) <= k:
                break
                
            # Remove k random edges
            # To make removal fast, we can just pick indices
            indices = random.sample(range(len(edges)), k)
            indices.sort(reverse=True)
            for i in indices:
                u, v = edges[i]
                edge_set.remove((min(u,v), max(u,v)))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                del edges[i]
            
            local_search(adj, edge_set, edges)
            
            current_count = len(edges)
            if current_count > best_count:
                best_count = current_count
                best_edges = edges[:]

    # Final polish
    if best_edges:
        adj = make_adj(best_edges)
        edge_set = set((min(u,v), max(u,v)) for u, v in best_edges)
        local_search(adj, edge_set, best_edges)
        
    return best_edges
