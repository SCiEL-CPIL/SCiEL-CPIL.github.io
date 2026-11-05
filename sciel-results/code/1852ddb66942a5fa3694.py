import random
import time

N = 20

def construct():
    best_edges = []
    best_count = 0
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    def can_add(adj, common, u, v):
        if common[u][v] > 2:
            return False
        
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            if common[u][w] > 1:
                return False
            mask_v ^= lsb
            
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if common[v][w] > 1:
                return False
            mask_u ^= lsb
            
        return True

    def add_edge(adj, common, edges, edge_set, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.append([u, v])
        edge_set.add((u, v))
        
        mask_u = adj[u] & ~(1 << v)
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            common[v][w] += 1
            common[w][v] += 1
            mask_u ^= lsb
            
        mask_v = adj[v] & ~(1 << u)
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            common[u][w] += 1
            common[w][u] += 1
            mask_v ^= lsb

    def remove_edge(adj, common, edges, edge_set, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edge_set.remove((u, v))
        
        idx = edges.index([u, v])
        edges.pop(idx)
        
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            common[v][w] -= 1
            common[w][v] -= 1
            mask_u ^= lsb
            
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            mask_v ^= lsb

    def greedy_construct():
        adj = [0] * N
        common = [[0] * N for _ in range(N)]
        edges = []
        edge_set = set()
        
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if can_add(adj, common, u, v):
                add_edge(adj, common, edges, edge_set, u, v)
                
        return edges, adj, common, edge_set

    def local_search(edges, adj, common, edge_set, max_steps=20000):
        current_edges = edges[:]
        current_adj = adj[:]
        current_common = [row[:] for row in common]
        current_edge_set = set(edge_set)
        current_count = len(current_edges)
        
        steps = 0
        while steps < max_steps:
            steps += 1
            
            u, v = random.choice(all_edges)
            if (u, v) in current_edge_set:
                continue
                
            if can_add(current_adj, current_common, u, v):
                add_edge(current_adj, current_common, current_edges, current_edge_set, u, v)
                current_count += 1
            else:
                if current_count > 0:
                    rem_idx = random.randint(0, current_count - 1)
                    ru, rv = current_edges[rem_idx]
                    
                    remove_edge(current_adj, current_common, current_edges, current_edge_set, ru, rv)
                    current_count -= 1
                    
                    if can_add(current_adj, current_common, u, v):
                        add_edge(current_adj, current_common, current_edges, current_edge_set, u, v)
                        current_count += 1
                    else:
                        add_edge(current_adj, current_common, current_edges, current_edge_set, ru, rv)
                        current_count += 1

        return current_edges

    # Phase 1: Greedy
    best_adj = None
    best_common = None
    best_edge_set = None
    
    for _ in range(200):
        if time.time() - start_time > time_limit:
            break
        edges, adj, common, edge_set = greedy_construct()
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            best_adj = adj
            best_common = common
            best_edge_set = edge_set
            
    if time.time() - start_time > time_limit or best_adj is None:
        return best_edges
        
    # Phase 2: Local Search
    for _ in range(10):
        if time.time() - start_time > time_limit:
            break
            
        # Start from current best
        edges = best_edges[:]
        adj = best_adj[:]
        common = [row[:] for row in best_common]
        edge_set = set(best_edge_set)
        
        remaining_time = time_limit - (time.time() - start_time)
        steps = int(remaining_time * 2000)
        if steps < 1000:
            steps = 1000
            
        improved_edges = local_search(edges, adj, common, edge_set, max_steps=steps)
        
        if len(improved_edges) > best_count:
            best_count = len(improved_edges)
            best_edges = improved_edges[:]
            # Reconstruct adj, common, edge_set for the new best
            best_adj = [0] * N
            best_common = [[0] * N for _ in range(N)]
            best_edge_set = set()
            for e in best_edges:
                u, v = e
                add_edge(best_adj, best_common, [], best_edge_set, u, v)
            # Note: add_edge appends to edges list, so we pass [] and ignore result
            # But we need to update best_adj/common/edge_set correctly.
            # The add_edge function modifies adj/common/edge_set in place.
            # We passed empty list for edges, so it's fine.

    return best_edges
