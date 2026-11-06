import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def try_greedy(n, seed):
        random.seed(seed)
        masks = [0] * n
        
        # Generate all edges
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                edges.append((u, v))
        
        random.shuffle(edges)
        
        for u, v in edges:
            if (masks[u] & masks[v]) == 0:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                
        min_deg = min(masks[i].bit_count() for i in range(n))
        if min_deg >= n - 39:
            return masks
        return None

    best_n = 39
    best_edges = []
    
    # Try n=47
    if time.time() < deadline - 5:
        for seed in range(100000):
            if time.time() > deadline - 5:
                break
            masks = try_greedy(47, seed)
            if masks is not None:
                best_n = 47
                edges = []
                for u in range(47):
                    for v in range(u + 1, 47):
                        if masks[u] & (1 << v):
                            edges.append([u, v])
                best_edges = edges
                break

    if best_n < 47 and time.time() < deadline - 5:
        for seed in range(100000):
            if time.time() > deadline - 5:
                break
            masks = try_greedy(46, seed)
            if masks is not None:
                best_n = 46
                edges = []
                for u in range(46):
                    for v in range(u + 1, 46):
                        if masks[u] & (1 << v):
                            edges.append([u, v])
                best_edges = edges
                break

    if best_n > 39:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
