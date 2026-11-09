import time
import random

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set(tuple(e) for e in P['edges'])
    base_edges.update((v, u) for u, v in base_edges)
    
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                all_edges.append((u, v))
                
    def is_triangle_free(adj, u, v):
        for w in range(n):
            if w != u and w != v:
                if (u, w) in adj and (v, w) in adj:
                    return False
        return True

    def compute_diameter_and_path(adj):
        max_dist = 0
        u_max, v_max = 0, 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in range(n):
                    if (u, v) in adj and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for i in range(n):
                if dist[i] > max_dist:
                    max_dist = dist[i]
                    u_max = start
                    v_max = i
        return max_dist, (u_max, v_max)

    best_edges = None
    best_count = float('inf')
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    for restart in range(500):
        if time.time() > deadline:
            break
            
        random.seed(restart)
        
        adj = set(base_edges)
        current_edges = set()
        
        # Phase 1: Greedy Add
        while True:
            d, (u, v) = compute_diameter_and_path(adj)
            if d <= 4:
                break
                
            # Find valid edges that reduce diameter
            # Or just pick a random valid edge?
            # Heuristic: Try to add (u, v) if valid.
            added = False
            if is_triangle_free(adj, u, v) and (u, v) not in current_edges:
                adj.add((u, v))
                adj.add((v, u))
                current_edges.add((u, v))
                added = True
                
            if not added:
                # Pick random valid edge
                valid = []
                for e in all_edges:
                    if e in current_edges:
                        continue
                    if is_triangle_free(adj, e[0], e[1]):
                        valid.append(e)
                if not valid:
                    break
                e = random.choice(valid)
                adj.add((e[0], e[1]))
                adj.add((e[1], e[0]))
                current_edges.add(e)
                
        if d <= 4:
            # Phase 2: Greedy Remove
            improved = True
            while improved:
                improved = False
                edges_list = list(current_edges)
                random.shuffle(edges_list)
                for e in edges_list:
                    if time.time() > deadline:
                        break
                    
                    u, v = e
                    adj.remove((u, v))
                    adj.remove((v, u))
                    current_edges.remove(e)
                    
                    d_new, _ = compute_diameter_and_path(adj)
                    if d_new <= 4:
                        improved = True
                    else:
                        adj.add((u, v))
                        adj.add((v, u))
                        current_edges.add(e)
                        
                if time.time() > deadline:
                    break
                    
            if len(current_edges) < best_count:
                best_count = len(current_edges)
                best_edges = list(current_edges)
                
    if best_edges:
        return [[u, v] for u, v in best_edges]
    else:
        return [[u, v] for u in range(n) for v in range(u + 1, n) if (u - v) % 2 and (u, v) not in base_edges]

# EVOLVE-BLOCK-END
