import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_n = 0
    best_edges = []
    
    for target_n in [47, 46]:
        if time.time() > deadline:
            break
            
        min_deg_req = target_n - 39
        
        found = False
        for attempt in range(5000):
            if time.time() > deadline:
                break
            
            p = 0.2
            adj = [set() for _ in range(target_n)]
            
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < p:
                        adj[u].add(v)
                        adj[v].add(u)
            
            if min(len(adj[i]) for i in range(target_n)) < min_deg_req:
                continue
                
            common = [[0] * target_n for _ in range(target_n)]
            bad_pairs = set()
            
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    c = len(adj[u] & adj[v])
                    common[u][v] = c
                    common[v][u] = c
                    if c >= 2:
                        bad_pairs.add((u, v))
                        
            max_iter = 10000
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                    
                if not bad_pairs:
                    if min(len(adj[i]) for i in range(target_n)) >= min_deg_req:
                        found = True
                    break
                    
                u, v = random.choice(list(bad_pairs))
                # Ensure u < v for consistency? Not strictly needed if we handle sets.
                # But bad_pairs stores (min, max).
                
                common_neighbors = list(adj[u] & adj[v])
                if len(common_neighbors) < 2:
                    # Should not happen if in bad_pairs, but maybe stale?
                    bad_pairs.discard((u, v))
                    continue
                    
                w, x = common_neighbors[0], common_neighbors[1]
                
                # Remove edge
                edge_to_remove = random.choice([(u,w), (w,v), (v,x), (x,u)])
                a, b = edge_to_remove
                if a > b: a, b = b, a
                
                adj[a].discard(b)
                adj[b].discard(a)
                
                # Update common
                # For each c in N(a) (old), if c in N(b), common[a][c] -= 1
                # We iterate over current N(a)
                for c in list(adj[a]):
                    if b in adj[c]:
                        old_val = common[a][c]
                        common[a][c] -= 1
                        common[c][a] -= 1
                        new_val = common[a][c]
                        
                        key = (a, c) if a < c else (c, a)
                        if old_val >= 2 and new_val < 2:
                            bad_pairs.discard(key)
                        elif old_val < 2 and new_val >= 2:
                            bad_pairs.add(key)
                            
                for c in list(adj[b]):
                    if a in adj[c]:
                        old_val = common[b][c]
                        common[b][c] -= 1
                        common[c][b] -= 1
                        new_val = common[b][c]
                        
                        key = (b, c) if b < c else (c, b)
                        if old_val >= 2 and new_val < 2:
                            bad_pairs.discard(key)
                        elif old_val < 2 and new_val >= 2:
                            bad_pairs.add(key)
                            
                # Update common[a][b]
                # As discussed, common[a][b] is unchanged by removing (a,b).
                # But wait, if we removed (a,b), is (a,b) still a "pair"?
                # Yes, we track common neighbors for all pairs.
                # The value common[a][b] is |N(a) \cap N(b)|.
                # N(a) and N(b) changed (lost each other), but their intersection (other vertices) didn't change.
                # So common[a][b] is correct.
                
            if found:
                edges = []
                for i in range(target_n):
                    for j in adj[i]:
                        if i < j:
                            edges.append([i, j])
                
                if target_n > best_n:
                    best_n = target_n
                    best_edges = edges
                if target_n == 47:
                    return {'n': best_n, 'edges': best_edges}

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    return {'n': 39, 'edges': []}
