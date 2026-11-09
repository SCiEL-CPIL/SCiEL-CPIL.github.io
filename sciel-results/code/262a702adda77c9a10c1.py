import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_n = 39
    best_edges = []
    
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break
            
        target_min_deg = n - 39
        
        found = False
        # Increase attempts for smaller n if 46/47 fail
        attempts = 100 if n >= 46 else 50
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
                
            edges = set()
            adj = {i: set() for i in range(n)}
            
            p = 0.22 if n == 46 else 0.20
            if n == 47: p = 0.19
            
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < p:
                        edges.add((u, v))
                        adj[u].add(v)
                        adj[v].add(u)
            
            c4_found = True
            iterations = 0
            max_iterations = 1000
            
            while c4_found and iterations < max_iterations:
                iterations += 1
                if time.time() > deadline:
                    break
                
                c4_found = False
                for (u, v) in list(edges):
                    common = adj[u] & adj[v]
                    if len(common) >= 2:
                        c4_found = True
                        common_list = list(common)
                        w1 = common_list[0]
                        w2 = common_list[1]
                        
                        # Cycle: u-w1-v-w2-u
                        cycle_edges = [(u, w1), (w1, v), (v, w2), (w2, u)]
                        
                        valid_edges = []
                        for a, b in cycle_edges:
                            if a > b: a, b = b, a
                            if (a, b) in edges:
                                valid_edges.append((a, b))
                        
                        if not valid_edges:
                            continue # Should not happen
                        
                        # Score edges for removal
                        # 1. Prefer edges where both degrees > target
                        # 2. Then max degree sum
                        best_edge = None
                        best_score = -1
                        
                        for a, b in valid_edges:
                            da = len(adj[a])
                            db = len(adj[b])
                            min_d = min(da, db)
                            max_d = max(da, db)
                            sum_d = da + db
                            
                            # Heuristic score
                            # If min_d > target, it's "safe"
                            # We want to remove unsafe edges? No, we want to keep min degree high.
                            # So we should remove edges that are NOT critical for min degree.
                            # i.e., remove edges where min_d is large.
                            # If min_d == target, removing it drops degree to target-1. Bad.
                            # So we prefer edges with min_d > target.
                            
                            score = 0
                            if min_d > target_min_deg:
                                score += 1000
                            score += sum_d
                            
                            if score > best_score:
                                best_score = score
                                best_edge = (a, b)
                        
                        if best_edge:
                            a, b = best_edge
                            edges.remove((a, b))
                            adj[a].remove(b)
                            adj[b].remove(a)
                            break # Restart scan
                        
                # End for edges
                
            # End while C4
            
            if not c4_found:
                min_deg = min(len(adj[i]) for i in range(n))
                if min_deg >= target_min_deg:
                    found = True
                    best_n = n
                    edge_list = []
                    for u, v in edges:
                        if u < v:
                            edge_list.append([u, v])
                        else:
                            edge_list.append([v, u])
                    best_edges = edge_list
                    break
        
        if found:
            break
        
    return {'n': best_n, 'edges': best_edges}
