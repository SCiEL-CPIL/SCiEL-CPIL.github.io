import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_result = {'n': 39, 'edges': []}
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        min_deg = n - 39
        if min_deg < 0:
            min_deg = 0
        
        if min_deg == 0:
            return {'n': n, 'edges': []}
        
        # Time budget per n value
        time_per_n = 15.0
        n_deadline = start_time + time_per_n * (47 - n + 1)
        
        max_restarts = 2000
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            adj = [0] * n
            degrees = [0] * n
            edges = []
            
            def can_add(u, v):
                if u == v:
                    return False
                if (adj[u] >> v) & 1:
                    return False
                # Check if adding (u,v) creates a C4
                # A C4 u-w-x-v exists if ∃w∈N(u) with ∃x∈N(w)∩N(v)
                m = adj[u]
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    if adj[w] & adj[v]:
                        return False
                    m ^= lsb
                return True
            
            # Randomized greedy construction
            # Randomize vertex order for diversity
            vertex_order = list(range(n))
            random.shuffle(vertex_order)
            
            added_any = True
            while added_any and time.time() < deadline:
                added_any = False
                
                # Find vertices below target degree
                low_verts = [i for i in range(n) if degrees[i] < min_deg]
                
                if not low_verts:
                    break
                
                # Most constrained first
                u = min(low_verts, key=lambda x: n - 1 - degrees[x])
                
                # Try non-neighbors, with randomization
                non_neighbors = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
                if not non_neighbors:
                    break
                
                # Sort by degree with some randomness
                random.shuffle(non_neighbors)
                non_neighbors.sort(key=lambda v: degrees[v] + random.random() * 0.1)
                
                for v in non_neighbors:
                    if can_add(u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degrees[u] += 1
                        degrees[v] += 1
                        edges.append([u, v])
                        added_any = True
                        break
            
            # Check if we met the requirement
            if min(degrees) >= min_deg:
                # Verify C4-free
                c4_free = True
                for i in range(n):
                    if not c4_free:
                        break
                    for j in range(i + 1, n):
                        if not c4_free:
                            break
                        if (adj[i] >> j) & 1:
                            common = adj[i] & adj[j]
                            if common & (common - 1):  # More than one common neighbor
                                c4_free = False
                                break
                if c4_free:
                    return {'n': n, 'edges': edges}
            
            # Polish: try to fix deficient vertices with more aggressive search
            if time.time() < deadline:
                for ls_iter in range(100):
                    if time.time() > deadline:
                        break
                    
                    deficient = [i for i in range(n) if degrees[i] < min_deg]
                    if not deficient:
                        break
                    
                    u = min(deficient, key=lambda x: n - 1 - degrees[x])
                    non_neighbors = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
                    random.shuffle(non_neighbors)
                    non_neighbors.sort(key=lambda v: degrees[v])
                    
                    improved = False
                    for v in non_neighbors:
                        if can_add(u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            degrees[u] += 1
                            degrees[v] += 1
                            edges.append([u, v])
                            improved = True
                            break
                    
                    if not improved:
                        break
                
                if min(degrees) >= min_deg:
                    c4_free = True
                    for i in range(n):
                        if not c4_free:
                            break
                        for j in range(i + 1, n):
                            if not c4_free:
                                break
                            if (adj[i] >> j) & 1:
                                common = adj[i] & adj[j]
                                if common & (common - 1):
                                    c4_free = False
                                    break
                    if c4_free:
                        return {'n': n, 'edges': edges}
        
        # If we couldn't find a solution for this n, continue to smaller n
    
    return best_result
