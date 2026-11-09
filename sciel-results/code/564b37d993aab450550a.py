import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = 0
    best_edges = []
    
    # Try n=47 first, then 46
    for target_n in [47, 46]:
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        found = False
        # More restarts since each is faster
        for restart in range(50):
            if time.time() > deadline:
                break
            
            random.seed(restart * 100 + target_n)
            
            # 1. Initialize with a random graph with appropriate density
            # Target avg degree: min_deg_needed + 5 (buffer for removals)
            # p = (min_deg_needed + 5) / (target_n - 1)
            target_avg_deg = min_deg_needed + 5
            p = target_avg_deg / (target_n - 1)
            
            adj_mask = [0] * target_n
            deg = [0] * target_n
            edge_set = set()
            
            # Generate random edges
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < p:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        deg[u] += 1
                        deg[v] += 1
                        edge_set.add((u, v))
                        edge_set.add((v, u))
            
            # 2. Ensure minimum degree is high enough to allow removals
            # We want min(deg) >= min_deg_needed + buffer
            # If not, add random edges
            buffer = 3
            target_min_deg_initial = min_deg_needed + buffer
            
            # Iteratively boost low degree vertices
            # Limit iterations to avoid infinite loop
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                low_vertices = [i for i in range(target_n) if deg[i] < target_min_deg_initial]
                if not low_vertices:
                    break
                
                # Pick a random low vertex
                u = random.choice(low_vertices)
                
                # Find non-neighbors
                non_neighbors = []
                for v in range(target_n):
                    if v != u and not (adj_mask[u] >> v) & 1:
                        non_neighbors.append(v)
                
                if not non_neighbors:
                    # Connected to everyone, can't add more
                    continue
                
                # Pick a random non-neighbor
                v = random.choice(non_neighbors)
                
                # Add edge
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                edge_set.add((u, v))
                edge_set.add((v, u))
            
            # 3. Remove edges to eliminate C4s
            # Greedy: Remove edge with max C4 participation
            # If removal violates degree, try next best or fail
            
            while True:
                if time.time() > deadline:
                    break
                
                # Find edge with max C4 count
                max_c4s = -1
                best_edge_to_remove = None
                candidates = []
                
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if (u, v) not in edge_set:
                            continue
                        
                        common = adj_mask[u] & adj_mask[v]
                        k = bin(common).count('1')
                        if k >= 2:
                            c4_count = k * (k - 1) // 2
                            if c4_count > max_c4s:
                                max_c4s = c4_count
                                best_edge_to_remove = (u, v)
                            
                            # Store candidates if we need to find an alternative
                            # Only store if it's a "good" candidate (reduces C4s)
                            # and check degree later?
                            # To be efficient, just collect top few?
                            # For n=47, list is small.
                            candidates.append((c4_count, u, v))
                
                if best_edge_to_remove is None:
                    break
                
                # Try to remove the best edge
                u, v = best_edge_to_remove
                
                if deg[u] > min_deg_needed and deg[v] > min_deg_needed:
                    # Remove it
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                    deg[u] -= 1
                    deg[v] -= 1
                    edge_set.remove((u, v))
                    edge_set.remove((v, u))
                else:
                    # Best edge is critical. Find another edge that breaks C4s
                    # and is not critical.
                    found_alternative = False
                    # Sort candidates by c4_count descending?
                    # They are already somewhat ordered if we just pick max, 
                    # but we didn't store all. Let's re-scan or use the list.
                    # Let's just re-scan for any valid edge.
                    for c4_count, u2, v2 in candidates:
                        if deg[u2] > min_deg_needed and deg[v2] > min_deg_needed:
                            best_edge_to_remove = (u2, v2)
                            found_alternative = True
                            break
                    
                    if not found_alternative:
                        # Stuck
                        break
                    
                    u, v = best_edge_to_remove
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                    deg[u] -= 1
                    deg[v] -= 1
                    edge_set.remove((u, v))
                    edge_set.remove((v, u))
            
            # 4. Verify
            if min(deg) >= min_deg_needed:
                is_c4_free = True
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        common = adj_mask[u] & adj_mask[v]
                        if common & (common - 1):
                            is_c4_free = False
                            break
                    if not is_c4_free:
                        break
                
                if is_c4_free:
                    final_edges = []
                    for u in range(target_n):
                        for v in range(u + 1, target_n):
                            if (u, v) in edge_set:
                                final_edges.append([u, v])
                    
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = final_edges
                    found = True
                    break # Break restart loop
        
        if best_n == 47:
            break
            
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback
        return {'n': 39, 'edges': []}
