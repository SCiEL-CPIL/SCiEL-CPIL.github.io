import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        target_min_deg = n - 39
        
        for _ in range(2000):
            if time.time() > deadline:
                break
                
            # Use adjacency sets for O(1) lookup
            adj = [set() for _ in range(n)]
            edges = set()
            
            p = target_min_deg / n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i].add(j)
                        adj[j].add(i)
                        edges.add((i, j))
            
            def can_add_edge(u, v):
                if v in adj[u]:
                    return False
                # Check common neighbors
                common_count = 0
                # Iterate over smaller set for efficiency
                if len(adj[u]) < len(adj[v]):
                    for w in adj[u]:
                        if w in adj[v]:
                            common_count += 1
                            if common_count > 1:
                                return False
                else:
                    for w in adj[v]:
                        if w in adj[u]:
                            common_count += 1
                            if common_count > 1:
                                return False
                return True

            def find_c4_edge_to_remove():
                for u in range(n):
                    for v in range(u + 1, n):
                        if v in adj[u]:
                            # Check common neighbors of u and v
                            common = []
                            if len(adj[u]) < len(adj[v]):
                                for w in adj[u]:
                                    if w in adj[v] and w != u and w != v:
                                        common.append(w)
                                        if len(common) > 1:
                                            return (u, common[0])
                            else:
                                for w in adj[v]:
                                    if w in adj[u] and w != u and w != v:
                                        common.append(w)
                                        if len(common) > 1:
                                            return (u, common[0])
                return None

            # Phase 1: Remove edges to make C4-free
            while True:
                if time.time() > deadline:
                    break
                edge_to_remove = find_c4_edge_to_remove()
                if edge_to_remove is None:
                    break
                u, v = edge_to_remove
                if v in adj[u]:
                    adj[u].remove(v)
                    adj[v].remove(u)
                    edges.discard((u, v))
                    edges.discard((v, u))
            
            if time.time() > deadline:
                break

            # Phase 2: Add edges to satisfy min degree
            success = True
            for _ in range(1000):
                if time.time() > deadline:
                    success = False
                    break
                    
                min_deg = n
                min_v = -1
                for v in range(n):
                    if len(adj[v]) < min_deg:
                        min_deg = len(adj[v])
                        min_v = v
                
                if min_deg >= target_min_deg:
                    break
                else:
                    added = False
                    candidates = [u for u in range(n) if u != min_v and u not in adj[min_v]]
                    random.shuffle(candidates)
                    for u in candidates:
                        if can_add_edge(min_v, u):
                            adj[min_v].add(u)
                            adj[u].add(min_v)
                            edges.add((min_v, u))
                            added = True
                            break
                    if not added:
                        success = False
                        break
            
            if success:
                # Final verification
                is_c4_free = True
                for u in range(n):
                    for v in range(u + 1, n):
                        if v in adj[u]:
                            common_count = 0
                            for w in adj[u]:
                                if w in adj[v] and w != u and w != v:
                                    common_count += 1
                                    if common_count > 1:
                                        is_c4_free = False
                                        break
                    if not is_c4_free:
                        break
                
                min_deg = n
                for v in range(n):
                    if len(adj[v]) < min_deg:
                        min_deg = len(adj[v])
                
                if is_c4_free and min_deg >= target_min_deg:
                    return {'n': n, 'edges': [list(e) for e in edges]}
    
    return {'n': 39, 'edges': []}
