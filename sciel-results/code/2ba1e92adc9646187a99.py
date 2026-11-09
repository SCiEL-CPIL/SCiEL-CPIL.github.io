import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_n = 39
    best_edges = []
    
    # Try n from 47 down to 40
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        found = False
        # More restarts for harder instances
        max_restarts = 3000 if target_n <= 44 else 1500
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            random.seed(restart * 13 + target_n * 7 + 42)
            
            adj = [0] * target_n
            deg = [0] * target_n
            edges = []
            
            # Precompute all potential edges
            all_edges = [(u, v) for u in range(target_n) for v in range(u + 1, target_n)]
            
            def can_add(u, v):
                # Check if adding (u,v) creates a C4
                # A C4 is formed if there exists a path u-a-b-v in the current graph
                # i.e., exists a in N(u) such that a is connected to some b in N(v)
                nu = adj[u]
                nv = adj[v]
                
                c = nu
                while c:
                    a_bit = c & -c
                    a = a_bit.bit_length() - 1
                    c ^= a_bit
                    # Check if any neighbor of a is in N(v)
                    if (adj[a] & nv) != 0:
                        return False
                return True
            
            # Greedy edge addition with degree balancing
            # Use a priority queue: prioritize edges that connect lower-degree vertices
            import heapq
            heap = []
            for u, v in all_edges:
                # Priority: negative min(deg[u], deg[v]) to prefer balancing
                # Tie-break with random for diversity
                heapq.heappush(heap, (-min(deg[u], deg[v]), random.random(), u, v))
            
            # Track which edges are already in the graph
            in_graph = [[False] * target_n for _ in range(target_n)]
            
            while heap:
                if time.time() > deadline:
                    break
                    
                pri, _, u, v = heapq.heappop(heap)
                
                if in_graph[u][v]:
                    continue
                
                # Check if priority is still valid (degrees may have changed)
                current_prio = -min(deg[u], deg[v])
                if pri != current_prio:
                    # Stale entry, skip
                    continue
                
                if can_add(u, v):
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    deg[u] += 1
                    deg[v] += 1
                    edges.append([u, v])
                    in_graph[u][v] = True
                    in_graph[v][u] = True
                    
                    # Update priorities for edges involving u or v
                    for w in range(target_n):
                        if w == u or w == v:
                            continue
                        if not in_graph[u][w]:
                            heapq.heappush(heap, (-min(deg[u], deg[w]), random.random(), u, w))
                        if not in_graph[v][w]:
                            heapq.heappush(heap, (-min(deg[v], deg[w]), random.random(), v, w))
                    
                    # Early termination if requirement met
                    if min(deg) >= min_deg_needed:
                        break
            
            if min(deg) >= min_deg_needed:
                if target_n > best_n:
                    best_n = target_n
                    best_edges = edges
                found = True
                break
        
        if best_n >= target_n:
            break
    
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
