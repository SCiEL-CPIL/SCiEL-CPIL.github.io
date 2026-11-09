import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = 39
    best_edges = []
    
    # Try n from 47 down to 40
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        found = False
        # More restarts needed for the harder balance task
        for restart in range(2000):
            if time.time() > deadline:
                break
                
            random.seed(restart * 7 + 1)
            
            in_graph = [[False] * target_n for _ in range(target_n)]
            adj = [0] * target_n
            deg = [0] * target_n
            edges = []
            
            def can_add(u, v):
                # Check if adding (u,v) creates a C4
                # A C4 is formed if u and v share a common neighbor w, and w and v share a common neighbor x?
                # No, adding (u,v) creates a C4 if there exists a path of length 3 from u to v.
                # i.e., exists x, y such that u-x, x-y, y-v are edges.
                # This is equivalent to: exists x in N(u) such that N(x) intersects N(v).
                # Or simply: check all common neighbors of u and v? No, that's for triangles.
                # C4: u-a-b-v.
                # If we add (u,v), we check if there is a path u-a-b-v in the current graph.
                # This is equivalent to checking if N(u) and N(v) have a "distance 2" connection?
                # No. The new edge (u,v) creates a cycle u-a-b-v-u if u-a, a-b, b-v exist.
                # So we need to check if any neighbor of u is connected to any neighbor of v.
                # Let N_u = set of neighbors of u.
                # Let N_v = set of neighbors of v.
                # For each a in N_u, check if any neighbor of a is in N_v.
                
                nu = adj[u]
                nv = adj[v]
                
                c = nu
                while c:
                    a_bit = c & -c
                    a = a_bit.bit_length() - 1
                    c ^= a_bit
                    # Check if a is connected to any neighbor of v
                    # Neighbors of a are adj[a]
                    # We need to check if adj[a] & nv != 0
                    if (adj[a] & nv) != 0:
                        return False
                return True

            heap = []
            
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    # Priority: min(deg[u], deg[v]) to balance degrees
                    # Ties broken by random
                    heapq.heappush(heap, (-min(0, 0), random.random(), u, v)) # Placeholder, we push initial priority 0
                    # Actually, we should push all potential edges with priority based on current deg (0)
                    # So all start with -0 = 0.
            
            # Initial push with 0 priority
            heap = []
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    heapq.heappush(heap, (0, random.random(), u, v))
                    
            while heap:
                if time.time() > deadline:
                    break
                    
                pri, _, u, v = heapq.heappop(heap)
                
                if in_graph[u][v]:
                    continue
                
                # The priority stored in heap might be stale if degs changed
                # We recompute priority to check if it's still the "best" available?
                # No, standard lazy deletion: if the key changed, the old entry is stale.
                # But we don't update keys in heap easily.
                # So we just pop and check can_add.
                # The priority order is an approximation.
                # To make it effective, we should push new priorities when degs change.
                
                # Current code just checks can_add. The order is determined by insertion time.
                # This is NOT a valid greedy if we don't re-order based on new degs.
                # We MUST update the heap or use a different structure.
                # Since heapq doesn't support decrease-key, we push new entries.
                
                # Let's verify the current program's logic.
                # It pushes (0, rand, u, v) initially.
                # When an edge is added, it pushes new priorities for affected edges.
                # The `if pri != deg[u] + deg[v]: continue` check in the seed program handles stale entries.
                # I should do the same for my new priority function.
                
                current_prio = -min(deg[u], deg[v])
                if pri != current_prio:
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
