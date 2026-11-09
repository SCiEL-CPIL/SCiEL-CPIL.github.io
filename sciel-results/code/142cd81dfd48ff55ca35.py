import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = 39
    best_edges = []
    
    def can_add_fast(adj, u, v):
        """Check if adding edge (u,v) creates a C4 using bit operations."""
        # C4 exists if u and v share a common neighbor w, and w shares another common neighbor
        # Actually: adding (u,v) creates C4 iff there exist w1 != w2 such that
        # w1 is neighbor of u and v, OR w1 neighbor of u and w2 neighbor of v with w1-w2 edge
        # More precisely: C4 u-a-v-b-u exists iff there are a,b with a~u, a~v, b~u, b~v, a!=b
        # AND a~b is NOT required... wait no.
        # C4 is a cycle of length 4: u-a-v-b-u
        # So we need: a is neighbor of u and v, b is neighbor of u and v, a != b, and a-b is an edge?
        # No! The cycle is u-a-v-b-u, so edges are (u,a),(a,v),(v,b),(b,u).
        # So a must be common neighbor of u and v, b must be common neighbor of u and v, and a != b.
        # We do NOT need a-b to be an edge.
        # So: adding (u,v) creates C4 iff |common_neighbors(u,v)| >= 2
        # Wait, that's not right either. Let me think again.
        # If we add edge (u,v), a C4 through this edge would be u-v-w-x-u where w~v, x~u, and w~x.
        # So: exists w in N(v), x in N(u) such that w~x and w != x, and w != u, x != v.
        # Equivalently: the bipartite graph between N(u) and N(v) has an edge.
        
        nu = adj[u]
        nv = adj[v]
        # Check if any neighbor of u is adjacent to any neighbor of v
        # i.e., is there w in N(u) and z in N(v) with w~z?
        # We can check: for each w in N(u), is adj[w] & nv != 0?
        c = nu
        while c:
            a_bit = c & -c
            a = a_bit.bit_length() - 1
            c ^= a_bit
            if (adj[a] & nv) != 0:
                return False
        return True

    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        found = False
        for restart in range(200):
            if time.time() > deadline:
                break
                
            random.seed(restart * 7919 + 13 * target_n)
            
            adj = [0] * target_n
            deg = [0] * target_n
            edges = []
            in_graph = [[False] * target_n for _ in range(target_n)]
            
            # Strategy: prioritize edges that connect low-degree vertices
            # Use a priority = deg[u] + deg[v], but with random tie-breaking
            # Also, bias towards vertices that are most deficient
            
            heap = []
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    heapq.heappush(heap, (0, random.random(), u, v))
            
            while heap:
                if time.time() > deadline - 5:
                    break
                    
                pri, _, u, v = heapq.heappop(heap)
                
                if in_graph[u][v]:
                    continue
                
                if pri != deg[u] + deg[v]:
                    continue
                
                if can_add_fast(adj, u, v):
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    deg[u] += 1
                    deg[v] += 1
                    edges.append([u, v])
                    in_graph[u][v] = True
                    in_graph[v][u] = True
                    
                    for w in range(target_n):
                        if w == u or w == v:
                            continue
                        if not in_graph[u][w]:
                            heapq.heappush(heap, (deg[u] + deg[w], random.random(), u, w))
                        if not in_graph[v][w]:
                            heapq.heappush(heap, (deg[v] + deg[w], random.random(), v, w))
            
            # Check if we achieved the target
            if min(deg) >= min_deg_needed:
                if target_n > best_n:
                    best_n = target_n
                    best_edges = edges[:]
                found = True
                break
        
        if best_n >= target_n:
            break
    
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
