import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = 39
    best_edges = []
    
    # We try to find a graph for n=47, 46, ...
    # Known R(C4, K1,39) > n implies existence of C4-free graph with min deg >= n-39
    # We maximize n.
    
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        found = False
        # Number of restarts: We can afford many because the check is fast with bitmasks
        # But heap operations are O(log E). E ~ n^2/2 ~ 1100.
        # Let's try 200 restarts for large n, more for smaller n if time permits.
        num_restarts = 100 if target_n > 44 else 300
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            random.seed(restart * 7919 + target_n * 131)
            
            adj = [0] * target_n
            deg = [0] * target_n
            edges = []
            
            # in_graph is used to avoid adding duplicate edges to the heap or processing them
            # We use a set of tuples for O(1) lookup
            in_graph = set()
            
            def can_add(u, v):
                # Check if adding (u,v) creates a C4
                # This happens if there exists w in N(u) and x in N(v) such that (w,x) is an edge.
                # i.e., for any w in N(u), adj[w] & adj[v] != 0
                nu = adj[u]
                nv = adj[v]
                # Iterate through neighbors of u
                c = nu
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    # Check if w is connected to any neighbor of v
                    if adj[w] & nv:
                        return False
                    c ^= lsb
                return True

            # Initialize heap with all potential edges
            # Priority: deg[u] + deg[v] (initially 0), then random tie-breaker
            heap = []
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    heapq.heappush(heap, (0, random.random(), u, v))
            
            success = True
            
            while heap:
                if time.time() > deadline:
                    success = False
                    break
                    
                pri, _, u, v = heapq.heappop(heap)
                
                # If this edge is already in the graph, skip
                if (u, v) in in_graph:
                    continue
                
                # If the priority is stale (degrees have changed), we could re-push,
                # but since we only increase degrees, the current priority is a lower bound.
                # However, if we popped an edge with low priority but degrees have increased,
                # it might still be a good candidate. But if we are looking for min degree,
                # we want to prioritize low degree sums.
                # The heap property ensures we pop the smallest priority.
                # Since degrees only increase, the priority of any edge in the heap is <= its current deg sum.
                # So if we pop (u,v), its current deg sum is >= pri.
                # This is fine. We just process it.
                
                if can_add(u, v):
                    # Add edge
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    deg[u] += 1
                    deg[v] += 1
                    edges.append([u, v])
                    in_graph.add((u, v))
                    
                    # Update heap with new priorities for edges incident to u and v
                    # We push new entries. Old entries will be skipped via in_graph check.
                    for w in range(target_n):
                        if w == u or w == v:
                            continue
                        if (u, w) not in in_graph:
                            heapq.heappush(heap, (deg[u] + deg[w], random.random(), u, w))
                        if (v, w) not in in_graph:
                            heapq.heappush(heap, (deg[v] + deg[w], random.random(), v, w))
                    
                    # Optimization: If min degree is reached, we can stop?
                    # Not necessarily, we want to maximize n, but for fixed n, any valid graph works.
                    # However, stopping early might leave us with a valid graph sooner.
                    # But we must ensure ALL degrees are >= min_deg_needed.
                    # So we can check min(deg) >= min_deg_needed.
                    if min(deg) >= min_deg_needed:
                        break
                # else: edge creates C4, discard it (do not re-add)
            
            if success and min(deg) >= min_deg_needed:
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
