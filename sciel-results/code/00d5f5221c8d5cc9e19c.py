import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_edges = 0
    best_graph = None

    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Initialize graph
        # Start with empty graph or random
        # Let's start with empty and greedily add
        adj = [0] * n  # Bitsets
        
        # Greedy construction
        # Order vertices randomly? Or just 0..39
        # Try to add edges in random order
        edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges_to_try)
        
        current_edges = 0
        
        # Fast check function
        def can_add(u, v):
            # Check if path of length 5 exists from u to v
            # S0 = {u}
            s = 1 << u
            # We want to compute S5
            # S_{k+1} = Union of N(x) for x in S_k
            # But we must ensure we don't use u or v in the interior?
            # Actually, if we find a walk, we reject.
            # This might be too conservative (rejecting valid additions that only create non-simple walks).
            # But it's safe.
            
            # BFS/Bitset expansion
            # S0
            s0 = 1 << u
            s1 = adj[u]
            s2 = 0
            for i in range(n):
                if (s1 >> i) & 1:
                    s2 |= adj[i]
            # Mask out u and v? 
            # If we hit u or v early, the walk is not simple.
            # But a walk of length 5 hitting u or v internally is not a simple path.
            # However, if we just want to be safe, we can mask out u and v from intermediate steps.
            # Let's mask out u and v from s1, s2, s3, s4 to ensure the path is simple (u, ..., v).
            mask_uv = ~((1 << u) | (1 << v))
            s1 &= mask_uv
            s2 &= mask_uv
            s3 = 0
            for i in range(n):
                if (s2 >> i) & 1:
                    s3 |= adj[i]
            s3 &= mask_uv
            s4 = 0
            for i in range(n):
                if (s3 >> i) & 1:
                    s4 |= adj[i]
            s4 &= mask_uv
            s5 = 0
            for i in range(n):
                if (s4 >> i) & 1:
                    s5 |= adj[i]
            
            if (s5 >> v) & 1:
                return False
            return True

        for u, v in edges_to_try:
            if time.time() > deadline:
                break
            if not (adj[u] >> v) & 1:
                if can_add(u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges += 1
        
        # Local search: try to add more, or swap
        # If we are stuck, try removing an edge to allow more
        # But greedy might be enough for a good start.
        # Let's add a simple local improvement step.
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding any missing edge
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] >> v) & 1:
                        if can_add(u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            current_edges += 1
                            improved = True
                            if time.time() > deadline:
                                break
                if time.time() > deadline:
                    break
            
            if not improved:
                # Try a swap: remove 1, add 2? 
                # Or just remove 1 and try to add more.
                # Let's try removing a random edge and re-running the greedy fill.
                # This is expensive. 
                # Maybe just accept the current state if no single addition works.
                # To escape local optima, we can do a "kick": remove k random edges, then greedily fill.
                if restart % 5 == 0: # Do kicks occasionally
                    # Remove 2 random edges
                    edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]
                    if len(edges_list) > 2:
                        random.shuffle(edges_list)
                        e1 = edges_list[0]
                        e2 = edges_list[1]
                        adj[e1[0]] &= ~(1 << e1[1])
                        adj[e1[1]] &= ~(1 << e1[0])
                        adj[e2[0]] &= ~(1 << e2[1])
                        adj[e2[1]] &= ~(1 << e2[0])
                        current_edges -= 2
                        
                        # Re-fill
                        for u, v in edges_to_try:
                            if not (adj[u] >> v) & 1:
                                if can_add(u, v):
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
                                    current_edges += 1
                        improved = True # Continue loop to see if more can be added
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_graph = adj[:]
            
    # Convert best_graph to list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_graph[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
