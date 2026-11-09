import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 140.0
    
    # Fallback function using the seed logic (refactored)
    def get_luw_fallback(k, g):
        # The seed code from the prompt
        q = k
        # d calculation from seed
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            nbrs = list(neighbours(vertex))
            for other in nbrs:
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
            if vertex[0] == 0:
                for other in nbrs:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Specific search for k=4, g=10
    if k == 4 and g == 10:
        # Try to find a small graph
        # Range of N to try
        # Lower bound 161. Known 320.
        # Let's try N in [162, 164, ..., 200] first?
        # Or just 320, 322...
        # 320 is the target.
        
        best_graph = None
        
        # Helper: Check if graph is simple, 4-regular, and has girth >= 10
        def check_validity(n, edges):
            # Build adjacency
            adj = [[] for _ in range(n)]
            for u, v in edges:
                if u == v: return False
                adj[u].append(v)
                adj[v].append(u)
            
            # Check degree
            for u in range(n):
                if len(adj[u]) != 4: return False
                # Check multi-edges
                if len(set(adj[u])) != 4: return False
            
            # Check Girth
            # BFS from each node
            for s in range(n):
                dist = [-1] * n
                dist[s] = 0
                queue = [s]
                while queue:
                    u = queue.pop(0)
                    if dist[u] >= 4: continue # Max depth 4 for girth 10 (dist u + dist v + 1 < 10)
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            queue.append(v)
                        else:
                            # Found an edge to a visited node
                            # Cycle length = dist[u] + dist[v] + 1
                            # If dist[u] + dist[v] + 1 < 10, invalid
                            # Note: if v is parent, dist[v] = dist[u]-1, len = dist[u] + dist[u]-1 + 1 = 2*dist[u].
                            # We should avoid counting parent? 
                            # Non-backtracking walks.
                            # If we encounter a visited node that is not the parent, we have a cycle.
                            # Length = dist[u] + dist[v] + 1.
                            # We need length < 10 to be invalid.
                            # But BFS tree edges don't create cycles.
                            # Cross/back edges do.
                            # To be safe, if dist[u] + dist[v] + 1 < 10 and v is not parent of u?
                            # Actually, if v is parent, length is 2*dist[u].
                            # If dist[u] = 3, len = 6. Invalid.
                            # So we just check dist[u] + dist[v] + 1 < 10.
                            if dist[u] + dist[v] + 1 < 10:
                                # Check if it's a back edge to parent
                                # Parent of u is not stored here, but if dist[v] == dist[u]-1 and v in adj[u], 
                                # it might be the parent.
                                # However, the *existence* of a cycle of length < 10 is what we care about.
                                # Even if it's a back edge in the BFS tree, it represents a cycle?
                                # No, the BFS tree path is unique. The edge (u,v) closes the loop.
                                # The cycle is path(s,v) + edge(v,u) + path(u,s)? No.
                                # Cycle is path(s,u) + edge(u,v) + path(v,s)?
                                # No, path(s,u) and path(s,v) meet at s.
                                # Cycle length is dist[u] + 1 + dist[v].
                                # This is correct for any non-tree edge.
                                # Is (u,v) a non-tree edge?
                                # If v is the parent of u, the edge is in the tree.
                                # The "cycle" formed is actually just the edge traversed back? No, that's not a cycle.
                                # A cycle requires a loop.
                                # In BFS, if we see (u,v) where dist[v] = dist[u]-1, it's the parent edge.
                                # It does NOT form a cycle of length dist[u]+dist[v]+1.
                                # It forms a path of length 2*dist[u] back to s? No.
                                # It just goes back.
                                # We must exclude the parent edge.
                                # We need to track parent.
                                pass # Implement with parent tracking
            
            return True
        
        # Efficient girth check using BFS with parent tracking
        def has_short_cycle(n, adj, girth_target):
            """Check if graph has any cycle shorter than girth_target.
            Returns (has_short_cycle, cycle_length, cycle_vertices) if found."""
            for s in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                queue = deque([s])
                while queue:
                    u = queue.popleft()
                    if dist[u] >= girth_target - 1:
                        continue
                    for v in adj[u]:
                        if v == parent[u]:
                            continue
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            # Found a potential cycle
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < girth_target:
                                return True, cycle_len, (u, v)
            return False, None, None
        
        # Search loop
        # Try N = 320, 322, 324...
        # 320 is the known best. Smaller ones likely don't exist.
        
        target_ns = [320, 322, 324, 326, 328, 330, 340, 350, 400]
        
        for N in target_ns:
            if time.time() > deadline - 5: break
            
            # Try to find a valid graph of size N
            found = False
            # Restarting
            for _ in range(50): # Limit restarts per N
                if time.time() > deadline - 5: break
                
                # Generate random 4-regular graph using configuration model
                while True:
                    stubs = []
                    for i in range(N):
                        stubs.extend([i]*4)
                    random.shuffle(stubs)
                    edges = []
                    valid = True
                    seen = set()
                    for i in range(0, 2*N, 2): # N*2 edges total, N*4/2 = 2N pairs
                        u = stubs[i]
                        v = stubs[i+1]
                        if u == v:
                            valid = False
                            break
                        key = (min(u,v), max(u,v))
                        if key in seen:
                            valid = False
                            break
                        seen.add(key)
                        edges.append((u,v))
                    if valid:
                        break
                    if time.time() > deadline - 10:
                        break
                
                if not valid: continue
                
                # Check girth
                adj = [[] for _ in range(N)]
                for u,v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                short, _, _ = has_short_cycle(N, adj)
                if not short:
                    return N, edges
                
                # If short cycle, try local search to break cycles
                # Simple approach: find a short cycle and try to reroute edges
                for _attempt in range(20):
                    if time.time() > deadline - 5: break
                    short, u, v = has_short_cycle(N, adj)
                    if not short:
                        return N, edges
                    # Try to fix by swapping edges
                    # Find edges incident to u and v that could be swapped
                    # This is heuristic; may not work well
                    # For now, just retry with new random graph
                    break
                
        # Fallback
        n, edges = get_luw_fallback(k, g)
        return n, edges

    else:
        # General case fallback
        n, edges = get_luw_fallback(k, g)
        return n, edges
