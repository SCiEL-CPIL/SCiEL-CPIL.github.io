import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    if k != 3:
        # Fallback for other k, though task is 3-regular
        # Try to use LUW with q=k if possible, or just return a dummy
        # For now, focus on k=3
        pass

    # For k=3, use LUW graph with q=3
    # We need girth >= g.
    # LUW graph CD(d, q) has girth roughly 2d+2.
    # So we need 2d+2 >= g => d >= (g-2)/2.
    # For g=16, d >= 7.
    
    q = k
    d = (g - 2) // 2 + 1
    if d < 3:
        d = 3
        
    # Try this d first
    # If it fails girth check, try d+1, etc.
    
    best_n = float('inf')
    best_edges = None
    
    for d_try in range(d, d + 5):
        if time.time() > deadline:
            break
            
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        # Generate rules for LUW graph
        # rule[c] = (a, b) for c in 2..d
        # Indices a, b must be in 0..d-1
        # Standard rule:
        # c=2: (0, 1) ? 
        # Let's use the rule from the "Cage" paper.
        # The map is defined by a sequence of coefficients.
        # For q=3, d=7.
        
        # We define the "twist" or the group action.
        # Vertices are (side, vec) where vec is in F_q^d.
        # side 0 and 1.
        # Neighbors of (0, x):
        # 1. (1, x) is NOT a neighbor in LUW. 
        # LUW is bipartite? Yes.
        # The edges are defined by:
        # (0, x) connected to (1, y) where y = f(x) ?
        # No, it's a 3-regular graph.
        # Each vertex has 3 neighbors.
        
        # Construction from "The (3,10)-cage":
        # Vertices are (i, x) for i in {0,1}, x in F_3^5.
        # Neighbors of (0, x):
        # 1. (1, x) -- No, this is a matching, degree 1 so far.
        # The other 2 neighbors come from the group structure.
        
        # Let's use the explicit adjacency from the seed code logic, fixed.
        # The seed code logic:
        # y[1] = first (0..q-1)
        # y[c] = add(x[c], mul(y[a], x[b]))
        
        # Let's define the rules for d=7, q=3.
        # We need to ensure the graph is 3-regular and simple.
        
        rule = {}
        # We need d-1 rules? 
        # The loop in seed was for c in 2..d.
        # So d-1 steps.
        # Let's use a fixed set of rules for q=3.
        # From Exoo's papers, the rules are specific.
        
        # Heuristic rules for small q:
        # c=2: (0, 1) -> y[2] = x[2] + y[0]*x[1] ? 
        # Wait, y[0] is the "first" component.
        # In the seed: y = [None, first] + [0]*(d-1).
        # So y[0] is None, y[1] is first.
        # Indices in x are 0..d-1.
        
        # Let's align indices:
        # Vector x has indices 0..d-1.
        # Vector y has indices 1..d. (y[1] is the free variable)
        # So y[c] for c=2..d depends on x.
        
        # Let's define rule[c] = (a, b) where a, b are indices in x (0..d-1)
        # And y[a] is used? No, y[a] might be y[1] if a=1.
        # In seed: mul(y[a], x[b]).
        # If a=1, y[1] is 'first'.
        
        # Let's try this rule set for q=3:
        # c=2: (1, 0) -> y[2] = x[2] + y[1]*x[0]
        # c=3: (1, 1) -> y[3] = x[3] + y[1]*x[1]
        # c=4: (1, 2) -> y[4] = x[4] + y[1]*x[2]
        # c=5: (1, 3) -> y[5] = x[5] + y[1]*x[3]
        # c=6: (1, 4) -> y[6] = x[6] + y[1]*x[4]
        # c=7: (1, 5) -> y[7] = x[7] + y[1]*x[5]
        
        # This creates a specific structure.
        
        for c in range(2, d_try + 1):
            # Simple cyclic rule
            rule[c] = (1, (c - 2) % d_try)
            
        def get_neighbours(vertex):
            side, x = vertex
            x_list = list(x)
            for first in range(q):
                # y indices 1..d
                # x indices 0..d-1
                # y[c] corresponds to x[c-1]? 
                # In seed: y[c] was updated.
                # Let's store y as a list of length d+1 (indices 1..d)
                y = [0] * (d_try + 1)
                y[1] = first
                for c in range(2, d_try + 1):
                    a, b = rule[c]
                    # a is index in y (1..d)
                    # b is index in x (0..d-1)
                    # If a == 1, y[1] is known.
                    # If a > 1, y[a] was computed in previous step?
                    # The rule y[c] = ... y[a] ... implies a < c.
                    # Our rule a=1 always, so y[1] is always available.
                    
                    val_y_a = y[a]
                    val_x_b = x_list[b]
                    
                    if side == 0:
                        y[c] = add(val_x_b, mul(val_y_a, val_x_b)) # Wait, x[c] in seed?
                        # Seed: y[c] = add(x[c], ...). 
                        # x has length d. x[c] is out of bounds if c=d.
                        # So x index should be c-1.
                        y[c] = add(x_list[c-1], mul(val_y_a, val_x_b))
                    else:
                        y[c] = sub(x_list[c-1], mul(val_x_b, val_y_a))
                        
                # The resulting vector is y[1..d]
                # But x was length d.
                # The neighbor vertex should have a vector of length d.
                # So map y[1..d] to indices 0..d-1.
                new_x = tuple(y[1:d_try+1])
                yield (1 - side, new_x)
                
        # Generate all vertices and edges
        # Vertices: (side, x)
        # We can enumerate all 2 * q^d vertices.
        # Check if it's connected. If not, take largest component.
        
        all_vertices = []
        for side in range(2):
            for x in range(q ** d_try):
                # Convert int x to tuple of digits base q
                digits = []
                temp = x
                for _ in range(d_try):
                    digits.append(temp % q)
                    temp //= q
                x_tuple = tuple(digits)
                all_vertices.append((side, x_tuple))
                
        # Map vertex to index
        v_index = {v: i for i, v in enumerate(all_vertices)}
        
        edges = []
        seen_edges = set()
        
        for i, v in enumerate(all_vertices):
            for u in get_neighbours(v):
                if u in v_index:
                    j = v_index[u]
                    if i < j:
                        e = (i, j)
                        if e not in seen_edges:
                            seen_edges.add(e)
                            edges.append(e)
        
        n = len(all_vertices)
        
        # Check regularity
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            
        if any(d != k for d in deg):
            continue
            
        # Check girth
        R = g // 2 - 1
        valid = True
        for v in range(n):
            if time.time() > deadline:
                valid = False
                break
                
            dist = [-1] * n
            parent = [-1] * n
            dist[v] = 0
            q_bfs = deque([v])
            
            while q_bfs:
                curr = q_bfs.popleft()
                if dist[curr] >= R:
                    continue
                for nei in all_vertices[curr][0] and [] or []: # Placeholder
                    pass
                    
            # Re-do BFS properly
            dist = [-1] * n
            parent = [-1] * n
            dist[v] = 0
            q_bfs = deque([v])
            
            adj_list = [[] for _ in range(n)]
            for u, w in edges:
                adj_list[u].append(w)
                adj_list[w].append(u)
                
            while q_bfs:
                curr = q_bfs.popleft()
                if dist[curr] >= R:
                    continue
                for nei in adj_list[curr]:
                    if dist[nei] == -1:
                        dist[nei] = dist[curr] + 1
                        parent[nei] = curr
                        q_bfs.append(nei)
                    else:
                        # Back edge or cross edge
                        if nei != parent[curr]:
                            cycle_len = dist[curr] + dist[nei] + 1
                            if cycle_len < g:
                                valid = False
                                break
                if not valid:
                    break
            if not valid:
                break
                
        if valid:
            if n < best_n:
                best_n = n
                best_edges = edges
            break # Found a valid graph with this d
            
    if best_n == float('inf'):
        # Fallback: return a small invalid graph? No, must be valid.
        # If no LUW graph works, try random?
        # For now, return the largest d tried if it was "valid" in structure but failed girth?
        # No, must return a valid graph.
        # If we can't find one, we might fail.
        # But d=7 should work for g=16.
        
        # Emergency fallback: return a cycle C_n? No, girth n.
        # If n >= g, a cycle is valid.
        # But it's not 3-regular.
        
        # Let's just return the last constructed edges even if girth failed?
        # No, that scores 0.
        
        # Try a larger d
        for d_try in range(d + 5, d + 10):
             # ... same logic ...
             pass
             
    return best_n, best_edges
