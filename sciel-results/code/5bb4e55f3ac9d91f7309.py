import time
import random
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses voltage assignment on K_{4,4} for k=4.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    if k != 4:
        # Fallback to seed-like behavior or error
        # The task specifies k=4, g=10
        pass
        
    # Base graph: K_{4,4}
    # Vertices: 0..3 (left), 4..7 (right)
    # Edges: (i, 4+j) for all i, j
    base_vertices = 8
    base_edges = []
    for i in range(4):
        for j in range(4):
            base_edges.append((i, 4 + j))
            
    # We want to find m and voltages in Z_m such that girth >= 10
    # n = 8 * m
    # Target n < 320 => m < 40
    # Lower bound n >= 243 => m >= 31
    
    best_n = 320 # Default to known cage size
    best_edges = None
    
    # Try m values from high to low to find smallest n
    # We can randomize the order or go down
    for m in range(39, 30, -1):
        if time.time() > deadline:
            break
            
        # Try to find a voltage assignment
        # Voltages are assigned to directed edges
        # For each undirected edge, assign v and -v
        # There are 16 edges in K_{4,4}
        # We need to avoid closed walks of length 4, 6, 8 having sum 0
        
        # Heuristic: Randomized local search
        found = False
        for attempt in range(1000):
            if time.time() > deadline:
                break
                
            # Initialize random voltages
            # v[e] in Z_m
            # e is (u, v)
            voltages = {}
            for u, v in base_edges:
                val = random.randint(1, m - 1) # Non-zero to avoid trivial loops?
                voltages[(u, v)] = val
                voltages[(v, u)] = (-val) % m
            
            # Check girth
            # We need to check all closed walks of length < 10
            # This is expensive.
            # Instead, check specific short cycles.
            # In K_{4,4}, cycles of length 4 are abundant.
            # Cycle: u1-v1-u2-v2-u1
            # Sum: v(u1,v1) + v(v1,u2) + v(u2,v2) + v(v2,u1)
            # Note: v(v1,u2) = -v(u2,v1)
            
            valid = True
            # Check all 4-cycles
            # u1, u2 in Left, v1, v2 in Right
            for u1 in range(4):
                for u2 in range(u1 + 1, 4):
                    for v1 in range(4):
                        for v2 in range(v1 + 1, 4):
                            # Cycle u1-v1-u2-v2-u1
                            s = (voltages[(u1, 4+v1)] + 
                                 voltages[(4+v1, u2)] + 
                                 voltages[(u2, 4+v2)] + 
                                 voltages[(4+v2, u1)]) % m
                            if s == 0:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid:
                continue
                
            # Check 6-cycles
            # u1-v1-u2-v2-u3-v3-u1
            for u1 in range(4):
                for u2 in range(u1 + 1, 4):
                    for u3 in range(u2 + 1, 4):
                        for v1 in range(4):
                            for v2 in range(4):
                                for v3 in range(4):
                                    # Ensure distinct? 
                                    # K_{4,4} bipartite.
                                    # u1 != u2 != u3
                                    # v1, v2, v3 can be anything?
                                    # If v1==v2, it's a 4-cycle.
                                    # We already checked 4-cycles.
                                    # But 6-cycle can reuse vertices?
                                    # No, simple cycle.
                                    # So v1, v2, v3 must be distinct?
                                    # In K_{4,4}, a 6-cycle uses 3 left, 3 right.
                                    # So v1, v2, v3 distinct.
                                    if v1 == v2 or v2 == v3 or v1 == v3:
                                        continue
                                    s = (voltages[(u1, 4+v1)] + 
                                         voltages[(4+v1, u2)] + 
                                         voltages[(u2, 4+v2)] + 
                                         voltages[(4+v2, u3)] + 
                                         voltages[(4+v3, u1)]) % m
                                    if s == 0:
                                        valid = False
                                        break
                                if not valid: break
                            if not valid: break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid:
                continue
                
            # Check 8-cycles
            # u1-v1-u2-v2-u3-v3-u4-v4-u1
            # 4 left, 4 right.
            # This is more complex.
            # For now, assume if 4 and 6 are avoided, 8 might be okay?
            # No, we must check.
            # But 8-cycles are many.
            # Given time constraint, we might skip 8-check or do a sample.
            # However, if 4 and 6 are non-zero, 8 is likely non-zero for random voltages.
            
            # Construct the lifted graph
            n = base_vertices * m
            edges = []
            # Vertex id: (base_vertex, group_element)
            # Map to 0..n-1
            # id = base_vertex * m + group_element
            
            for u, v in base_edges:
                val = voltages[(u, v)]
                for g_elem in range(m):
                    u_new = u * m + g_elem
                    v_new = v * m + (g_elem + val) % m
                    edges.append((u_new, v_new))
            
            # Verify degree
            # Each vertex has 4 neighbors in base, so 4 in lift.
            
            # Verify girth (exact)
            # This is expensive. We'll trust the 4/6 check for now 
            # and do a quick BFS girth check on the result.
            if _check_girth(n, edges, g):
                best_n = n
                best_edges = edges
                found = True
                break # Found a valid graph for this m, try smaller m
        
        if found:
            # We found a graph with this m. Since we iterate m downwards, 
            # this is the smallest n for this m. 
            # But we want the smallest n overall.
            # So we can break if we found one?
            # Yes, because smaller m means smaller n.
            # But we might find a valid one for m=39, but a better one for m=31?
            # No, we iterate m from 39 down to 31.
            # So the first valid one found is the smallest m (largest n)?
            # No, range(39, 30, -1) is 39, 38, ..., 31.
            # So we find the largest m first?
            # We want the SMALLEST n, which is SMALLEST m.
            # So we should iterate m from 31 up to 39?
            # And return the first valid one?
            # Yes.
            pass

    # If we found a best_edges, return it
    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback to seed implementation
    return _seed_construct(k, g)

def _check_girth(n, edges, g):
    """Check if graph has girth >= g."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex to find shortest cycle
    # This is O(n * (n+E)) which is too slow for n=256?
    # 256 * 256 * 1000?
    # Better: BFS from each vertex, track parent.
    # If we encounter a visited node not parent, cycle length = dist[u] + dist[v] + 1.
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _seed_construct(k, g):
    # Copy of the seed code
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
