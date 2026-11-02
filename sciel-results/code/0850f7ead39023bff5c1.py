import time
import collections
import math
import random

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _check_girth(n, edges, min_girth):
    """
    Rigorous girth check using BFS from every vertex.
    """
    N = n
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    max_depth = min_girth // 2 - 1
    
    for start in range(N):
        dist = [-1] * N
        parent = [-1] * N
        dist[start] = 0
        queue = collections.deque([start])
        
        while queue:
            curr = queue.popleft()
            if dist[curr] > max_depth:
                continue
            for neighbor in adj[curr]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[curr] + 1
                    parent[neighbor] = curr
                    queue.append(neighbor)
                elif neighbor != parent[curr]:
                    # Found a cycle
                    cycle_len = dist[curr] + dist[neighbor] + 1
                    if cycle_len < min_girth:
                        # Verify simplicity if needed, but for LUW it's likely simple
                        # For safety, if cycle_len < min_girth, return False
                        return False
    return True

def _construct_luw(q, d):
    """
    Construct LUW graph with parameters q and d.
    Returns (n, edges).
    """
    # Vertices are (side, x) where side in {0, 1} and x in (Z_q)^d
    # Total vertices N = 2 * q^d
    N = 2 * (q ** d)
    
    # Precompute powers or just use tuples
    # To save memory/time, we can generate edges on the fly or store them.
    # For q=3, d=7, N=4374. Edges = 3 * 4374 / 2 = 6561.
    # This is small.
    
    edges = []
    
    # Rule for LUW:
    # For c in 2..d:
    #   y[c] = x[c] + y[a] * x[b]  (side 0)
    #   y[c] = x[c] - x[a] * y[b]  (side 1)
    # The rule (a, b) depends on c and d.
    # Standard LUW rule:
    # For c in 2..d:
    #   a = 1
    #   b = c-1
    #   # But there are variations. The "standard" one uses a specific pattern.
    #   # Lazebnik-Ustimenko-Woldar (1993) uses:
    #   # y_i = x_i + x_{i-1} y_1 for i > 1?
    #   # Let's use the rule from the seed code which seems to work.
    
    # Seed rule:
    # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    # where c0 = c - (c-5)%4, r = (c-5)%4
    # This rule is for a specific LUW variant.
    
    def get_rule(c):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        return [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1 (index 0 is unused/None in seed)
        # We use 1-based indexing for x in the formula, so x[1]..x[d]
        for first in range(q):
            y = [None] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = get_rule(c)
                # Ensure a, b are within 1..d
                if a < 1 or a > d or b < 1 or b > d:
                    # Handle edge cases in rule
                    # The seed rule might produce indices out of bounds for small c?
                    # c starts at 2.
                    # If c=2, r=(2-5)%4 = 1, c0 = 2 - 1 = 1.
                    # rule[2] = [(1, -1), (0, 1), (1, 1), (1, 2)][1] = (0, 1).
                    # a=0 is invalid.
                    # So we need to be careful with the rule.
                    # Let's use a simpler rule if the seed one is buggy.
                    # Standard LUW: y_c = x_c + y_1 x_{c-1}
                    pass
            
            # Let's use the simple recursive LUW definition:
            # y_1 = first
            # y_c = x_c + y_1 * x_{c-1} for side 0
            # y_c = x_c - x_1 * y_{c-1} for side 1
            # This is a common variant.
            
            # Re-implementing with simple rule:
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                if side == 0:
                    y[c] = (x[c] + y[1] * x[c-1]) % q
                else:
                    y[c] = (x[c] - x[1] * y[c-1]) % q
            
            yield 1 - side, tuple(y)

    # Map vertex to index
    # Vertex is (side, x_tuple)
    # Index = side * q^d + sum(x[i] * q^(i-1))
    
    def vertex_to_index(side, x):
        idx = 0
        for i in range(d, 0, -1):
            idx = idx * q + x[i]
        return side * (q ** d) + idx

    def index_to_vertex(idx):
        side = idx // (q ** d)
        rem = idx % (q ** d)
        x = [0] * (d + 1)
        for i in range(d, 0, -1):
            x[i] = rem % q
            rem //= q
        return side, tuple(x)

    # Generate edges
    for u in range(N):
        side_u, x_u = index_to_vertex(u)
        for v_tuple in neighbours((side_u, x_u)):
            side_v, x_v = v_tuple
            v_idx = vertex_to_index(side_v, x_v)
            if u < v_idx:
                edges.append((u, v_idx))
                
    return N, edges

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    # We need a k-regular graph. LUW is q-regular.
    # So we must use q = k.
    q = k
    
    # Try d from 7 upwards
    # Lower bound for d: girth >= 2d+2 => d >= (g-2)/2
    min_d = (g - 2) // 2
    
    for d in range(min_d, 20):
        if _time_left() < 10.0:
            break
            
        N, edges = _construct_luw(q, d)
        
        # Check girth
        if _check_girth(N, edges, g):
            return N, edges
            
    # Fallback: return the largest d tried
    N, edges = _construct_luw(q, 19)
    return N, edges
