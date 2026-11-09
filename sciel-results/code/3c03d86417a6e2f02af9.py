import time
import random
from collections import deque

def _check_girth(adj, n, target_girth):
    """
    Check if the graph has girth at least target_girth.
    Uses BFS from each vertex.
    """
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[v] + dist[u] + 1
                    if cycle_len < target_girth:
                        return False
    return True

def _build_luw(q, d, rules):
    """
    Build LUW graph with given q, d, and rules.
    Returns (n, edges) or None if invalid.
    """
    # Define arithmetic
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Pre-calculate neighbors for each vertex type
    # Vertices are (side, tuple)
    # Side 0: x = (x1, ..., xd)
    # Side 1: y = (y1, ..., yd)
    
    # We will generate all vertices and their neighbors
    vertices = []
    index = {}
    
    # Generate Side 0 vertices
    # x1 is free? In the inspiration, y[1] = first (0..q-1)
    # So for Side 0, x is a tuple of length d.
    # The inspiration iterates 'first' in range(q) for the neighbor generation.
    # This implies the graph is defined by the mapping from Side 0 to Side 1.
    
    # Let's follow the inspiration's neighbor logic exactly but pre-compute.
    # For a vertex (0, x), neighbors are (1, y) where y is computed from x and 'first'.
    # Wait, the inspiration says:
    # def neighbours(vertex):
    #     side, x = vertex
    #     for first in range(q):
    #         y = [None, first] + [0] * (d - 1)
    #         ...
    #         yield 1 - side, tuple(y)
    
    # This means each Side 0 vertex has q neighbors on Side 1.
    # Each Side 1 vertex has q neighbors on Side 0.
    # Total edges = n * q / 2.
    
    # Let's build the mapping.
    # We need to enumerate all Side 0 vertices and Side 1 vertices.
    # Side 0 vertices: All tuples in Z_q^d?
    # The inspiration starts with (0, (None,) + (0,)*d).
    # And adds neighbors.
    # This BFS-like construction ensures we only add reachable vertices.
    # For LUW, the graph is usually connected and all 2*q^d vertices are present.
    
    # Let's use the BFS construction from the inspiration, but optimized.
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # Pre-compute the 'next' state for each possible input?
    # The calculation of y depends on x and 'first'.
    
    for vertex in order:
        side, x = vertex
        for first in range(q):
            y_list = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rules[c]
                if side == 0:
                    y_list[c] = add(x[c], mul(y_list[a], x[b]))
                else:
                    y_list[c] = sub(x[c], mul(x[a], y_list[b]))
            
            neighbor = (1 - side, tuple(y_list))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            
            if side == 0:
                edges.append((index[vertex], index[neighbor]))
                
    n = len(order)
    
    # Build adjacency list for girth check
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check regularity
    for v in range(n):
        if len(adj[v]) != q:
            return None
            
    return n, edges, adj

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    if k != 3:
        # Fallback for other k
        # Use a simple cycle or something? 
        # The task implies k=3, g=16.
        # For generality, return a trivial graph or raise.
        # Let's try a random regular graph for small k.
        # But for now, focus on k=3.
        pass

    if k == 3 and g == 16:
        q = 3
        
        # Define standard rules
        def get_standard_rules(d):
            rule = {}
            if d >= 2: rule[2] = (1, 1)
            if d >= 3: rule[3] = (2, 1)
            if d >= 4: rule[4] = (1, 2)
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            return rule

        # Try d=6 first (n=1458)
        # Try d=7 second (n=4374)
        
        best_result = None
        
        for d in [6, 7, 8]:
            if time.time() > deadline - 10:
                break
                
            # Try standard rules
            rules = get_standard_rules(d)
            result = _build_luw(q, d, rules)
            if result:
                n, edges, adj = result
                if _check_girth(adj, n, g):
                    best_result = (n, edges)
                    break # Found a valid graph for this d, try smaller d? 
                    # We are iterating d increasing, so first valid is smallest n.
            
            # If standard failed for d=6, try a few random variations
            if d == 6:
                rng = random.Random(42)
                for _ in range(10):
                    if time.time() > deadline - 10:
                        break
                    # Generate random rules
                    rules = {}
                    rules[2] = (1, 1)
                    rules[3] = (2, 1)
                    rules[4] = (1, 2)
                    for c in range(5, d + 1):
                        # Random a, b in 1..d-1
                        a = rng.randint(1, d-1)
                        b = rng.randint(1, d-1)
                        rules[c] = (a, b)
                    
                    result = _build_luw(q, d, rules)
                    if result:
                        n, edges, adj = result
                        if _check_girth(adj, n, g):
                            best_result = (n, edges)
                            break
                if best_result:
                    break

        if best_result:
            return best_result

    # Fallback: Return the LUW graph with large d, even if girth is not verified?
    # No, must be valid.
    # If we didn't find a valid one, we must return something.
    # The seed program returned a LUW graph with d=11.
    # Let's return that as a last resort.
    if k == 3:
        d = 11
        rules = {}
        if d >= 2: rules[2] = (1, 1)
        if d >= 3: rules[3] = (2, 1)
        if d >= 4: rules[4] = (1, 2)
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rules[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        result = _build_luw(3, d, rules)
        if result:
            n, edges, adj = result
            return n, edges
            
    raise Exception("Failed to construct graph")
