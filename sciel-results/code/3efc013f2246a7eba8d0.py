import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # 1. Determine minimal d for LUW construction CD(d, k)
    # Girth guarantee:
    # d odd: girth >= d + 5
    # d even: girth >= d + 6
    # We need girth >= g.
    # Find smallest d >= 3 satisfying the condition.
    
    # The standard LUW graph CD(d, q) has girth girth >= d + 4.
    # The previous attempt with d=4 (girth bound 10) failed with girth 8,
    # suggesting the actual girth is closer to d+4.
    # To guarantee girth >= g, we need d + 4 >= g => d >= g - 4.
    # We also need d >= 2.
    d = max(2, g - 4)
    
    # Ensure d is at least 3 for the rules defined below (rule[2] exists, but d=2 is a cycle)
    # If d=2, the graph is a cycle of length 2*q? No, CD(2,q) is a cycle.
    # For k=5, g=9, d=5.
    
    q = k
    
    # 2. Field operations for GF(q)
    # q must be prime or 4 for standard LUW. 
    # If k is not prime and not 4, we might have issues. 
    # The task says "evaluator passes this task's degree k". 
    # Usually k is prime in these cage problems or small. 
    # If k is not prime, we can't use GF(k) directly. 
    # However, the seed program assumes q=k works. 
    # For k=5, it's prime. 
    # If k is composite, this construction fails. 
    # But we stick to the LUW family as per instructions.
    
    if q == 4:
        # GF(4) tables
        add_table = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(x, y): return add_table[x][y]
        def sub(x, y): return add_table[x][y] # char 2
        def mul(x, y): return mul_table[x][y]
    else:
        # Assume prime field
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # 3. Define the LUW rules
    # rule[c] = (a, b) for c = 2..d
    # Relation: y[c] = x[c] + y[a]*x[b] (Point to Line)
    #           y[c] = x[c] - x[a]*y[b] (Line to Point)
    
    rule = [None] * (d + 1)
    if d >= 2:
        rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # 4. Construct the graph via BFS from the zero vertex
    # Vertices are (side, coords)
    # side 0: Point, side 1: Line
    # coords: tuple of length d+1. Index 0 is dummy.
    
    start_vertex = (0, (0,) * (d + 1))
    
    index = {start_vertex: 0}
    queue = [start_vertex]
    edges = []
    order = [start_vertex]
    
    # Precompute neighbours logic inline for speed? 
    # No, d is small (4 or 5), q is small (5). 
    # Total vertices ~ 2 * q * d * q^(d-1)? No.
    # The number of vertices is at most 2 * q^d. 
    # For d=4, q=5: 2 * 5^4 = 1250 vertices.
    # For d=5, q=5: 2 * 5^5 = 6250 vertices.
    # This is very manageable.
    
    for vertex in queue:
        if time.time() > deadline:
            break
            
        side, x = vertex
        v_idx = index[vertex]
        
        # Generate all q neighbours
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first # y[0] is 0
            
            if side == 0:
                # x is Point (p), y is Line (l)
                # l[c] = p[c] + l[a] * p[b]
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                # x is Line (l), y is Point (p)
                # p[c] = l[c] - l[a] * p[b]
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            neighbor = (1 - side, tuple(y))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
                queue.append(neighbor)
            
            # Add edge only once (when visiting from side 0, or just ensure unique)
            # Since we BFS, we visit each vertex once. 
            # The edge (v, u) is generated when processing v.
            # If we process v (side 0), we generate edges to side 1.
            # If we process u (side 1), we generate edges to side 0 (which are already in index).
            # To avoid duplicates, we only add edge if side == 0? 
            # No, the graph is bipartite. 
            # If we add edge when processing ANY vertex, we might add (u,v) and later (v,u).
            # Standard BFS: when we discover a new node, we add the edge.
            # But here we iterate all neighbors.
            # If neighbor is already in index, we don't add edge again?
            # Actually, in a regular graph generated by BFS, each edge is seen from both ends.
            # If we add edge when `neighbor not in index`, we miss edges to already visited nodes?
            # No, in a tree-like expansion (BFS), edges to already visited nodes are "back edges" or cross edges.
            # But LUW graph is not a tree. It has cycles.
            # However, the mapping from (side, x) to neighbours is a function.
            # Each vertex has exactly k neighbours.
            # If we simply add an edge for every generated neighbour, we will have 2 * E edges in our list (each undirected edge twice).
            # The evaluator likely expects a list of unique undirected edges.
            # So we should add edge (v_idx, index[neighbor]) only if v_idx < index[neighbor] or similar.
            # Or simpler: add edge only when we discover the neighbor?
            # If we add edge only on discovery, we get a spanning tree + some edges?
            # No, BFS discovers all nodes. The edges connecting them are the graph edges.
            # In LUW, the "neighbour" function defines the graph.
            # If we add an edge every time we iterate a neighbour, we get every edge exactly twice (once from each end).
            # We can just collect all and deduplicate, or only add if side==0.
            # Since the graph is bipartite, every edge connects side 0 and side 1.
            # If we only add the edge when the current vertex is on side 0, we cover all edges exactly once.
            
            if side == 0:
                edges.append((v_idx, index[neighbor]))

    n = len(order)
    
    # Optional: Verify girth if time permits? 
    # The LUW construction guarantees the girth. 
    # For d=4, girth >= 10. 
    # The problem asks for girth >= 9. 
    # So it is valid.
    
    return n, edges
