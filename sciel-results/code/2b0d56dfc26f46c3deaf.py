"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) construction.
    We use the parameter d such that the guaranteed girth d+5 is at least g.
    Thus d = max(2, g-5).
    The number of vertices is 2 * k**d.
    """
    start_time = time.time()
    deadline = 120.0  # Safety margin for 150s limit
    
    q = k
    # Calculate d
    # LUW graph CD(d, q) has girth at least d + 5.
    # We need girth >= g.
    d = max(2, g - 5)
    
    # If d is too large, we might exceed time limit. 
    # But for g=9, d=4, n=1250, which is very fast.
    
    # Define field operations
    if q == 4:
        # GF(4) arithmetic
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # Prime field
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
        
    # Define the rule for LUW graph
    # The rule determines how coordinates are related.
    # From the original LUW paper and standard implementations:
    # The equations are y_i - x_i = x_{a_i} y_{b_i}
    # The seed code used a specific recursive generation.
    # Let's reproduce the seed code's logic for the rule table to ensure compatibility/validity.
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # The list in seed code: [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        # Index r selects the pair
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def get_neighbours(point_tuple):
        """
        Given a point tuple (x_1, ..., x_d), yield the indices of connected lines.
        We represent vertices as (side, coords).
        Side 0: Point, Side 1: Line.
        """
        # point_tuple is x_1...x_d
        # We iterate over all possible 'first' coordinates for the line?
        # In the seed code, it iterated 'first' in range(q) and computed the rest.
        # This implies the mapping is injective for each 'first'.
        
        for first in range(q):
            # y is the line coordinates
            # y[1] = first
            # y[c] is computed for c=2..d
            y = [0] * (d + 1) # 1-indexed internally for ease
            y[1] = first
            
            for c in range(2, d + 1):
                a, b = rule[c]
                # Equation: y[c] - x[c] = x[a] * y[b]  (if side is point)
                # Or: y[c] - x[c] = ... 
                # The seed code:
                # if side == 0 (Point to Line): y[c] = x[c] + x[a] * y[b]
                # if side == 1 (Line to Point): x[c] = y[c] - x[a] * y[b] -> x[c] = y[c] - ...
                # Wait, the seed code `neighbours` function was generic for both sides.
                # It yielded `1 - side, tuple(y)`.
                # If we are at a Point (side 0), we compute Line coords y.
                # If we are at a Line (side 1), we compute Point coords x? No, the variable name was `y` in the yield.
                # Let's trace seed code `neighbours(vertex)`:
                # side, x = vertex.
                # It computes `y` (the neighbor's coordinates).
                # If side==0 (Point), it computes Line coords.
                #   y[c] = add(x[c], mul(y[a], x[b]))
                # If side==1 (Line), it computes Point coords.
                #   y[c] = sub(x[c], mul(x[a], y[b]))
                
                # Since we are generating edges, we can just generate from Points to Lines.
                # Then we have all edges.
                
                # We are at Point (side 0).
                # x is the point coordinates.
                y[c] = add(point_tuple[c-1], mul(y[a-1], point_tuple[b-1]))
                # Note: point_tuple is 0-indexed in python list, so index i is coord i+1.
                # So point_tuple[c-1] is x_c.
                # y[a-1] is y_a.
                # point_tuple[b-1] is x_b.
                
            # The neighbor is a Line with coords y[1..d]
            yield (1, tuple(y[1:]))

    # We will build the graph by BFS from the zero point to ensure we get a connected component.
    # The LUW graph is connected.
    
    start_vertex = (0, tuple([0] * d))
    
    index = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    # To speed up, we can precompute neighbors or just iterate.
    # n = 2 * q**d. For d=4, q=5, n=1250.
    # This is small. BFS is fast.
    
    for vertex in order:
        side, coords = vertex
        if time.time() > deadline - 10:
            break
            
        for neighbor in get_neighbours(coords) if side == 0 else reversed_get_neighbours(coords):
            # We need a function for side==1 too?
            # The adjacency is symmetric. 
            # If we define neighbors for Points, we get all edges (Point, Line).
            # We don't need to generate from Lines if we add edges (u, v) for u in Points.
            
            if side == 0:
                # neighbor is (1, line_coords)
                if neighbor not in index:
                    index[neighbor] = len(order)
                    order.append(neighbor)
                edges.append((index[vertex], index[neighbor]))
            else:
                # We don't generate from Lines to avoid duplicate edges or complexity.
                # But wait, if we only generate from Points, do we cover all vertices?
                # Yes, if the graph is connected and we start from a Point, we will reach all Lines 
                # and then from those Lines... wait, if we don't expand from Lines, 
                # we might miss Points that are only connected to already visited Lines?
                # No, if a Point is connected to a Line, and we visit the Line, we should find the Point.
                # But my `get_neighbours` only works for Points.
                # I need a `get_neighbours_for_line` function.
                
                pass

    # Let's implement proper BFS that handles both sides.
    
    # Re-implementing neighbor generation for clarity and correctness.
    
    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d
        if side == 0:
            # Point to Line
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule[c]
                    # y[c] = x[c] + x[a] * y[b]
                    y[c] = add(x[c-1], mul(y[a-1], x[b-1]))
                yield (1, tuple(y[1:]))
        else:
            # Line to Point
            # Equation: y[c] - x[c] = x[a] * y[b]  => x[c] = y[c] - x[a] * y[b]
            # Here x is the line coords (confusing naming in seed, let's call it l)
            # And we want to find p (point coords).
            # l[c] - p[c] = p[a] * l[b]
            # p[c] = l[c] - p[a] * l[b]
            # This is recursive on p.
            # We can iterate over p[1]?
            # p[1] is not in the equation for c=2?
            # Rule for c=2 is (1,1). p[2] = l[2] - p[1] l[1].
            # So if we pick p[1], we can determine all p[c].
            for first in range(q):
                p = [0] * (d + 1)
                p[1] = first
                for c in range(2, d + 1):
                    a, b = rule[c]
                    # p[c] = l[c] - p[a] * l[b]
                    p[c] = sub(x[c-1], mul(p[a-1], x[b-1]))
                yield (0, tuple(p[1:]))
                
    # Restart BFS with correct neighbor function
    index = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    for vertex in order:
        side, coords = vertex
        if time.time() > deadline - 10:
            break
        for neighbor in neighbours(vertex):
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            # Add edge only once. 
            # We can add edge if vertex index < neighbor index to avoid duplicates, 
            # or just collect all and deduplicate. 
            # Since it's bipartite and regular, generating from all vertices 
            # will produce each edge twice. 
            # To be safe and simple, we add if vertex[0] == 0 (Point to Line).
            if vertex[0] == 0:
                edges.append((index[vertex], index[neighbor]))
                
    # Check if we need to verify girth?
    # The task says "certified exactly with non-backtracking walks". 
    # However, the LUW construction is theoretically guaranteed to have girth >= d+5.
    # For d=4, girth >= 9.
    # We trust the construction.
    
    return len(order), edges
# EVOLVE-BLOCK-END
