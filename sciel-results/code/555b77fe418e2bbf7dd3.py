"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Fast path: known optimal cages for small parameters
    if k == 3:
        if g <= 3:
            return 3, [(0,1),(1,2),(0,2)]
        elif g <= 4:
            # K4 is 3-regular, girth 3, but we need girth >= 4
            # Actually K4 has triangles, so girth 3. Need girth >= 4.
            # The 3-regular graph with girth 4 on fewest vertices: C6? No, C6 is 2-regular.
            # K_{3,3} is 3-regular, girth 4, 6 vertices.
            return 6, [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)]
        elif g <= 5:
            # Petersen graph: 3-regular, girth 5, 10 vertices
            outer = list(range(5))
            inner = [5, 6, 7, 8, 9]
            edges = []
            for i in range(5):
                edges.append((outer[i], outer[(i+1)%5]))
                edges.append((outer[i], inner[i]))
                edges.append((inner[i], inner[(i+2)%5]))
            return 10, edges
        elif g <= 6:
            # 3-regular, girth 6: smallest is 14 vertices (the (3,6)-cage is not standard, 
            # actually the smallest 3-regular girth 6 graph has 14 vertices)
            # Let me use a known construction. The dodecahedron is 3-regular girth 5 with 20 vertices.
            # For girth 6, 3-regular: the smallest known has 14 vertices.
            # Actually, let me just use LUW for this.
            pass
        elif g <= 7:
            # 3-regular, girth 7: 24 vertices (Heawood is girth 6 with 14)
            pass
        elif g <= 8:
            pass
        elif g <= 9:
            pass
    elif k == 4:
        if g <= 3:
            return 4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        elif g <= 4:
            # K_{4,4} is 4-regular, girth 4, 8 vertices
            return 8, [(i,j) for i in range(4) for j in range(4,8)]
        elif g <= 5:
            # 4-regular, girth 5: 10 vertices? The Petersen is 3-regular.
            # Smallest 4-regular girth 5: 10 vertices? Let me think...
            # Actually the complement of the Petersen graph is 4-regular on 10 vertices with girth 3.
            # The smallest 4-regular graph with girth 5 has 10 vertices? No.
            # Let me use LUW.
            pass
        elif g <= 6:
            pass
        elif g <= 7:
            # 4-regular, girth 7: 22 vertices (known)
            pass
        elif g <= 8:
            pass
    elif k == 5:
        if g <= 3:
            # K6 is 5-regular, girth 3
            return 6, [(i,j) for i in range(6) for j in range(i+1,6)]
        elif g <= 4:
            # K_{5,5} is 5-regular, girth 4, 10 vertices
            return 10, [(i,j) for i in range(5) for j in range(5,10)]
        elif g <= 5:
            pass
        elif g <= 6:
            pass
        elif g <= 7:
            pass
        elif g <= 8:
            pass
        elif g <= 9:
            # This is the main case: 5-regular, girth 9
            pass
    
    # Use LUW construction for general case
    return _luw_construct(k, g, deadline)

def _luw_construct(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_table = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
        sub_table = [[0,3,2,1],[1,0,3,2],[2,1,0,3],[3,2,1,0]]
        mul_table = table
    else:
        # Precompute addition, subtraction, multiplication tables
        add_table = [[(x + y) % q for y in range(q)] for x in range(q)]
        sub_table = [[(x - y) % q for y in range(q)] for x in range(q)]
        mul_table = [[x * y % q for y in range(q)] for x in range(q)]
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Precompute rule lookup as list for speed
    rule_list = [None] * (d + 1)
    for c in range(2, d + 1):
        rule_list[c] = rule[c]
    
    def neighbours(vertex):
        side, x = vertex
        if side == 0:
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule_list[c]
                    y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
                yield 1, tuple(y)
        else:
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule_list[c]
                    y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]
                yield 0, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    edge_set = set()
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                e = (index[vertex], index[other])
                if e not in edge_set:
                    edge_set.add(e)
                    edges.append(e)
        if time.time() > deadline:
            break
    
    return len(order), edges
# EVOLVE-BLOCK-END
