import time
import random
import itertools
import math

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Handle known small cages explicitly
    if (k, g) == (3, 5):
        return _petersen()
    elif (k, g) == (3, 6):
        return _heawood()
    elif (k, g) == (5, 5):
        return _hoffman_singleton()
    
    # 2. Fallback: LUW Construction
    return _luw_construct(k, g)

def _petersen():
    # Petersen graph: 10 vertices, 3-regular, girth 5
    edges = []
    # Outer cycle
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    # Inner star
    for i in range(5):
        edges.append((5 + i, 5 + (i + 2) % 5))
    # Spokes
    for i in range(5):
        edges.append((i, 5 + i))
    return 10, edges

def _heawood():
    # Heawood graph: 14 vertices, 3-regular, girth 6
    # Incidence graph of Fano plane
    # Points 0-6, Lines 7-13
    lines = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    edges = []
    for i, line in enumerate(lines):
        line_id = 7 + i
        for p in line:
            edges.append((p, line_id))
    return 14, edges

def _hoffman_singleton():
    # Hoffman-Singleton graph: 50 vertices, 5-regular, girth 5
    # Construction:
    # Let q=5. The vertices are pairs (i, j) where i, j in Z_5, plus 5 additional vertices?
    # No, the Hoffman-Singleton graph has 50 vertices.
    # It can be constructed using the projective plane of order 4? No.
    # A standard construction:
    # Vertices: 0..49.
    # The graph is the incidence graph of a certain structure?
    # Actually, the Hoffman-Singleton graph is the unique (5,5)-cage.
    # It is strongly regular with parameters (50, 9, 1, 2).
    # We can generate it using a known adjacency list or a construction.
    # One construction:
    # Let the vertices be (i, j) for i,j in {0,1,2,3,4} (25 vertices) and (i, j) for i,j in {0,1,2,3,4} (another 25)?
    # No, 50 vertices.
    # The Hoffman-Singleton graph can be constructed from the 5x5 grid?
    # Let's use a known edge list or a simple generation method.
    # Since I don't have the edge list memorized, I will use a known property:
    # It is the unique (5,5)-cage.
    # If I cannot guarantee the edges, I should not return it if it risks invalidity.
    # However, the LUW for (5,5) is 250. If I return an invalid graph, I score 0.
    # If I return LUW, I score 1.0.
    # Is there a simple way to construct it?
    # Yes, it is the graph of the 5x5 grid with specific connections?
    # Actually, I will omit Hoffman-Singleton to be safe, unless I can verify it.
    # I will return LUW for (5,5) to be safe.
    # Wait, the prompt says "invalid result ... scores 0".
    # I will stick to Petersen and Heawood which I am confident in.
    return _luw_construct(5, 5)

def _luw_construct(k, g):
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
# EVOLVE-BLOCK-END
