import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s limit
    
    # Hardcoded known cages for small (k, g)
    cages = {}
    
    # Petersen Graph: k=3, g=5, n=10
    petersen_edges = [(0,1), (1,2), (2,3), (3,4), (4,0),
                      (5,7), (7,9), (9,6), (6,8), (8,5),
                      (0,5), (1,6), (2,7), (3,8), (4,9)]
    cages[(3, 5)] = (10, petersen_edges)
    
    # Heawood Graph: k=3, g=6, n=14
    heawood_edges = []
    fano_lines = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    for i, line in enumerate(fano_lines):
        line_id = 7 + i
        for p in line:
            heawood_edges.append((p, line_id))
    cages[(3, 6)] = (14, heawood_edges)
    
    # Möbius-Kantor Graph: k=3, g=7, n=16
    mk_edges = []
    for i in range(8):
        mk_edges.append((i, 8 + i))
        mk_edges.append((i, 8 + (i + 1) % 8))
        mk_edges.append((i, 8 + (i + 3) % 8))
    cages[(3, 7)] = (16, mk_edges)
    
    # Clebsch Graph: k=5, g=5, n=16
    # The Clebsch graph is the 5-regular graph on 16 vertices with girth 5
    # It can be constructed as the complement of the line graph of K_8? No.
    # Actually, the Clebsch graph is the halved 5-cube.
    # Vertices are binary strings of length 5 with even parity.
    # Two vertices are adjacent if they differ in exactly 2 positions.
    clebsch_edges = []
    vertices = []
    for i in range(16):
        # Binary representation with even number of 1s
        if bin(i).count('1') % 2 == 0:
            vertices.append(i)
    # Map to 0..15
    vmap = {v: i for i, v in enumerate(vertices)}
    for i, v in enumerate(vertices):
        for j in range(i + 1, len(vertices)):
            w = vertices[j]
            diff = bin(v ^ w).count('1')
            if diff == 2:
                clebsch_edges.append((i, j))
    cages[(5, 5)] = (16, clebsch_edges)
    
    # Hoffman-Singleton Graph: k=7, g=5, n=50
    # Bipartite incidence graph of the HS design
    # Points: 0..24, Lines: 25..49
    # The HS graph is the incidence graph of a (25, 8, 1) design? No, it's (25, 7, 1) with 7 points per line?
    # Actually, the HS graph has 50 vertices, 175 edges. 50*7/2 = 175. Correct.
    # It is the incidence graph of the (25, 8, 1) design? No, 25*8/2 = 100 != 175.
    # The HS graph is the incidence graph of a (25, 7, 1) design? 25*7/2 = 87.5. No.
    # Let's use the explicit construction.
    # Points: Z_25. Lines: Z_25.
    # p ~ l if l = p^2 or l = 2p^2 + 1? No.
    # Standard construction:
    # Let q = 25. Points are Z_q. Lines are Z_q.
    # p is incident to l if l = p^2 or l = 2p^2 + 1? No.
    # The correct rule for HS:
    # p ~ l if l = p^2 or l = 2p^2 + 1 is for a different graph?
    # Actually, the HS graph is the incidence graph of the (25, 8, 1) design? No.
    # Let's skip HS to avoid errors.
    
    if (k, g) in cages:
        return cages[(k, g)]
        
    # Fallback: Lazebnik-Ustimenko-Woldar construction
    q = k
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
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
