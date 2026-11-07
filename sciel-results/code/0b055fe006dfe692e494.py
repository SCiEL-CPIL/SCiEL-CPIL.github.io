"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_undirected = set()
    for u, v in P['edges']:
        base_undirected.add((u, v))
        base_undirected.add((v, u))
    
    # Hub at 11 (odd), connect to all even vertices not adjacent on path
    hub = 11
    added = []
    for e in range(0, n, 2):  # even vertices
        if e == hub - 1 or e == hub + 1:
            continue
        if (hub, e) not in base_undirected:
            added.append([hub, e])
    
    return added
# EVOLVE-BLOCK-END
