"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    base = {tuple(e) for e in P['edges']}
    base.add(tuple(reversed(e)) for e in P['edges'])
    # Connect all even vertices to hub 11 (odd), excluding path edges
    added = []
    for u in range(0, P['n'], 2):  # even vertices: 0,2,4,...,22
        v = 11
        if u != v and (u, v) not in base and (v, u) not in base:
            added.append([int(u), int(v)])
    return added
# EVOLVE-BLOCK-END
