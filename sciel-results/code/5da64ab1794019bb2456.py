import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge indices
edge_to_idx = {}
edges = []
for i in range(N):
    for j in range(i + 1, N):
        edge_to_idx[(i, j)] = len(edges)
        edges.append((i, j))

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return edge_to_idx[(i, j)]

# Precompute all 4-cycles: each is a tuple of 4 edge indices
# For each 4-vertex set {a,b,c,d}, the 3 cycles are:
# (a,b,c,d): ab, bc, cd, da
# (a,b,d,c): ab, bd, dc, ca  
# (a,c,b,d): ac, cb, bd, da
all_cycles = []
for a, b, c, d in combinations(range(N), 4):
    # Cycle 1: a-b-c-d-a
    e1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
    all_cycles.append(e1)
    # Cycle 2: a-b-d-c-a
    e2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
    all_cycles.append(e2)
    # Cycle 3: a-c-b-d-a
    e3 = (edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a))
    all_cycles.append(e3)

# For each edge, which cycles contain it
edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for idx, cycle in enumerate(all_cycles):
    for e in cycle:
        edge_to_cycles[e].append(idx)

def count_colors(colors):
    return len(set(colors))

def is_valid(colors):
    """Check if no 4-cycle has 4 distinct colors."""
    for cycle in all_cycles:
        c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def check_edge_change(colors, edge, new_color):
    """Check if setting edge to new_color would violate any constraint."""
    old_color = colors[edge]
    if old_color == new_color:
        return True
    colors[edge] = new_color
    valid = True
    for cycle_idx in edge_to_cycles[edge]:
        cycle = all_cycles[cycle_idx]
        c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        if len({c0, c1, c2, c3}) == 4:
            valid = False
            break
    colors[edge] = old_color
    return valid

def construct():
    # Star construction:
    # Center at vertex 0.
    # Edges (0, j) for j in 1..15 get distinct colors 0..14.
    # All other edges get color 15.
    # Total colors: 16.
    
    colors = [15] * NUM_EDGES
    for j in range(1, N):
        idx = edge_idx(0, j)
        colors[idx] = j - 1
        
    return colors
