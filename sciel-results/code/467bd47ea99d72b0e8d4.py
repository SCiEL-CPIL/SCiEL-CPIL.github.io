import time
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def _edge_idx(i, j):
    # lexicographic order of pairs (i,j) with 0<=i<j<N
    # index = sum_{k=0}^{i-1} (N-1-k) + (j - i - 1)
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def _pairs_list():
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
    return pairs

PAIRS = _pairs_list()
EDGE_IDX = {p: idx for idx, p in enumerate(PAIRS)}

# Precompute for each edge, the list of 4-cycles it participates in
# A 4-cycle is a set of 4 vertices {a,b,c,d} with 3 possible cycles.
# For constraint checking: for each 4-vertex subset, check all 3 cycles.

def _build_cycle_constraints():
    """For each edge, find all 4-cycles containing it.
    A 4-cycle on vertices {a,b,c,d} (sorted) has 3 distinct cycles:
    a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
    Each cycle uses 4 edges.
    We need: for each edge e, list of (cycle_id, other_3_edges)
    But simpler: for each 4-set, store the 3 cycles as tuples of 4 edge indices.
    """
    all_cycles = []  # list of (edge_idx1, edge_idx2, edge_idx3, edge_idx4)
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles on 4 vertices:
        # Cycle 1: a-b-c-d-a  edges: (a,b),(b,c),(c,d),(d,a)
        e1 = (_edge_idx(a,b), _edge_idx(b,c), _edge_idx(c,d), _edge_idx(a,d))
        # Cycle 2: a-b-d-c-a  edges: (a,b),(b,d),(d,c),(c,a)
        e2 = (_edge_idx(a,b), _edge_idx(b,d), _edge_idx(c,d), _edge_idx(a,c))
        # Cycle 3: a-c-b-d-a  edges: (a,c),(c,b),(b,d),(d,a)
        e3 = (_edge_idx(a,c), _edge_idx(b,c), _edge_idx(b,d), _edge_idx(a,d))
        all_cycles.extend([e1, e2, e3])
    return all_cycles

ALL_CYCLES = _build_cycle_constraints()
NUM_CYCLES = len(ALL_CYCLES)

# For fast checking: for each edge, which cycles does it appear in?
edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(ALL_CYCLES):
    for e in cycle:
        edge_to_cycles[e].append(ci)

def _is_valid(coloring):
    """Check no 4-cycle has 4 distinct colors."""
    for cycle in ALL_CYCLES:
        c0 = coloring[cycle[0]]
        c1 = coloring[cycle[1]]
        c2 = coloring[cycle[2]]
        c3 = coloring[cycle[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def _check_edge_change(coloring, edge, new_color):
    """Check if changing coloring[edge] to new_color keeps validity.
    Only need to check cycles containing this edge."""
    old_color = coloring[edge]
    if old_color == new_color:
        return True
    for ci in edge_to_cycles[edge]:
        cycle = ALL_CYCLES[ci]
        colors = set()
        for e in cycle:
            if e == edge:
                colors.add(new_color)
            else:
                colors.add(coloring[e])
        if len(colors) == 4:
            return False
    return True

def _base_coloring():
    """Color edge (i,j) with color i. Gives N-1 = 15 colors."""
    coloring = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(PAIRS):
        coloring[idx] = i
    return coloring

def _local_search(coloring, deadline):
    """Try to increase number of distinct colors via local search."""
    coloring = list(coloring)
    used_colors = set(coloring)
    current_count = len(used_colors)
    
    # Precompute: for each edge, what colors are "safe" to try
    # We'll try to introduce new colors and also shuffle existing ones.
    
    next_color = max(used_colors) + 1 if used_colors else 0
    max_color_id = 119
    
    # Build adjacency: for each vertex, its incident edges
    vertex_edges = [[] for _ in range(N)]
    for idx, (i, j) in enumerate(PAIRS):
        vertex_edges[i].append(idx)
        vertex_edges[j].append(idx)
    
    # Local search: repeatedly try to recolor an edge to a new or different color
    # that increases the palette or maintains it without violating constraints.
    
    max_iterations = 50000
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        if next_color > max_color_id:
            break
        
        improved = False
        # Try each edge, in random-ish order (use deterministic shuffle based on iteration)
        # For speed, try a subset of edges per iteration
        for e in range(NUM_EDGES):
            if time.time() > deadline:
                break
            # Try to assign next_color to edge e
            if _check_edge_change(coloring, e, next_color):
                coloring[e] = next_color
                if next_color not in used_colors:
                    used_colors.add(next_color)
                    current_count += 1
                next_color += 1
                improved = True
                break
        
        if not improved:
            # Try to recolor edges to shuffle colors, possibly enabling new colors
            # Try changing an edge to a different existing color to free up a "slot"
            found = False
            for e in range(NUM_EDGES):
                if time.time() > deadline:
                    break
                old_c = coloring[e]
                # Try a few alternative colors
                for alt in range(max_color_id + 1):
                    if alt == old_c:
                        continue
                    if _check_edge_change(coloring, e, alt):
                        coloring[e] = alt
                        found = True
                        break
                if found:
                    break
            if not found:
                # Try introducing next_color with a different strategy
                # Maybe we need to first recolor some edge to make room
                # For now, break if we can't improve
                break
    
    return coloring, len(set(coloring))

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Start with base coloring
    coloring = _base_coloring()
    
    # Run local search to increase color count
    coloring, count = _local_search(coloring, deadline)
    
    # Convert to list of ints
    return [int(c) for c in coloring]
