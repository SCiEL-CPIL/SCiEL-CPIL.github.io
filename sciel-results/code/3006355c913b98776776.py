"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    """Return the index of edge (i,j) in the lexicographically ordered list."""
    return i * (N - 1) - i * (i - 1) // 2 + j - i - 1

# Precompute edge indices for all pairs
_edge_idx = [[0] * N for _ in range(N)]
for i in range(N):
    for j in range(i + 1, N):
        _edge_idx[i][j] = edge_idx(i, j)
        _edge_idx[j][i] = _edge_idx[i][j]

def build_four_cycles():
    """Build list of all 4-cycles, each as a tuple of 4 edge indices."""
    cycles = []
    for verts in combinations(range(N), 4):
        a, b, c, d = verts
        # Cycle 1: a-b-c-d-a
        cycles.append((
            _edge_idx[a][b],
            _edge_idx[b][c],
            _edge_idx[c][d],
            _edge_idx[d][a]
        ))
        # Cycle 2: a-b-d-c-a
        cycles.append((
            _edge_idx[a][b],
            _edge_idx[b][d],
            _edge_idx[d][c],
            _edge_idx[c][a]
        ))
        # Cycle 3: a-c-b-d-a
        cycles.append((
            _edge_idx[a][c],
            _edge_idx[c][b],
            _edge_idx[b][d],
            _edge_idx[d][a]
        ))
    return cycles

FOUR_CYCLES = build_four_cycles()
NUM_CYCLES = len(FOUR_CYCLES)  # C(16,4)*3 = 1820*3 = 5460

# For each edge, precompute which cycles it participates in
edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(FOUR_CYCLES):
    for e in cycle:
        edge_to_cycles[e].append(ci)

def is_valid_change(colors, edge, new_color):
    """Check if changing edge to new_color maintains validity."""
    old_color = colors[edge]
    if old_color == new_color:
        return True
    
    for ci in edge_to_cycles[edge]:
        cycle = FOUR_CYCLES[ci]
        c1, c2, c3, c4 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        if cycle[0] == edge:
            c1 = new_color
        elif cycle[1] == edge:
            c2 = new_color
        elif cycle[2] == edge:
            c3 = new_color
        else:
            c4 = new_color
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            return False
    return True

def greedy_coloring(edge_order, deadline):
    """
    Greedy coloring: process edges in given order, assign smallest valid color.
    Returns the color array.
    """
    colors = [-1] * NUM_EDGES
    used_colors = set()
    
    for e in edge_order:
        if time.time() > deadline:
            # Fill remaining with 0 to ensure validity
            for i in range(len(edge_order)):
                if colors[edge_order[i]] == -1:
                    colors[edge_order[i]] = 0
            break
        # Try colors 0, 1, 2, ... until we find a valid one
        found = False
        for c in range(120):
            if is_valid_change(colors, e, c):
                colors[e] = c
                used_colors.add(c)
                found = True
                break
        if not found:
            colors[e] = 0
            used_colors.add(0)
            
    # Safety: fill any -1 with 0
    for i in range(NUM_EDGES):
        if colors[i] == -1:
            colors[i] = 0
            
    return colors, used_colors

def construct():
    """
    Greedy edge-coloring with multiple orderings to maximize distinct colors.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    best_count = 0
    best_colors = [0] * NUM_EDGES
    
    # Strategy 1: Lexicographic order (natural)
    edge_order = list(range(NUM_EDGES))
    colors, used = greedy_coloring(edge_order, deadline)
    cnt = len(set(colors))
    if cnt > best_count:
        best_count = cnt
        best_colors = colors[:]
    
    # Strategy 2: Reverse lexicographic
    edge_order = list(range(NUM_EDGES - 1, -1, -1))
    if time.time() < deadline:
        colors, used = greedy_coloring(edge_order, deadline)
        cnt = len(set(colors))
        if cnt > best_count:
            best_count = cnt
            best_colors = colors[:]
    
    # Strategy 3: Sort by degree/constraint (edges with fewer 4-cycles first)
    edge_constraint = [len(edge_to_cycles[e]) for e in range(NUM_EDGES)]
    edge_order = sorted(range(NUM_EDGES), key=lambda e: edge_constraint[e])
    if time.time() < deadline:
        colors, used = greedy_coloring(edge_order, deadline)
        cnt = len(set(colors))
        if cnt > best_count:
            best_count = cnt
            best_colors = colors[:]
    
    # Strategy 4: Reverse constraint order
    edge_order = sorted(range(NUM_EDGES), key=lambda e: -edge_constraint[e])
    if time.time() < deadline:
        colors, used = greedy_coloring(edge_order, deadline)
        cnt = len(set(colors))
        if cnt > best_count:
            best_count = cnt
            best_colors = colors[:]
    
    # Strategy 5: Random orderings
    random.seed(42)
    num_random_orders = 0
    while time.time() < deadline:
        num_random_orders += 1
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        colors, used = greedy_coloring(edge_order, deadline)
        cnt = len(set(colors))
        if cnt > best_count:
            best_count = cnt
            best_colors = colors[:]
    
    # Phase 2: Local search polish on best solution
    colors = best_colors[:]
    used_colors = set(colors)
    current_count = len(used_colors)
    
    # Try to introduce new colors via local search
    edges_by_constraint = sorted(range(NUM_EDGES), key=lambda e: edge_constraint[e])
    
    steps = 0
    max_steps = 5000
    while steps < max_steps and time.time() < deadline:
        steps += 1
        # Pick an edge
        if random.random() < 0.7:
            e = edges_by_constraint[random.randint(0, 30)]
        else:
            e = random.randint(0, NUM_EDGES - 1)
        
        # Try a random new color
        new_c = random.randint(0, 119)
        if new_c == colors[e]:
            continue
        if is_valid_change(colors, e, new_c):
            old_c = colors[e]
            colors[e] = new_c
            new_used = set(colors)
            new_count = len(new_used)
            if new_count > current_count:
                current_count = new_count
                used_colors = new_used
                if current_count > best_count:
                    best_count = current_count
                    best_colors = colors[:]
            else:
                colors[e] = old_c
    
    return best_colors

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
