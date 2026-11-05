"""Optimized coloring search for K16 with no rainbow C4."""
import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

# Precompute edge indices
def edge_idx(i, j):
    return i * (N - 1) - i * (i - 1) // 2 + j - i - 1

_edge_idx = [[0] * N for _ in range(N)]
for i in range(N):
    for j in range(i + 1, N):
        _edge_idx[i][j] = edge_idx(i, j)
        _edge_idx[j][i] = _edge_idx[i][j]

# Build all 4-cycles
def build_four_cycles():
    cycles = []
    for verts in combinations(range(N), 4):
        a, b, c, d = verts
        cycles.append((_edge_idx[a][b], _edge_idx[b][c], _edge_idx[c][d], _edge_idx[d][a]))
        cycles.append((_edge_idx[a][b], _edge_idx[b][d], _edge_idx[d][c], _edge_idx[c][a]))
        cycles.append((_edge_idx[a][c], _edge_idx[c][b], _edge_idx[b][d], _edge_idx[d][a]))
    return cycles

FOUR_CYCLES = build_four_cycles()
NUM_CYCLES = len(FOUR_CYCLES)

# For each edge, which cycles it's in
edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(FOUR_CYCLES):
    for e in cycle:
        edge_to_cycles[e].append(ci)

# For fast validity check: precompute cycle positions
cycle_edge_pos = [[0]*4 for _ in range(NUM_CYCLES)]
for ci, cycle in enumerate(FOUR_CYCLES):
    for pos, e in enumerate(cycle):
        cycle_edge_pos[ci][pos] = e

def is_valid_change(colors, edge, new_color):
    """Check if setting edge to new_color keeps all cycles valid."""
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

def count_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    random.seed(12345)
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # Phase 1: Greedy build - assign colors to edges one by one
    # Try to use a new color for each edge if possible
    colors = [0] * NUM_EDGES
    used = set()
    
    # Random edge order for greedy
    edge_order = list(range(NUM_EDGES))
    random.shuffle(edge_order)
    
    for e in edge_order:
        if time.time() > deadline:
            break
        # Try new colors first (unused), then any valid color
        tried = set()
        for attempt in range(30):
            if attempt < 15:
                # Try unused colors
                c = random.randint(0, MAX_COLOR)
                if c in used:
                    continue
            else:
                c = random.randint(0, MAX_COLOR)
            if c in tried:
                continue
            tried.add(c)
            if is_valid_change(colors, e, c):
                colors[e] = c
                used.add(c)
                break
    
    cur_count = count_colors(colors)
    if cur_count > best_count:
        best_count = cur_count
        best_colors = colors[:]
    
    # Phase 2: Local search to improve
    # Strategy: randomly pick an edge, try to change it to a new color
    # Accept if it increases color count or is neutral (for diversity)
    
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        if restart < 3:
            colors = best_colors[:]
        elif restart < 10:
            # Perturb best
            colors = best_colors[:]
            for _ in range(random.randint(5, 20)):
                e = random.randint(0, NUM_EDGES - 1)
                c = random.randint(0, MAX_COLOR)
                if is_valid_change(colors, e, c):
                    colors[e] = c
        else:
            # Fresh random start with some structure
            colors = [0] * NUM_EDGES
            # Assign a few colors
            for _ in range(random.randint(10, 40)):
                e = random.randint(0, NUM_EDGES - 1)
                c = random.randint(0, MAX_COLOR)
                if is_valid_change(colors, e, c):
                    colors[e] = c
        
        cur_count = count_colors(colors)
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
        
        # Local search steps
        steps = 0
        max_steps = 3000
        while steps < max_steps and time.time() < deadline:
            steps += 1
            e = random.randint(0, NUM_EDGES - 1)
            
            # Try to introduce a new color
            # Sample from unused colors preferentially
            current_used = set(colors)
            if len(current_used) < MAX_COLOR + 1:
                # Pick an unused color
                unused = [c for c in range(MAX_COLOR + 1) if c not in current_used]
                if unused:
                    new_c = random.choice(unused[:50]) if len(unused) > 50 else random.choice(unused)
                else:
                    new_c = random.randint(0, MAX_COLOR)
            else:
                new_c = random.randint(0, MAX_COLOR)
            
            if new_c == colors[e]:
                continue
            
            if is_valid_change(colors, e, new_c):
                old_c = colors[e]
                colors[e] = new_c
                new_count = count_colors(colors)
                if new_count >= cur_count:
                    cur_count = new_count
                    if cur_count > best_count:
                        best_count = cur_count
                        best_colors = colors[:]
                else:
                    colors[e] = old_c
    
    # Phase 3: Final polish - try to add colors to specific edges
    if time.time() < deadline - 2.0:
        colors = best_colors[:]
        cur_count = best_count
        used_set = set(colors)
        
        # For each edge, try all unused colors
        for e in range(NUM_EDGES):
            if time.time() > deadline:
                break
            for c in range(MAX_COLOR + 1):
                if c in used_set:
                    continue
                if is_valid_change(colors, e, c):
                    colors[e] = c
                    used_set.add(c)
                    cur_count += 1
                    if cur_count > best_count:
                        best_count = cur_count
                        best_colors = colors[:]
    
    return best_colors

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
