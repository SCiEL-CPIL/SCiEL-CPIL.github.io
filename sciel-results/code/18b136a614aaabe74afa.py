"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    """Return the index of edge (i,j) in the lexicographically ordered list."""
    # For vertices 0..N-1, edge (i,j) with i<j
    # Index = sum_{k=0}^{i-1} (N-1-k) + (j - i - 1)
    # = i*(N-1) - i*(i-1)//2 + j - i - 1
    # Let's precompute for speed
    return i * (N - 1) - i * (i - 1) // 2 + j - i - 1

# Precompute edge indices for all pairs
_edge_idx = [[0] * N for _ in range(N)]
for i in range(N):
    for j in range(i + 1, N):
        _edge_idx[i][j] = edge_idx(i, j)
        _edge_idx[j][i] = _edge_idx[i][j]

# Precompute all 4-cycles: for each set of 4 vertices, there are 3 ways to form a 4-cycle
# A 4-cycle on vertices a,b,c,d (sorted) can be:
# (a,b,c,d), (a,b,d,c), (a,c,b,d) — but we need to be careful about what constitutes a "four-cycle"
# Actually, for 4 vertices a<b<c<d, the three distinct 4-cycles (as sets of 4 edges) are:
# 1. a-b-c-d-a: edges (a,b),(b,c),(c,d),(d,a)
# 2. a-b-d-c-a: edges (a,b),(b,d),(d,c),(c,a)
# 3. a-c-b-d-a: edges (a,c),(c,b),(b,d),(d,a)

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

def count_distinct_colors(colors):
    """Count number of distinct colors used."""
    return len(set(colors))

def is_valid_change(colors, edge, new_color):
    """Check if changing edge to new_color maintains validity."""
    old_color = colors[edge]
    if old_color == new_color:
        return True
    
    # Check all cycles containing this edge
    for ci in edge_to_cycles[edge]:
        cycle = FOUR_CYCLES[ci]
        # Get the 4 colors in this cycle
        c1, c2, c3, c4 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        # Replace the color of 'edge' with new_color
        # Find which position in the cycle is 'edge'
        for pos, e in enumerate(cycle):
            if e == edge:
                if pos == 0:
                    c1 = new_color
                elif pos == 1:
                    c2 = new_color
                elif pos == 2:
                    c3 = new_color
                elif pos == 3:
                    c4 = new_color
                break
        # Check if all 4 are distinct
        if len({c1, c2, c3, c4}) == 4:
            return False
    return True

def construct():
    """
    Local search to maximize number of distinct colors in a valid coloring.
    """
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Start with all zeros
    colors = [0] * NUM_EDGES
    best_colors = colors[:]
    best_count = 1
    
    # Phase 1: Greedy assignment - try to assign new colors to edges
    # For each edge, try to assign a new color not yet used
    used_colors = {0}
    
    # Try to increase color count by recoloring edges
    random.seed(42)
    
    # Greedy phase: for each edge in random order, try to assign a new color
    edges_order = list(range(NUM_EDGES))
    random.shuffle(edges_order)
    
    for e in edges_order:
        if time.time() > deadline - 10:
            break
        # Try colors from high to low (prefer new colors)
        for c in range(119, -1, -1):
            if is_valid_change(colors, e, c):
                if c not in used_colors:
                    colors[e] = c
                    used_colors.add(c)
                    break
    
    best_count = count_distinct_colors(colors)
    best_colors = colors[:]
    
    # Phase 2: Local search - try to improve by swapping colors or introducing new ones
    # Strategy: for each edge, try changing to a color not in its cycle neighbors
    # to potentially free up a color or introduce a new one
    
    # Build adjacency: for each edge, what other edges share a cycle with it
    # This helps us understand constraints
    
    # Local search with random restarts
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        
        # Start from best or random
        if num_restarts <= 3:
            colors = best_colors[:]
        else:
            # Random restart with some structure
            colors = [0] * NUM_EDGES
            # Assign some random colors
            for e in range(NUM_EDGES):
                c = random.randint(0, min(20, num_restarts))
                if is_valid_change(colors, e, c):
                    colors[e] = c
        
        current_count = count_distinct_colors(colors)
        
        # Local search steps
        steps = 0
        max_steps = 50000
        while steps < max_steps and time.time() < deadline - 5:
            steps += 1
            e = random.randint(0, NUM_EDGES - 1)
            
            # Try a few random new colors
            for _ in range(5):
                new_c = random.randint(0, 119)
                if new_c == colors[e]:
                    continue
                if is_valid_change(colors, e, new_c):
                    old_c = colors[e]
                    colors[e] = new_c
                    new_count = count_distinct_colors(colors)
                    if new_count > current_count:
                        current_count = new_count
                        if current_count > best_count:
                            best_count = current_count
                            best_colors = colors[:]
                        break
                    else:
                        # Accept with some probability (simulated annealing-like)
                        # For simplicity, reject if not improving
                        colors[e] = old_c
        
        if time.time() > deadline - 5:
            break
    
    return best_colors

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
