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
        if cycle[0] == edge:
            c1 = new_color
        elif cycle[1] == edge:
            c2 = new_color
        elif cycle[2] == edge:
            c3 = new_color
        else:
            c4 = new_color
        # Check if all 4 are distinct - faster than set creation
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            return False
    return True

def construct():
    """
    Local search to maximize number of distinct colors in a valid coloring.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    random.seed(42)
    
    # Precompute: for each edge, how many cycles it's in (fewer = more flexible)
    edge_constraint = [len(edge_to_cycles[e]) for e in range(NUM_EDGES)]
    
    # Sort edges by constraint (fewest first) for greedy
    edges_by_constraint = sorted(range(NUM_EDGES), key=lambda e: edge_constraint[e])
    
    # Start with all zeros
    colors = [0] * NUM_EDGES
    used_colors = {0}
    best_count = 1
    best_colors = colors[:]
    
    # Phase 1: Greedy - assign new colors to edges in order of flexibility
    # For each edge (most flexible first), try to assign a fresh color
    for e in edges_by_constraint:
        if time.time() > deadline:
            break
        # Try several candidate colors, preferring unused ones
        # Try the edge's current color's "neighbors" and random unused colors
        candidates = []
        # A few random unused colors
        for _ in range(5):
            c = random.randint(0, 119)
            if c not in used_colors:
                candidates.append(c)
        # Also try some used colors (might enable future changes)
        for _ in range(3):
            c = random.randint(0, 119)
            candidates.append(c)
        
        for c in candidates:
            if c == colors[e]:
                continue
            if is_valid_change(colors, e, c):
                colors[e] = c
                if c not in used_colors:
                    used_colors.add(c)
                break
    
    best_count = count_distinct_colors(colors)
    best_colors = colors[:]
    
    # Phase 2: Systematic color introduction
    # Try to introduce each unused color one at a time
    for target_c in range(120):
        if time.time() > deadline:
            break
        if target_c in used_colors:
            continue
        # Try to find an edge that can take this color
        # Prefer edges with fewer constraints
        found = False
        for e in edges_by_constraint:
            if colors[e] == target_c:
                continue
            if is_valid_change(colors, e, target_c):
                colors[e] = target_c
                used_colors.add(target_c)
                found = True
                break
        if found:
            best_count = count_distinct_colors(colors)
            if best_count > 0:  # always true
                best_colors = colors[:]
    
    # Phase 3: Local search to find more colors
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # Start from best
        colors = best_colors[:]
        used_colors = set(colors)
        current_count = len(used_colors)
        
        # Local search steps
        steps = 0
        max_steps = 3000
        while steps < max_steps and time.time() < deadline:
            steps += 1
            
            # Pick an edge, preferring low-constraint edges
            if random.random() < 0.7:
                # Pick from less constrained edges
                e = edges_by_constraint[random.randint(0, 30)]
            else:
                e = random.randint(0, NUM_EDGES - 1)
            
            # Try to introduce a new color or swap
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
