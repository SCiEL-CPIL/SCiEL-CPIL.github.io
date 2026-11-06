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

# Precompute all 4-cycles
all_cycles = []
for a, b, c, d in combinations(range(N), 4):
    e1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
    all_cycles.append(e1)
    e2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
    all_cycles.append(e2)
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
    for cycle in all_cycles:
        c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def check_edge_change(colors, edge, new_color):
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
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    # Start with star construction (16 colors)
    colors = [0] * NUM_EDGES
    for j in range(1, N):
        idx = edge_idx(0, j)
        colors[idx] = j - 1  # colors 0..14
    # All other edges are 0... wait, that conflicts. Let me use a base color.
    
    # Actually let me restart: use color 15 as base, and 0..14 for star
    colors = [15] * NUM_EDGES
    for j in range(1, N):
        idx = edge_idx(0, j)
        colors[idx] = j - 1
    best_colors = colors[:]
    best_count = count_colors(best_colors)  # 16
    
    # Local search: try to increase color count
    # Strategy: pick an edge, try to give it a new unique color
    # If that fails, try reassigning other edges to free up a color
    
    random.seed(42)
    
    for iteration in range(2000000):
        if time.time() > deadline:
            break
        
        # Try to add a new color to a random edge
        edge = random.randint(0, NUM_EDGES - 1)
        current_color = colors[edge]
        
        # Try a new color (one not currently used, or a high number)
        used_colors = set(colors)
        # Try to assign a fresh color
        new_color = 100 + random.randint(0, 50)  # fresh color range
        
        if check_edge_change(colors, edge, new_color):
            colors[edge] = new_color
            new_count = count_colors(colors)
            if new_count > best_count:
                best_count = new_count
                best_colors = colors[:]
        
        # Also try to change an edge to a different existing color to potentially
        # free up a color for another edge
        if random.random() < 0.3:
            edge = random.randint(0, NUM_EDGES - 1)
            # Try a random different color
            new_color = random.randint(0, 119)
            if new_color != colors[edge]:
                if check_edge_change(colors, edge, new_color):
                    colors[edge] = new_color
                    new_count = count_colors(colors)
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = colors[:]
    
    return best_colors
