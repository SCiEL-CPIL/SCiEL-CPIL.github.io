import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def build_edge_index():
    idx = [[-1]*N for _ in range(N)]
    counter = 0
    for i in range(N):
        for j in range(i+1, N):
            idx[i][j] = counter
            idx[j][i] = counter
            counter += 1
    return idx

EDGE_IDX = build_edge_index()

def build_cycles():
    cycles = []
    for a in range(N-3):
        for b in range(a+1, N-2):
            for c in range(b+1, N-1):
                for d in range(c+1, N):
                    cycles.append((EDGE_IDX[a][b], EDGE_IDX[b][c], EDGE_IDX[c][d], EDGE_IDX[d][a]))
                    cycles.append((EDGE_IDX[a][b], EDGE_IDX[b][d], EDGE_IDX[d][c], EDGE_IDX[c][a]))
                    cycles.append((EDGE_IDX[a][c], EDGE_IDX[c][b], EDGE_IDX[b][d], EDGE_IDX[d][a]))
    return cycles

CYCLES = build_cycles()
NUM_CYCLES = len(CYCLES)

EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(CYCLES):
    for e in cycle:
        EDGE_CYCLES[e].append(ci)

def is_valid(colors):
    for cycle in CYCLES:
        if len(set(colors[e] for e in cycle)) > 3:
            return False
    return True

def can_take_color(colors, edge, new_color):
    """Check if assigning new_color to edge is valid (no 4-cycle gets 4 distinct colors)."""
    for ci in EDGE_CYCLES[edge]:
        cycle = CYCLES[ci]
        c0 = new_color if cycle[0] == edge else colors[cycle[0]]
        c1 = new_color if cycle[1] == edge else colors[cycle[1]]
        c2 = new_color if cycle[2] == edge else colors[cycle[2]]
        c3 = new_color if cycle[3] == edge else colors[cycle[3]]
        # Check if all four are distinct
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def local_search(colors, deadline):
    colors = list(colors)
    used_colors = set(colors)
    best_score = len(used_colors)
    best_colors = list(colors)
    
    no_improve_count = 0
    max_no_improve = 50
    
    all_colors = list(range(120))
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Kick: randomly recolor several edges to escape local optimum
            num_kicks = random.randint(5, 15)
            for _ in range(num_kicks):
                e = random.randint(0, NUM_EDGES - 1)
                shuffled_colors = all_colors[:]
                random.shuffle(shuffled_colors)
                for nc in shuffled_colors:
                    if can_take_color(colors, e, nc):
                        colors[e] = nc
                        break
            
            used_colors = set(colors)
            no_improve_count = 0
            max_no_improve = random.randint(50, 150)
            continue

        # Try to improve: find an edge that can take an unused color
        improved = False
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        
        for e in edges:
            if time.time() >= deadline:
                break
            
            # Try unused colors first (shuffled for diversity)
            unused = [c for c in range(120) if c not in used_colors]
            if not unused:
                break # No unused colors left to try
            
            random.shuffle(unused)
            
            found = False
            for nc in unused:
                if can_take_color(colors, e, nc):
                    colors[e] = nc
                    used_colors.add(nc)
                    improved = True
                    found = True
                    break
            
            if found:
                no_improve_count = 0
                if len(used_colors) > best_score:
                    best_score = len(used_colors)
                    best_colors = list(colors)
            else:
                no_improve_count += 1
        
        if not improved:
            no_improve_count += 1

    return best_colors, best_score

def recursive_split_coloring(vertices, next_color, colors, edge_idx):
    """
    Recursively split vertices and assign a new color to cross edges.
    Returns the next available color index.
    """
    if len(vertices) <= 1:
        return next_color
    
    mid = len(vertices) // 2
    left = vertices[:mid]
    right = vertices[mid:]
    
    # Assign current color to all edges between left and right
    for u in left:
        for v in right:
            e = edge_idx[u][v]
            colors[e] = next_color
    
    next_color += 1
    next_color = recursive_split_coloring(left, next_color, colors, edge_idx)
    next_color = recursive_split_coloring(right, next_color, colors, edge_idx)
    
    return next_color

def construct():
    deadline = time.time() + 55  # 55s safety
    
    best_colors = [0] * NUM_EDGES
    best_score = 0
    
    # Strategy 1: Recursive Split (Valid, ~15 colors)
    rs_colors = [0] * NUM_EDGES
    next_c = recursive_split_coloring(list(range(N)), 0, rs_colors, EDGE_IDX)
    if is_valid(rs_colors):
        best_colors = rs_colors
        best_score = len(set(rs_colors))

    # Strategy 2: Star Coloring (Valid, 16 colors)
    star_colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i+1, N):
            e = EDGE_IDX[i][j]
            if i == 0:
                star_colors[e] = j
            else:
                star_colors[e] = 0
    if is_valid(star_colors):
        if len(set(star_colors)) > best_score:
            best_colors = star_colors
            best_score = len(set(star_colors))

    # Strategy 3: Local Search from Star
    # Allocate most time to this
    remaining = deadline - time.time()
    ls_deadline = time.time() + remaining * 0.8
    
    improved_colors, score = local_search(star_colors, ls_deadline)
    if score > best_score:
        best_score = score
        best_colors = improved_colors

    # Strategy 4: Local Search from Recursive Split
    if time.time() < deadline:
        ls_deadline2 = time.time() + (deadline - time.time()) * 0.5
        improved_colors2, score2 = local_search(rs_colors, ls_deadline2)
        if score2 > best_score:
            best_score = score2
            best_colors = improved_colors2

    # Final verification
    if not is_valid(best_colors):
        return [0] * NUM_EDGES
        
    return best_colors

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
