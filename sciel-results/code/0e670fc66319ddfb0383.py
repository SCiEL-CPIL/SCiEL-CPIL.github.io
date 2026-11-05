import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + j - i - 1

# Precompute edge indices for all pairs for fast lookup
_edge_idx = [[0] * N for _ in range(N)]
for i in range(N):
    for j in range(i + 1, N):
        _edge_idx[i][j] = edge_idx(i, j)
        _edge_idx[j][i] = _edge_idx[i][j]

# Build all 4-cycles
# For 4 vertices a < b < c < d, the 3 distinct 4-cycles are:
# 1. a-b-c-d-a
# 2. a-b-d-c-a
# 3. a-c-b-d-a
def build_four_cycles():
    cycles = []
    for verts in combinations(range(N), 4):
        a, b, c, d = verts
        # Cycle 1
        cycles.append((_edge_idx[a][b], _edge_idx[b][c], _edge_idx[c][d], _edge_idx[d][a]))
        # Cycle 2
        cycles.append((_edge_idx[a][b], _edge_idx[b][d], _edge_idx[d][c], _edge_idx[c][a]))
        # Cycle 3
        cycles.append((_edge_idx[a][c], _edge_idx[c][b], _edge_idx[b][d], _edge_idx[d][a]))
    return cycles

FOUR_CYCLES = build_four_cycles()
NUM_CYCLES = len(FOUR_CYCLES)

# For each edge, precompute the other 3 edges in each cycle it participates in.
# This allows O(1) lookup for the neighbors in a specific cycle constraint.
edge_other_edges_in_cycle = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(FOUR_CYCLES):
    for i, e in enumerate(cycle):
        others = [cycle[j] for j in range(4) if j != i]
        edge_other_edges_in_cycle[e].append(others)

def count_distinct_colors(colors):
    return len(set(colors))

def is_valid_change(colors, edge, new_color):
    """
    Check if changing 'edge' to 'new_color' maintains validity.
    Validity: No 4-cycle has 4 distinct colors.
    """
    old_color = colors[edge]
    if old_color == new_color:
        return True
    
    # Check all cycles containing this edge
    for others in edge_other_edges_in_cycle[edge]:
        c1 = colors[others[0]]
        c2 = colors[others[1]]
        c3 = colors[others[2]]
        
        # The 4 colors in this cycle would be: new_color, c1, c2, c3
        # We need to check if they are all distinct.
        # If any of c1, c2, c3 equals new_color, they are not all distinct.
        # If any of c1, c2, c3 equals another, they are not all distinct.
        
        if new_color != c1 and new_color != c2 and new_color != c3:
            # Check if c1, c2, c3 are distinct
            if c1 != c2 and c1 != c3 and c2 != c3:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    random.seed(42)
    
    # Initial greedy construction
    # Try to assign colors to edges in order, preferring unused colors
    colors = [0] * NUM_EDGES
    used_colors = {0}
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # Try 5 different greedy initializations with different random orders
    for trial in range(5):
        if time.time() > deadline - 10: break
        
        random.seed(100 + trial)
        c_trial = [0] * NUM_EDGES
        used = {0}
        
        # Random order for greedy
        order = list(range(NUM_EDGES))
        random.shuffle(order)
        
        for e in order:
            # Try to find a new color
            found_new = False
            # Try random unused colors
            unused = [c for c in range(MAX_COLOR + 1) if c not in used]
            random.shuffle(unused)
            
            for new_c in unused[:20]: # Limit attempts for speed
                if is_valid_change(c_trial, e, new_c):
                    c_trial[e] = new_c
                    used.add(new_c)
                    found_new = True
                    break
            
            if not found_new:
                # If no new color works, pick a color that is "safe"
                # Usually, keeping 0 is safe, but might limit future diversity.
                # Let's try to match a neighbor to reduce constraint tightness.
                if edge_other_edges_in_cycle[e]:
                    others = edge_other_edges_in_cycle[e][0]
                    c_trial[e] = c_trial[others[0]]
                else:
                    c_trial[e] = 0
        
        count = count_distinct_colors(c_trial)
        if count > best_count:
            best_count = count
            best_colors = c_trial[:]
            
    # Local Search Phase
    current_colors = best_colors[:]
    current_count = best_count
    best_colors_ls = current_colors[:]
    best_count_ls = current_count
    
    no_improve_steps = 0
    max_no_improve = 2000
    steps = 0
    
    while time.time() < deadline - 2:
        steps += 1
        if no_improve_steps > max_no_improve:
            # Restart from best
            current_colors = best_colors_ls[:]
            current_count = best_count_ls
            no_improve_steps = 0
            # Optional: slight perturbation
            if steps % 1000 == 0:
                e_rand = random.randint(0, NUM_EDGES - 1)
                c_rand = random.randint(0, MAX_COLOR)
                if is_valid_change(current_colors, e_rand, c_rand):
                    current_colors[e_rand] = c_rand
                    current_count = count_distinct_colors(current_colors)
            continue
        
        e = random.randint(0, NUM_EDGES - 1)
        
        # Strategy:
        # 1. If current edge has a "common" color, try to change to a rare/new color.
        # 2. Try to introduce a new color if possible.
        
        # Pick a target color
        # 70% chance: try a new color (unused)
        # 30% chance: try a random existing color (to move around and escape local optima)
        
        if random.random() < 0.7:
            used = set(current_colors)
            unused = [c for c in range(MAX_COLOR + 1) if c not in used]
            if not unused:
                # All colors used, we are likely optimal or stuck at high count
                new_c = random.randint(0, MAX_COLOR)
            else:
                new_c = random.choice(unused)
        else:
            new_c = random.randint(0, MAX_COLOR)
            
        if new_c == current_colors[e]:
            continue
            
        if is_valid_change(current_colors, e, new_c):
            old_c = current_colors[e]
            current_colors[e] = new_c
            new_count = count_distinct_colors(current_colors)
            
            if new_count > current_count:
                current_count = new_count
                no_improve_steps = 0
                if current_count > best_count_ls:
                    best_count_ls = current_count
                    best_colors_ls = current_colors[:]
            else:
                # Revert if it didn't improve (Hill Climbing)
                current_colors[e] = old_c
                no_improve_steps += 1
        else:
            no_improve_steps += 1

    return best_colors_ls
