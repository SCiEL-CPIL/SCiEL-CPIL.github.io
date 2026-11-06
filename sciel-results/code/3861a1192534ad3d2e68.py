"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 16
M = N * (N - 1) // 2  # 120 edges

# Precompute edge indices
edge_idx = {}
for i in range(N):
    for j in range(i + 1, N):
        edge_idx[(i, j)] = i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def edge_idx_func(i, j):
    if i > j:
        i, j = j, i
    return edge_idx[(i, j)]

# Precompute all 4-cycles and which edges they use
# A 4-cycle is defined by 4 vertices in order: a-b-c-d-a
# Edges: (a,b), (b,c), (c,d), (d,a)
all_4cycles = []
for verts in combinations(range(N), 4):
    a, b, c, d = verts
    # Three distinct 4-cycles on these 4 vertices:
    # 1. a-b-c-d-a : edges ab, bc, cd, da
    # 2. a-b-d-c-a : edges ab, bd, dc, ca
    # 3. a-c-b-d-a : edges ac, cb, bd, da
    cycles = [
        [(a,b), (b,c), (c,d), (d,a)],
        [(a,b), (b,d), (d,c), (c,a)],
        [(a,c), (c,b), (b,d), (d,a)],
    ]
    for cyc in cycles:
        edge_list = []
        for (u, v) in cyc:
            if u > v:
                u, v = v, u
            edge_list.append(edge_idx[(u, v)])
        all_4cycles.append(edge_list)

# For each edge, precompute which 4-cycles contain it
edge_to_cycles = [[] for _ in range(M)]
for ci, cyc in enumerate(all_4cycles):
    for e in cyc:
        edge_to_cycles[e].append(ci)

# For fast checking: for a given cycle and its 4 edge indices, check if all 4 colors are distinct
def is_rainbow_cycle(cycle_edges, colors):
    """Check if the 4 edges in cycle_edges have all distinct colors."""
    c0 = colors[cycle_edges[0]]
    c1 = colors[cycle_edges[1]]
    c2 = colors[cycle_edges[2]]
    c3 = colors[cycle_edges[3]]
    if c0 == c1 or c0 == c2 or c0 == c3 or c1 == c2 or c1 == c3 or c2 == c3:
        return False
    return True

def count_distinct_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s
    
    best_result = [0] * M
    best_count = 1
    
    # Try multiple random seeds
    for seed in range(200):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        colors = [-1] * M
        used_colors = set()
        
        # Order edges: try to process edges in a way that constrains early
        # Use a random permutation for variety
        edge_order = list(range(M))
        random.shuffle(edge_order)
        
        valid = True
        for ei in edge_order:
            if time.time() > deadline:
                valid = False
                break
            
            # Try colors from 0 upward, preferring new colors
            # First try to assign a new color if possible
            assigned = False
            
            # Try new colors first (from high to low to encourage diversity)
            # Actually, let's try: first see if we can use a color not yet used
            # Then fall back to existing colors
            
            # Collect constraints: for each 4-cycle containing edge ei,
            # the other 3 edges must not all have distinct colors that differ from our choice
            # i.e., if other 3 edges have colors {x, y, z} all distinct, we can't use a 4th distinct color
            
            # For each cycle containing ei:
            #   Let other edges be e1, e2, e3 with colors c1, c2, c3
            #   If c1, c2, c3 are all distinct, then our color must be one of {c1, c2, c3}
            #   Otherwise, any color is fine
            
            forbidden_new = False
            allowed_colors = None  # if set, we must pick from this set
            
            for ci in edge_to_cycles[ei]:
                cyc = all_4cycles[ci]
                # Find the other 3 edges
                others = [cyc[k] for k in range(4) if cyc[k] != ei]
                other_colors = [colors[e] for e in others]
                
                # If any other edge is uncolored, no constraint yet
                if -1 in other_colors:
                    continue
                
                # Check if the 3 other colors are all distinct
                if len(set(other_colors)) == 3:
                    # Our color must be one of these 3
                    forbidden_new = True
                    if allowed_colors is None:
                        allowed_colors = set(other_colors)
                    else:
                        allowed_colors &= set(other_colors)
                        if not allowed_colors:
                            # Conflict - this edge can't be assigned
                            valid = False
                            break
            
            if not valid:
                break
            
            if allowed_colors is not None:
                # Must pick from allowed_colors
                # Prefer a color already used to minimize new colors? No, we want max colors.
                # Actually if constrained, we have no choice but to pick from allowed.
                # Pick the one that's already used if possible, else the smallest new one.
                # Wait - we want to MAXIMIZE distinct colors. If forced to pick from a set,
                # we should pick a color we've already used to save new colors for later.
                # But actually, picking a new color here doesn't hurt if we're forced.
                # Let's just pick the first available.
                candidates = sorted(allowed_colors)
                # Prefer colors already in used_colors to conserve
                for c in candidates:
                    if c in used_colors:
                        colors[ei] = c
                        assigned = True
                        break
                if not assigned:
                    colors[ei] = candidates[0]
                    used_colors.add(candidates[0])
                    assigned = True
            else:
                # No constraint - we can use any color. Prefer a new color to maximize diversity.
                # Try to use the smallest unused color
                new_color = 0
                while new_color in used_colors:
                    new_color += 1
                colors[ei] = new_color
                used_colors.add(new_color)
                assigned = True
        
        if valid:
            cnt = count_distinct_colors(colors)
            if cnt > best_count:
                best_count = cnt
                best_result = colors[:]
    
    return best_result

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
