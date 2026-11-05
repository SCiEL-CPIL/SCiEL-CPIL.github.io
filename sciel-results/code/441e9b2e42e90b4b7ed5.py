import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def edge_vertices(idx):
    # inverse of edge_index
    # find i, j such that edge_index(i,j) = idx
    i = 0
    while i * N - i * (i + 1) // 2 + (N - i - 1) <= idx:
        i += 1
    i -= 1
    offset = idx - (i * N - i * (i + 1) // 2)
    j = i + 1 + offset
    return i, j

def construct():
    deadline = time.time() + 55  # safety margin
    
    # Precompute all 4-cycles: for each 4-set {a,b,c,d}, the 3 possible 4-cycles
    # Actually, for 4 vertices, there are 3 distinct 4-cycles (up to rotation):
    # (a,b,c,d), (a,b,d,c), (a,c,b,d) - but these are the same cycle traversed differently
    # Wait: a 4-cycle on vertices {a,b,c,d} is a cyclic ordering. The edges are:
    # Cycle 1: a-b, b-c, c-d, d-a
    # Cycle 2: a-b, b-d, d-c, c-a  
    # Cycle 3: a-c, c-b, b-d, d-a
    # These are the 3 distinct 4-cycles (each uses a different perfect matching as "opposite" edges)
    
    # Precompute: for each edge, which 4-cycles contain it
    # For edge (i,j), a 4-cycle containing it uses 2 other vertices k,l
    # The cycle is i-j-k-l-i or i-j-l-k-i
    # So for edge (i,j) and vertices k,l (k,l != i,j), there are 2 cycles:
    #   (i,j,k,l): edges ij, jk, kl, li
    #   (i,j,l,k): edges ij, jl, lk, ki
    
    # Let's precompute all 4-cycles as tuples of 4 edge indices
    cycles = []
    cycle_to_edges = []
    
    vertices = list(range(N))
    from itertools import combinations
    
    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        # 3 distinct 4-cycles
        # Cycle 1: a-b-c-d-a: edges ab, bc, cd, da
        e1 = (edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a))
        # Cycle 2: a-b-d-c-a: edges ab, bd, dc, ca
        e2 = (edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a))
        # Cycle 3: a-c-b-d-a: edges ac, cb, bd, da
        e3 = (edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a))
        
        for cycle in [e1, e2, e3]:
            cycles.append(cycle)
            cycle_to_edges.append(cycle)
    
    # For each edge, list of cycle indices that contain it
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(ci)
    
    # Initial coloring: use a 5-coloring via 1-factorization approach
    # For K_16, we can use round-robin tournament scheduling
    # Color based on (i+j) % 5 won't work well. Let's use a proper edge coloring with 7 colors (chromatic index of K_16 is 15, but we can use fewer for a non-proper coloring that satisfies our constraint)
    
    # Actually, let's start with all zeros, then build up
    colors = [0] * NUM_EDGES
    
    # Helper: check if changing edge e to color c is valid
    def is_valid_change(e, new_color, colors):
        old_color = colors[e]
        if new_color == old_color:
            return True
        
        # Check all cycles containing edge e
        for ci in edge_to_cycles[e]:
            cycle = cycles[ci]
            edge_colors = []
            for ce in cycle:
                if ce == e:
                    edge_colors.append(new_color)
                else:
                    edge_colors.append(colors[ce])
            # Check if all 4 are distinct
            if len(set(edge_colors)) == 4:
                return False
        return True
    
    # Greedy: try to assign new colors to edges
    # Start with color 0 everywhere, then try to introduce color 1, 2, ...
    
    # Better approach: local search
    # State: colors array
    # Objective: maximize number of distinct colors
    # Constraint: no 4-cycle has 4 distinct edge colors
    
    # Start with a valid coloring: use color = (i * N + j) % 5
    for idx in range(NUM_EDGES):
        i, j = edge_vertices(idx)
        colors[idx] = (i + j) % 5
    
    # Verify initial coloring is valid
    def count_distinct_colors(colors):
        return len(set(colors))
    
    def is_fully_valid(colors):
        for cycle in cycles:
            ec = [colors[e] for e in cycle]
            if len(set(ec)) == 4:
                return False
        return True
    
    # Verify
    if not is_fully_valid(colors):
        # Fallback: all zeros
        colors = [0] * NUM_EDGES
    
    best_colors = colors[:]
    best_count = count_distinct_colors(best_colors)
    
    # Local search: try to change random edges to new colors
    rng = random.Random(42)
    
    # Try to increase colors by recoloring
    for iteration in range(500000):
        if time.time() > deadline:
            break
        
        e = rng.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Try a new color: either a random color in 0..119, or old_color +/- 1
        if rng.random() < 0.7:
            new_color = rng.randint(0, 119)
        else:
            new_color = (old_color + rng.choice([-1, 1])) % 120
        
        if new_color == old_color:
            continue
        
        if is_valid_change(e, new_color, colors):
            colors[e] = new_color
            new_count = count_distinct_colors(colors)
            if new_count > best_count:
                best_count = new_count
                best_colors = colors[:]
        
        # Occasionally try to reduce colors to escape local optima? No, we want max.
    
    # Final verification
    if not is_fully_valid(best_colors):
        # Return a safe fallback
        return [0] * NUM_EDGES
    
    return best_colors
