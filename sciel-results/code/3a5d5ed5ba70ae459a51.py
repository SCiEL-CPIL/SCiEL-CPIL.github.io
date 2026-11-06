import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 85.0

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def count_used(colors):
    return len(set(colors))

def is_valid_coloring(colors):
    """Check if no 4-cycle has four distinct edge colors."""
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    c1 = (colors[edge_idx(a,b)], colors[edge_idx(b,c)],
                          colors[edge_idx(c,d)], colors[edge_idx(d,a)])
                    if len(set(c1)) == 4:
                        return False
                    c2 = (colors[edge_idx(a,b)], colors[edge_idx(b,d)],
                          colors[edge_idx(d,c)], colors[edge_idx(c,a)])
                    if len(set(c2)) == 4:
                        return False
                    c3 = (colors[edge_idx(a,c)], colors[edge_idx(c,b)],
                          colors[edge_idx(b,d)], colors[edge_idx(d,a)])
                    if len(set(c3)) == 4:
                        return False
    return True

def fast_valid_after_recolor(colors, ei, ej, new_color):
    """Check validity after recoloring edge (ei,ej) to new_color.
    Only checks 4-cycles that contain edge (ei,ej)."""
    other = [v for v in range(N) if v != ei and v != ej]
    for xi in range(len(other)):
        x = other[xi]
        for xj in range(xi + 1, len(other)):
            y = other[xj]
            # Cycle 1: ei-ej-x-y-ei
            c1 = (new_color, colors[edge_idx(ej, x)], colors[edge_idx(x, y)], colors[edge_idx(y, ei)])
            if len(set(c1)) == 4:
                return False
            # Cycle 2: ei-ej-y-x-ei
            c2 = (new_color, colors[edge_idx(ej, y)], colors[edge_idx(y, x)], colors[edge_idx(x, ei)])
            if len(set(c2)) == 4:
                return False
    return True

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Structural approach: Group-based coloring with 4 groups of 4 vertices each.
    # Groups: G0={0,1,2,3}, G1={4,5,6,7}, G2={8,9,10,11}, G3={12,13,14,15}
    # 
    # Key insight: If we color edges between groups using only 2 colors (say 0 and 1),
    # then any 4-cycle that uses cross-group edges will have at most 2 distinct colors
    # on those cross edges, making it impossible to have 4 distinct colors total.
    #
    # Edges within groups can use many colors, but we must ensure no rainbow C4 forms
    # within or across group boundaries.
    #
    # Strategy:
    # 1. Color all cross-group edges with colors 0 and 1 (alternating pattern)
    # 2. Color within-group edges with distinct new colors
    # 3. Use local search to add more colors while maintaining validity
    
    # Initialize: cross-group edges get color 0 or 1 based on group parity
    colors = [0] * NUM_EDGES
    group_of = [i // 4 for i in range(N)]
    
    for i in range(N):
        for j in range(i + 1, N):
            gi, gj = group_of[i], group_of[j]
            if gi != gj:
                # Cross-group edge: use color 0 or 1
                colors[edge_idx(i, j)] = (gi + gj) % 2
            else:
                # Within-group edge: start with color 2 (will be improved)
                colors[edge_idx(i, j)] = 2
    
    # Verify initial coloring
    if not is_valid_coloring(colors):
        # Fallback: all edges get color 0
        colors = [0] * NUM_EDGES
    
    best_colors = list(colors)
    best_count = count_used(best_colors)
    
    # Precompute edge list
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    
    # Local search: try to recolor edges to new colors
    # Focus on within-group edges first (they have more freedom)
    
    within_group_edges = [(i, j) for i, j in edges if group_of[i] == group_of[j]]
    cross_group_edges = [(i, j) for i, j in edges if group_of[i] != group_of[j]]
    
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < DEADLINE - 5:
        if no_improve_count > max_no_improve:
            # Perturbation: try a different random seed and restart local search
            rng.seed(int(time.time() * 1000) % 100000)
            no_improve_count = 0
            # Randomly recolor a few within-group edges
            for _ in range(3):
                if within_group_edges:
                    ei, ej = rng.choice(within_group_edges)
                    c = rng.randint(0, 119)
                    if fast_valid_after_recolor(best_colors, ei, ej, c):
                        best_colors[edge_idx(ei, ej)] = c
            continue
        
        # Choose which edges to try
        if rng.random() < 0.7:
            # Focus on within-group edges (more likely to accept new colors)
            ei, ej = rng.choice(within_group_edges)
        else:
            ei, ej = rng.choice(edges)
        
        idx = edge_idx(ei, ej)
        old_color = best_colors[idx]
        
        # Try to assign a new color (not currently used)
        used = set(best_colors)
        new_colors = [c for c in range(120) if c not in used]
        
        if new_colors:
            # Try a few new colors
            for _ in range(min(3, len(new_colors))):
                new_color = rng.choice(new_colors)
                if fast_valid_after_recolor(best_colors, ei, ej, new_color):
                    best_colors[idx] = new_color
                    no_improve_count = 0
                    break
            else:
                no_improve_count += 1
            continue
        
        # All 120 colors used? Try to maintain validity with random recolors
        c = rng.randint(0, 119)
        if c != old_color and fast_valid_after_recolor(best_colors, ei, ej, c):
            best_colors[idx] = c
            if old_color not in best_colors:
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
    
    # Final verification
    if not is_valid_coloring(best_colors):
        # Fallback
        best_colors = [0] * NUM_EDGES
    
    return best_colors
