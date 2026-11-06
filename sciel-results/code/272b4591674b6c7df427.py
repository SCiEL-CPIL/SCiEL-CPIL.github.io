import time
import sys
import random
import itertools
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Precompute edge indices for all pairs (i, j) with 0 <= i < j < N
    # We need to map (i, j) to index 0..119
    # Lexicographic order: (0,1), (0,2), ..., (0,15), (1,2), ..., (15,14), (15,15) is not a pair.
    # Total edges = 16*15/2 = 120.
    
    # Let's build a lookup table for edge indices to speed up checks
    # edge_idx[i][j] = index
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
            
    # Precompute all 4-cycles (sets of 4 vertices)
    # A 4-cycle on vertices {a,b,c,d} has 3 possible pairings of opposite edges:
    # 1. (a,b) and (c,d) are opposite? No, in a K4, the 3 perfect matchings are:
    #    (a,b)-(c,d), (a,c)-(b,d), (a,d)-(b,c)
    # Wait, the condition is "No four-cycle may have four distinct edge colors".
    # A 4-cycle in a graph usually refers to a cycle of length 4. In K4, there are 3 cycles of length 4:
    # a-b-c-d-a (edges ab, bc, cd, da)
    # a-b-d-c-a (edges ab, bd, dc, ca)
    # a-c-b-d-a (edges ac, cb, bd, da)
    # The task says "all three cycles on every four-vertex subset are checked".
    # This implies we check the 3 Hamiltonian cycles of the K4 induced by the 4 vertices.
    # Let vertices be v0, v1, v2, v3.
    # Cycle 1: v0-v1-v2-v3-v0. Edges: (v0,v1), (v1,v2), (v2,v3), (v3,v0).
    # Cycle 2: v0-v1-v3-v2-v0. Edges: (v0,v1), (v1,v3), (v3,v2), (v2,v0).
    # Cycle 3: v0-v2-v1-v3-v0. Edges: (v0,v2), (v2,v1), (v1,v3), (v3,v0).
    
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        # Cycle 2: a-b-d-c-a
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        # Cycle 3: a-c-b-d-a
        cycles.append(((a,c), (c,b), (b,d), (d,a)))
        
    # Convert cycles to edge indices for faster lookup
    cycle_indices = []
    for c in cycles:
        cycle_indices.append(tuple(edge_idx[u][v] for u, v in c))

    def is_valid(colors):
        # Check if any cycle has 4 distinct colors
        for c_idx in cycle_indices:
            c0 = colors[c_idx[0]]
            c1 = colors[c_idx[1]]
            c2 = colors[c_idx[2]]
            c3 = colors[c_idx[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True

    def count_colors(colors):
        return len(set(colors))

    # Initialize with a known valid coloring (the partition one from the seed)
    # This ensures we start with a valid state.
    def get_seed_colors():
        colors = [0] * NUM_EDGES
        color_id = 0
        def process_partition(vertices):
            nonlocal color_id
            if len(vertices) <= 1:
                return
            mid = len(vertices) // 2
            left = vertices[:mid]
            right = vertices[mid:]
            for u in left:
                for v in right:
                    idx = edge_idx[u][v]
                    colors[idx] = color_id
            color_id += 1
            process_partition(left)
            process_partition(right)
        process_partition(list(range(N)))
        return colors

    current_colors = get_seed_colors()
    best_score = count_colors(current_colors)
    best_colors = current_colors.copy()
    
    # Local Search: Hill Climbing
    # Strategy: 
    # 1. Pick a random edge.
    # 2. Try to assign it a new color (from 0..119) that hasn't been used much, or specifically try to introduce a NEW color to the palette.
    # Actually, to maximize distinct colors, we want to replace a "common" color with a "rare" or "new" color if valid.
    # Better: Randomly pick an edge, and try to recolor it to a random color from 0..119.
    # If the new coloring is valid, accept it.
    # To encourage diversity, we can bias towards colors that are currently under-represented or new.
    
    # Let's track frequency of colors to bias selection
    freq = [0] * MAX_COLORS
    for c in current_colors:
        freq[c] += 1
        
    random.seed(42)
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = current_colors[e]
        
        # Try to find a better color
        # We want to increase the number of distinct colors.
        # If we change a color that appears many times to a color that appears 0 times (and is valid), we gain +1 distinct color (if the old color still exists elsewhere) or 0 (if old color disappears).
        # If we change a color that appears 1 time to a new color, distinct count stays same.
        # So we prefer to recolor edges that have high frequency to new/rare colors.
        
        # Let's just try a random new color first.
        # To be smarter: try a color that is currently unused if possible?
        # There are 120 slots. If we use 15 colors, there are 105 unused.
        
        # Heuristic: Pick a target color.
        # If we pick a random color from 0..119, we might just shuffle existing colors.
        # Let's bias towards colors that are less frequent.
        
        # Simple approach: Try a random color. If valid, accept.
        # To improve, if valid and new_color is not in set(current_colors) and old_color is in set(current_colors) with freq > 1, it's a win.
        # But checking set every time is expensive.
        
        # Let's just do random walk with acceptance if valid.
        # Since the space of valid colorings is likely connected or has large components, this should explore.
        
        new_color = random.randint(0, MAX_COLORS - 1)
        
        if new_color == old_color:
            continue
            
        current_colors[e] = new_color
        
        # Validate
        if is_valid(current_colors):
            # Accept
            # Update freq
            freq[old_color] -= 1
            freq[new_color] += 1
            
            # Check if this improves the distinct color count
            # We don't need to recalculate count every time if we track it smartly, but N=16 is small.
            # Let's just update best if it looks better.
            # Actually, just keep the current state as "current".
            # We track best separately.
            
            # Optimization: Only check score periodically or if we suspect improvement?
            # No, just check score. It's O(120).
            score = count_colors(current_colors)
            if score > best_score:
                best_score = score
                best_colors = current_colors.copy()
        else:
            # Revert
            current_colors[e] = old_color
            
    return best_colors
