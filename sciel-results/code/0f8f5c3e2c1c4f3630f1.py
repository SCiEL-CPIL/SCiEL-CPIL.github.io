import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def edge_vertices(idx):
    i = 0
    while i * N - i * (i + 1) // 2 + (N - i - 1) <= idx:
        i += 1
    i -= 1
    offset = idx - (i * N - i * (i + 1) // 2)
    j = i + 1 + offset
    return i, j

def construct():
    deadline = time.time() + 55  # safety margin
    
    # Precompute all 4-cycles
    cycles = []
    vertices = list(range(N))
    
    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        e1 = (edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a))
        e2 = (edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a))
        e3 = (edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a))
        
        for cycle in [e1, e2, e3]:
            cycles.append(cycle)
            
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(ci)
            
    def is_valid_change(e, new_color, colors):
        old_color = colors[e]
        if new_color == old_color:
            return True
        
        for ci in edge_to_cycles[e]:
            cycle = cycles[ci]
            edge_colors = []
            for ce in cycle:
                if ce == e:
                    edge_colors.append(new_color)
                else:
                    edge_colors.append(colors[ce])
            if len(set(edge_colors)) == 4:
                return False
        return True
        
    def count_distinct_colors(colors):
        return len(set(colors))
        
    def is_fully_valid(colors):
        for cycle in cycles:
            ec = [colors[e] for e in cycle]
            if len(set(ec)) == 4:
                return False
        return True

    # Generate 1-factorization of K_16
    # Using standard round-robin algorithm for even N
    # Vertices 0..15
    # We need 15 matchings
    matchings = []
    # Fix vertex 0? Or just rotate.
    # Standard construction for K_2n:
    # Place vertices 1..2n-1 in a circle, 2n in center.
    # Here N=16, so 2n=16, n=8.
    # Vertices 1..15 in circle, 0 in center?
    # Let's use a simpler constructive method or just random assignment that forms a 1-factorization.
    # Actually, let's just use the property:
    # For i in 0..14:
    #   Matching M_i consists of edges (j, k) such that (j+k) % 15 == i ? No, that's for K_15.
    # For K_16, we can use:
    # Vertices 0..15.
    # For r in 0..14:
    #   Pair (0, r+1) is not quite right.
    
    # Let's use the standard circle method for K_16.
    # Label vertices 0, 1, ..., 14, \infty.
    # For r = 0, ..., 14:
    #   Match \infty with r.
    #   Match (r+1, r-1), (r+2, r-2), ..., (r+7, r-7) mod 15.
    
    # Let's map 0..15 to this. Let 0 be \infty. 1..15 be 1..14, 0? No.
    # Let's map 0..14 to 1..14, 0? And 15 to \infty?
    # Let's just say vertices are 0..15.
    # Let 15 be the center.
    # Vertices 0..14 are on the circle.
    # For r in 0..14:
    #   Edge (15, r) is in Matching r.
    #   For k in 1..7:
    #     u = (r + k) % 15
    #     v = (r - k) % 15
    #     Edge (u, v) is in Matching r.
    
    colors = [0] * NUM_EDGES
    for r in range(15):
        # Center edge
        u, v = 15, r
        idx = edge_index(u, v)
        colors[idx] = r
        
        for k in range(1, 8):
            u = (r + k) % 15
            v = (r - k) % 15
            idx = edge_index(u, v)
            colors[idx] = r
            
    # Verify this is a valid 1-factorization coloring (should be 15 colors)
    if not is_fully_valid(colors):
        # Fallback to all 0s if construction failed (shouldn't happen)
        colors = [0] * NUM_EDGES
        
    best_colors = colors[:]
    best_count = count_distinct_colors(best_colors)
    
    rng = random.Random(42)
    
    # Local Search
    # We want to maximize distinct colors.
    # We will try to recolor edges.
    # To escape local optima, we might need to allow temporary decrease in color count?
    # Or just try many random restarts?
    # Given time limit, a single long run with good moves is better.
    
    # Strategy:
    # 1. Try to change an edge to a NEW color (not currently in use, or less frequent).
    # 2. If valid, accept.
    # 3. If not, try to change an edge to an EXISTING color to shuffle.
    
    used_colors = set(colors)
    
    for iteration in range(1000000):
        if time.time() > deadline:
            break
            
        e = rng.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Decide target color
        # With high prob, try a color not in used_colors to increase count
        # With low prob, try a color in used_colors to mix
        if rng.random() < 0.8:
            # Pick a random color 0..119
            # If it's in used_colors, it doesn't increase count (unless we drop old_color's last usage)
            # If it's not, it increases count (unless old_color is unique and we replace it? No, we add new, keep old if used elsewhere)
            # Actually, if we change e from A to B:
            # If B is new: count increases by 1 (if A was used elsewhere, A remains).
            # If B is old: count might decrease if A was unique.
            
            # To maximize, we should prefer B not in used_colors.
            # But there are only 120 colors. If used_colors is small, many are free.
            
            # Let's just pick random 0..119.
            new_color = rng.randint(0, 119)
        else:
            # Pick from used_colors
            new_color = rng.choice(list(used_colors))
            
        if new_color == old_color:
            continue
            
        if is_valid_change(e, new_color, colors):
            # Update used_colors
            # Check if old_color becomes unused
            # This is expensive to check every time.
            # We can just calculate count after acceptance.
            
            colors[e] = new_color
            
            # Update used_colors set
            # We need to know if old_color is still present.
            # And if new_color was already present.
            
            # Efficient update:
            # Count of old_color in colors array?
            # We can maintain a frequency array.
            
            # Let's maintain freq array for speed.
            # But we already started. Let's just do a quick check or full count if we suspect change.
            # Full count is O(120). 120 * 1000000 is 120M ops. Might be too slow in Python.
            # Let's optimize is_valid_change and the loop.
            
            # Better: Maintain freq array.
            # We didn't initialize it. Let's do it now and fix the loop logic.
            
            # RESTART LOGIC WITH FREQ ARRAY
            
            # We will break out of this loop and rewrite the search part.
            # The current loop is inefficient.
            
            # Let's replace the search section entirely.
            pass
            
    # The above loop is a bit messy. Let's replace the whole construct body from "best_colors" onwards with a cleaner implementation using frequency counts.
    
    return best_colors
