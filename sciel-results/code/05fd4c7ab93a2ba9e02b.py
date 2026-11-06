"""Construct edge coloring of K16 with no rainbow 4-cycle, maximizing distinct colors."""
import time
import random

N = 16
DEADLINE = 75.0  # safety margin before 90s kill

def _edge_index(i, j):
    """Lexicographic index for edge (i,j) with 0<=i<j<N."""
    # Sum of edges before vertex i: i*(i-1)/2, plus j-i-1
    return i * (i - 1) // 2 + (j - i - 1)

def _validate(colors):
    """Check no 4-vertex subset has a 4-cycle with 4 distinct colors.
    Returns True if valid."""
    n = len(colors)
    if n != N * (N - 1) // 2:
        return False
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Three 4-cycles on {a,b,c,d}
                    # Cycle 1: a-b-c-d-a
                    e1 = colors[_edge_index(a, b)]
                    e2 = colors[_edge_index(b, c)]
                    e3 = colors[_edge_index(c, d)]
                    e4 = colors[_edge_index(a, d)]
                    if len({e1, e2, e3, e4}) == 4:
                        return False
                    # Cycle 2: a-b-d-c-a
                    e1 = colors[_edge_index(a, b)]
                    e2 = colors[_edge_index(b, d)]
                    e3 = colors[_edge_index(c, d)]
                    e4 = colors[_edge_index(a, c)]
                    if len({e1, e2, e3, e4}) == 4:
                        return False
                    # Cycle 3: a-c-b-d-a
                    e1 = colors[_edge_index(a, c)]
                    e2 = colors[_edge_index(b, c)]
                    e3 = colors[_edge_index(b, d)]
                    e4 = colors[_edge_index(a, d)]
                    if len({e1, e2, e3, e4}) == 4:
                        return False
    return True

def _build_block(n, start_color):
    """Build a valid coloring of K_n using colors in [start_color, start_color+n-1] or fewer.
    Returns (colors_list_of_length_n*(n-1)//2, num_colors_used).
    Uses a recursive partition strategy."""
    if n <= 1:
        return [], 0
    if n == 2:
        return [start_color], 1
    if n == 3:
        # Triangle: use 2 colors (one color repeated) to allow more colors overall
        # Actually for a triangle, we can use up to 3 colors, but let's use 2 to be safe
        # and save colors. Actually let's use 3 distinct colors for max.
        return [start_color, start_color + 1, start_color + 2], 3
    if n == 4:
        # K4: 6 edges. We need to avoid rainbow 4-cycles.
        # A 4-cycle in K4 uses 4 of the 6 edges. 
        # Strategy: use 3 colors. Color the two diagonals with color 0, 
        # and the 4 cycle edges with colors 1,2,1,2 (alternating).
        # Edges: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        # Let's try: (0,1)=c1, (1,2)=c2, (2,3)=c1, (3,0)=c2, (0,2)=c0, (1,3)=c0
        # Cycle 0-1-2-3-0: c1,c2,c1,c2 -> 2 colors. OK.
        # Cycle 0-1-3-2-0: c1,c0,c2,c2 -> 3 colors. OK.
        # Cycle 0-2-1-3-0: c0,c2,c0,c2 -> 2 colors. OK.
        # This uses 3 colors. Can we use 4?
        # If we use 4 colors, say c0,c1,c2,c3 on the 4-cycle edges, and the diagonals
        # must be from {c0,c1,c2,c3}. 
        # Actually, it's known that the max colors for K4 without rainbow C4 is 3.
        # Let's stick with 3.
        return [start_color + 1, start_color + 2, start_color + 1, start_color + 2, start_color, start_color], 3
    if n == 5:
        # K5: 10 edges. 
        # Use a 5-cycle with 2 colors, and the 5 diagonals (pentagram) with 3 more colors.
        # Total 5 colors? Let's check.
        # Actually, let's use a recursive split: 2+3 or 1+4.
        # Let's try 2+3.
        # Partition vertices into A (size 2) and B (size 3).
        # Internal A: 1 edge. Internal B: 3 edges. Cross: 6 edges.
        # Total 10 edges.
        # To avoid rainbow C4, we need to be careful with cross edges.
        # A simpler approach for small n: just use a greedy or known construction.
        # For now, let's use a simple scheme: color edge (i,j) with (i+j) % 4.
        # This uses 4 colors. Let's verify no rainbow C4.
        # This might not be optimal but is a valid start.
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(start_color + (i + j) % 4)
        return colors, 4
    
    # Recursive split for n >= 6
    # Split into two parts as balanced as possible
    mid = n // 2
    size_a = mid
    size_b = n - mid
    
    # Recursively build A and B
    colors_a, used_a = _build_block(size_a, start_color)
    colors_b, used_b = _build_block(size_b, start_color + used_a)
    
    # Now assign cross edges
    # To avoid rainbow C4, a common strategy is to use a limited palette for cross edges.
    # If we use only 2 colors for all cross edges, it's hard to create a rainbow C4
    # unless the internal edges are very diverse.
    # Let's use 2 colors for cross edges: start_color + used_a + used_b and start_color + used_a + used_b + 1
    # But we need to ensure these don't create rainbows with internal edges.
    # A safer approach: use the same 2 colors for all cross edges.
    cross_color_1 = start_color + used_a + used_b
    cross_color_2 = start_color + used_a + used_b + 1
    
    # Build the full color list
    full_colors = [0] * (n * (n - 1) // 2)
    idx = 0
    # Fill internal A edges
    for i in range(size_a):
        for j in range(i + 1, size_a):
            full_colors[idx] = colors_a[i * (i - 1) // 2 + (j - i - 1)]
            idx += 1
    # Fill cross edges
    for i in range(size_a):
        for j in range(size_b):
            # Alternate between cross_color_1 and cross_color_2
            full_colors[idx] = cross_color_1 if (i + j) % 2 == 0 else cross_color_2
            idx += 1
    # Fill internal B edges
    for i in range(size_b):
        for j in range(i + 1, size_b):
            full_colors[idx] = colors_b[i * (i - 1) // 2 + (j - i - 1)]
            idx += 1
    
    total_colors = used_a + used_b + 2
    return full_colors, total_colors

def construct():
    """Main entry point. Returns 120 integer colors in 0..119 for K16 edges."""
    start_time = time.time()
    
    # Start with the recursive construction
    colors, num_colors = _build_block(N, 0)
    
    if len(colors) != N * (N - 1) // 2:
        return [0] * (N * (N - 1) // 2)
    
    # Validate
    if not _validate(colors):
        # Fallback to a simple valid coloring
        colors = [(i + j) % 10 for i in range(N) for j in range(i + 1, N)]
        if not _validate(colors):
            return [0] * (N * (N - 1) // 2)
    
    # Try to improve by local search
    best_colors = colors[:]
    best_count = len(set(colors))
    
    # Local search: try changing individual edges to new colors
    # and see if it increases the color count while maintaining validity.
    # This is expensive, so we'll do a limited number of iterations.
    max_iter = 5000
    iter_count = 0
    rng = random.Random(42)
    
    while iter_count < max_iter:
        if time.time() - start_time > DEADLINE:
            break
        iter_count += 1
        
        # Pick a random edge
        e = rng.randrange(len(colors))
        old_color = colors[e]
        
        # Try a new color
        # Prefer unused colors
        used = set(colors)
        unused = [c for c in range(120) if c not in used]
        if unused:
            new_color = rng.choice(unused)
        else:
            new_color = rng.randrange(120)
        
        if new_color == old_color:
            continue
        
        colors[e] = new_color
        
        if _validate(colors):
            new_count = len(set(colors))
            if new_count > best_count:
                best_count = new_count
                best_colors = colors[:]
            else:
                # Revert if not better
                colors[e] = old_color
        else:
            # Revert if invalid
            colors[e] = old_color
    
    # Ensure colors are in range 0..119
    for i in range(len(best_colors)):
        if best_colors[i] < 0 or best_colors[i] > 119:
            best_colors[i] = best_colors[i] % 120
    
    return best_colors

# Pre-compute to avoid runtime delay
_construct_result = None
def _init():
    global _construct_result
    _construct_result = construct()

# Don't call _init() here, as it might take time.
# The construct function will be called by the evaluator.
