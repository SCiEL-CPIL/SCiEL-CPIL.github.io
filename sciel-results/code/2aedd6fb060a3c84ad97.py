import itertools
import random

def construct():
    # Start with the additive mod 17 construction (uses 17 colors, valid)
    # Then attempt to reduce color count via local search / recoloring
    # Goal: minimize number of distinct colors used while maintaining validity
    # (every K4 spans >= 5 distinct colors)
    
    # Build initial coloring: color(i,j) = (i+j) % 17
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {}
    for idx, (i, j) in enumerate(edges):
        edge_index[(i, j)] = idx
        edge_index[(j, i)] = idx
    
    colors = [(i + j) % 17 for i, j in edges]
    
    # Precompute all K4s (4-cliques)
    k4s = list(itertools.combinations(range(n), 4))
    
    def check_validity(col_arr):
        """Check if every K4 spans at least 5 distinct colors."""
        for quad in k4s:
            a, b, c, d = quad
            e_colors = {
                col_arr[edge_index[(a, b)]],
                col_arr[edge_index[(a, c)]],
                col_arr[edge_index[(a, d)]],
                col_arr[edge_index[(b, c)]],
                col_arr[edge_index[(b, d)]],
                col_arr[edge_index[(c, d)]]
            }
            if len(e_colors) < 5:
                return False
        return True
    
    def count_colors(col_arr):
        return len(set(col_arr))
    
    # Local search: try to recolor edges to reduce total distinct colors
    # Strategy: iteratively try to assign fewer colors
    # We'll try to reduce from 17 down
    
    best_colors = colors[:]
    best_count = count_colors(best_colors)
    
    # Try reducing color count by attempting to merge colors
    # For each target color count from 16 down to 13:
    for target_k in range(best_count - 1, 12, -1):
        # Try to find a valid coloring using at most target_k colors
        # Use a simple iterative approach: randomly reassign colors and check
        success = False
        attempts = 0
        max_attempts = 5000
        
        while attempts < max_attempts and not success:
            attempts += 1
            # Start from current best and try to reduce
            cand = best_colors[:]
            
            # Try to recolor some edges to use fewer colors
            # Identify the most "expensive" color (used on most edges) and try to merge
            color_counts = {}
            for c in cand:
                color_counts[c] = color_counts.get(c, 0) + 1
            
            # Sort colors by frequency
            sorted_colors = sorted(color_counts.keys(), key=lambda x: -color_counts[x])
            
            # Try to eliminate the least-used color first
            for old_color in sorted_colors:
                if old_color >= target_k:
                    # Try to recolor all edges with old_color to another color
                    # Pick a random alternative color from 0..target_k-1
                    alt = random.randint(0, target_k - 1)
                    for idx in range(len(cand)):
                        if cand[idx] == old_color:
                            cand[idx] = alt
                    
                    if check_validity(cand):
                        success = True
                        break
        
        if success:
            best_colors = cand
            best_count = count_colors(best_colors)
            # Continue trying to reduce further
    
    # Return the coloring in lexicographic order
    return best_colors
