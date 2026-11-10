import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    # Pairs (i, j) in lexicographic order
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    # Map pair to index
    pair_index = {p: idx for idx, p in enumerate(pairs)}
    
    # Precompute all K4s with their edge indices
    k4s = list(combinations(range(n), 4))
    k4_edges = []
    for k4 in k4s:
        v1, v2, v3, v4 = k4
        edges = [pair_index[(v1, v2)], pair_index[(v1, v3)], pair_index[(v1, v4)],
                 pair_index[(v2, v3)], pair_index[(v2, v4)], pair_index[(v3, v4)]]
        k4_edges.append(edges)
    
    # Smarter initial coloring: use a structured pattern
    # Start with (i + j) % 13 to have fewer colors from the start
    colors = [(i + j) % 13 for i, j in pairs]
    num_colors = len(set(colors))
    
    def get_k4_color_count(k4_idx, col_list):
        """Count distinct colors in a K4."""
        edges = k4_edges[k4_idx]
        c = set()
        for e in edges:
            c.add(col_list[e])
        return len(c)
    
    def get_k4_colors(k4_idx, col_list):
        """Get set of colors in a K4."""
        edges = k4_edges[k4_idx]
        return set(col_list[e] for e in edges)
    
    def check_all_valid(col_list):
        """Check if all K4s have at least 5 distinct colors."""
        for k4_idx in range(len(k4s)):
            if get_k4_color_count(k4_idx, col_list) < 5:
                return False
        return True
    
    def is_valid_merge(c1, c2, col_list):
        """Check if merging c1 and c2 is valid."""
        # Only need to check K4s that contain both c1 and c2
        for k4_idx in range(len(k4s)):
            k4_col_set = get_k4_colors(k4_idx, col_list)
            if len(k4_col_set) == 5 and c1 in k4_col_set and c2 in k4_col_set:
                return False
        return True
    
    def try_merge():
        nonlocal colors, num_colors
        if num_colors <= 1:
            return False
        
        # Count color usage
        color_counts = {}
        for c in colors:
            color_counts[c] = color_counts.get(c, 0) + 1
        
        # Try to merge least-used colors first
        active_colors = sorted(color_counts.keys(), key=lambda x: color_counts[x])
        
        for i in range(len(active_colors)):
            for j in range(i + 1, len(active_colors)):
                c1 = active_colors[i]
                c2 = active_colors[j]
                if is_valid_merge(c1, c2, colors):
                    # Merge c2 into c1 (c1 is less used, so we keep it)
                    # Actually, merge into the more used one to reduce changes
                    if color_counts[c1] < color_counts[c2]:
                        c1, c2 = c2, c1
                    for k in range(len(colors)):
                        if colors[k] == c2:
                            colors[k] = c1
                    num_colors -= 1
                    return True
        return False
    
    # Main search loop
    while time.time() < deadline:
        # Try to merge colors
        if try_merge():
            continue
        
        # If no merge possible, try random recoloring
        # Pick a random edge
        idx = random.randint(0, len(colors) - 1)
        old_color = colors[idx]
        
        # Try to change to an existing color to potentially free up old_color
        active_colors = list(set(colors))
        if len(active_colors) > 1:
            # Prefer colors that are more used (to make old_color less used)
            color_counts = {}
            for c in colors:
                color_counts[c] = color_counts.get(c, 0) + 1
            
            # Choose a new color, preferring more used ones
            candidates = [c for c in active_colors if c != old_color]
            new_color = max(candidates, key=lambda c: color_counts[c])
            
            colors[idx] = new_color
            
            # Check validity - only check K4s containing this edge
            valid = True
            for k4_idx in range(len(k4s)):
                if idx in k4_edges[k4_idx]:
                    if get_k4_color_count(k4_idx, colors) < 5:
                        valid = False
                        break
            
            if valid:
                # Check if old_color is still used
                if old_color not in colors:
                    num_colors -= 1
                # Keep the change
            else:
                # Revert
                colors[idx] = old_color
    
    return colors
