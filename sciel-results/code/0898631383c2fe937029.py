import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin
    
    n = 16
    # Generate all edges in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)  # 120
    
    # Generate all K4s and map them to edge indices
    # edge_idx: (i,j) -> index
    edge_idx = {e: idx for idx, e in enumerate(edges)}
    
    # k4s: list of tuples of 4 vertices
    # k4_edge_indices: list of lists of 6 edge indices for each K4
    k4s = []
    k4_edge_indices = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        k4_edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        k4s.append(quad)
        k4_edge_indices.append([edge_idx[e] for e in k4_edges])
        
    # edge_to_k4s: for each edge, list of K4 indices that contain it
    edge_to_k4s = [[] for _ in range(m)]
    for k4_idx, edge_indices in enumerate(k4_edge_indices):
        for e_idx in edge_indices:
            edge_to_k4s[e_idx].append(k4_idx)
            
    # Initial coloring: (i+j) % 17
    coloring = np.array([(i + j) % 17 for i, j in edges], dtype=int)
    best_coloring = coloring.copy()
    best_colors = len(set(coloring))
    
    def is_valid_coloring(col):
        """Check if the entire coloring is valid. Used for initialization and final check."""
        for k4_e in k4_edge_indices:
            colors = set(col[i] for i in k4_e)
            if len(colors) < 5:
                return False
        return True
        
    def check_local_validity(col, changed_edge_idx):
        """Check if the coloring is valid, assuming only changed_edge_idx was modified.
        Only checks K4s containing changed_edge_idx."""
        for k4_idx in edge_to_k4s[changed_edge_idx]:
            e_indices = k4_edge_indices[k4_idx]
            colors = set(col[i] for i in e_indices)
            if len(colors) < 5:
                return False
        return True

    if not is_valid_coloring(coloring):
        # Fallback if seed is invalid (though it should be)
        # Random restart
        while True:
            coloring = np.random.randint(0, 17, size=m)
            if is_valid_coloring(coloring):
                break
            if time.time() > deadline:
                break
        if time.time() > deadline:
             return [int(c) for c in coloring]
        
    best_coloring = coloring.copy()
    best_colors = len(set(coloring))
    
    random.seed(42)
    
    # Precompute unique colors for faster sampling?
    # We will try to reduce the number of distinct colors.
    
    while time.time() < deadline:
        # Strategy:
        # 1. Try to recolor an edge with a "rare" or "high" color to a more common one.
        # 2. Try random swaps.
        
        # Get color counts
        unique_colors, counts = np.unique(coloring, return_counts=True)
        
        # If we have more than 13 colors, try to eliminate one.
        # Pick a color with the lowest count (or highest index)
        if len(unique_colors) > 13:
            # Pick the color with the lowest count to eliminate
            # Tie-break by highest value to reduce range
            min_count = np.min(counts)
            candidates = np.where(counts == min_count)[0]
            # Pick the one with the highest color value among candidates
            cand_colors = unique_colors[candidates]
            target_color = cand_colors[np.argmax(cand_colors)]
            
            # Find edges with this color
            target_edges = np.where(coloring == target_color)[0]
            
            if len(target_edges) > 0:
                # Pick a random edge with this color
                e_idx = random.choice(target_edges)
                
                # Try to recolor it to another existing color
                # Exclude the target color itself
                other_colors = [c for c in unique_colors if c != target_color]
                if other_colors:
                    # Shuffle to try different ones
                    random.shuffle(other_colors)
                    found_valid = False
                    for new_color in other_colors:
                        old_color = coloring[e_idx]
                        coloring[e_idx] = new_color
                        
                        if check_local_validity(coloring, e_idx):
                            # Check if it reduced the number of colors
                            new_unique = set(coloring)
                            if len(new_unique) < best_colors:
                                best_colors = len(new_unique)
                                best_coloring = coloring.copy()
                            # Keep the change if it doesn't increase colors?
                            # Or just keep it if valid?
                            # We want to minimize colors.
                            # If it didn't reduce colors, it's neutral.
                            # Let's keep it to allow further moves.
                            found_valid = True
                            break
                        else:
                            coloring[e_idx] = old_color
                            
                    if not found_valid:
                        # Revert if no valid move found
                        coloring[e_idx] = old_color
        else:
            # If we are at 13 colors (or lower), try to maintain validity while exploring
            # or just random swaps to find a better structure?
            # Actually, if we are at 13, we can't go lower (bound is 13).
            # So we can stop or just verify.
            # But maybe we are at 14 and stuck?
            # Let's try random swaps to escape local minima.
            
            if random.random() < 0.5:
                # Random swap
                i1 = random.randint(0, m - 1)
                i2 = random.randint(0, m - 1)
                if i1 != i2:
                    c1, c2 = coloring[i1], coloring[i2]
                    if c1 != c2:
                        coloring[i1], coloring[i2] = c2, c1
                        # Check validity for both edges
                        # We need to check K4s containing i1 OR i2
                        # But check_local_validity only checks one edge.
                        # We need a combined check or check both.
                        
                        # Optimization: Check K4s containing i1
                        valid1 = check_local_validity(coloring, i1)
                        if valid1:
                            # Check K4s containing i2 (some might be shared, but set handles it)
                            # Actually, if we change i1 and i2, we need to check all K4s containing either.
                            # For simplicity and safety, let's just do a full check if we are close to deadline?
                            # No, full check is slow.
                            # Let's create a helper that checks a set of edges.
                            
                            # Re-verify:
                            # We changed i1 and i2.
                            # Affected K4s are union of edge_to_k4s[i1] and edge_to_k4s[i2].
                            # Let's just check the union.
                            affected_k4s = set(edge_to_k4s[i1]) | set(edge_to_k4s[i2])
                            valid = True
                            for k4_idx in affected_k4s:
                                e_indices = k4_edge_indices[k4_idx]
                                colors = set(coloring[i] for i in e_indices)
                                if len(colors) < 5:
                                    valid = False
                                    break
                            
                            if valid:
                                new_unique = set(coloring)
                                if len(new_unique) < best_colors:
                                    best_colors = len(new_unique)
                                    best_coloring = coloring.copy()
                            else:
                                # Revert
                                coloring[i1], coloring[i2] = c1, c2
            else:
                # Try to recolor a random edge to a random existing color
                e_idx = random.randint(0, m - 1)
                old_color = coloring[e_idx]
                # Pick a random color from the palette
                palette = list(set(coloring))
                if len(palette) > 1:
                    new_color = random.choice(palette)
                    if new_color != old_color:
                        coloring[e_idx] = new_color
                        if check_local_validity(coloring, e_idx):
                            new_unique = set(coloring)
                            if len(new_unique) < best_colors:
                                best_colors = len(new_unique)
                                best_coloring = coloring.copy()
                        else:
                            coloring[e_idx] = old_color

    # Remap colors to 0..k-1
    result = [int(c) for c in best_coloring]
    unique_colors = sorted(set(result))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    result = [color_map[c] for c in result]
    
    return result
