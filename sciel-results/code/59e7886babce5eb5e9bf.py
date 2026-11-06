import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 148  # 2s margin for safety
    
    n = 16
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    # Map (i, j) to edge index
    ij_to_edge_idx = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute for each K4, the indices of its 6 edges
    k4s = list(combinations(range(n), 4))
    k4_edge_indices = []
    for verts in k4s:
        edges_in_k4 = []
        for i in range(4):
            for j in range(i + 1, 4):
                u = verts[i]
                v = verts[j]
                edges_in_k4.append(ij_to_edge_idx[(u, v)])
        k4_edge_indices.append(edges_in_k4)
    
    # Convert to numpy array for faster operations
    k4_edge_indices_np = np.array(k4_edge_indices, dtype=np.int32)
    
    # Precompute: for each edge, which K4s contain it (for fast invalidity tracking)
    # Actually, let's precompute the transpose: for each edge index, list of K4 indices
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, eidxs in enumerate(k4_edge_indices):
        for eidx in eidxs:
            edge_to_k4s[eidx].append(k4_idx)
    
    def is_valid_fast(col_arr):
        """Check if every K4 has at least 5 distinct colors using vectorized numpy.
        col_arr should be a numpy array of shape (120,)."""
        k4_colors = col_arr[k4_edge_indices_np]  # shape: (1820, 6)
        sorted_colors = np.sort(k4_colors, axis=1)
        diffs = np.diff(sorted_colors, axis=1)  # shape: (1820, 5)
        distinct_counts = np.sum(diffs != 0, axis=1) + 1  # shape: (1820,)
        return bool(np.all(distinct_counts >= 5))
    
    def get_tight_k4s(col_arr):
        """Return indices of K4s that have exactly 5 distinct colors."""
        k4_colors = col_arr[k4_edge_indices_np]
        sorted_colors = np.sort(k4_colors, axis=1)
        diffs = np.diff(sorted_colors, axis=1)
        distinct_counts = np.sum(diffs != 0, axis=1) + 1
        tight_mask = (distinct_counts == 5)
        return np.where(tight_mask)[0]
    
    def count_colors(col):
        """Count distinct colors used."""
        return len(set(col))
    
    best_colors = None
    best_count = 999
    
    # Initial coloring strategies - more diverse
    strategies = [
        lambda i, j: (i + j) % 17,
        lambda i, j: (i * 2 + j) % 17,
        lambda i, j: (i + 2*j) % 17,
        lambda i, j: (i * j) % 17,
        lambda i, j: (i + j + i*j) % 17,
        lambda i, j: (i * 3 + j * 2) % 17,
        lambda i, j: (i + j + 1) % 17,
        lambda i, j: (i * 5 + j * 3) % 17,
        lambda i, j: (i * 4 + j) % 17,
        lambda i, j: (i + j) % 16,
        lambda i, j: (i * 2 + j * 3) % 16,
        lambda i, j: (i * 3 + j) % 16,
        lambda i, j: (i * j + i + j) % 16,
        lambda i, j: (i * 7 + j * 5) % 17,
        lambda i, j: (i * 6 + j * 4) % 17,
    ]
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        
        colors = [0] * 120
        for idx, (i, j) in enumerate(edges):
            colors[idx] = strategy(i, j)
        
        col_arr = np.array(colors, dtype=np.int32)
        if is_valid_fast(col_arr):
            c = count_colors(colors)
            if c < best_count:
                best_count = c
                best_colors = colors[:]
    
    if best_colors is None:
        # Fallback to simple coloring
        colors = [0] * 120
        for idx, (i, j) in enumerate(edges):
            colors[idx] = (i + j) % 17
        return colors
    
    colors = best_colors[:]
    used_colors = set(colors)
    best_count = len(used_colors)
    best_col_arr = np.array(colors, dtype=np.int32)
    
    # Build color-to-edge-index mapping for fast merging
    def build_color_map(col_list):
        cmap = {}
        for idx, c in enumerate(col_list):
            if c not in cmap:
                cmap[c] = []
            cmap[c].append(idx)
        return cmap
    
    color_map = build_color_map(colors)
    
    # Merging loop with restarts
    max_restarts = 500
    restarts = 0
    
    while time.time() < deadline and restarts < max_restarts:
        merged_any = False
        current_colors_list = sorted(used_colors)
        
        # Sort colors by frequency (ascending) - try to eliminate rare colors first
        color_freq = {c: len(color_map.get(c, [])) for c in current_colors_list}
        current_colors_list.sort(key=lambda c: color_freq.get(c, 0))
        
        # Try all pairs (u, v) with u != v
        for u in current_colors_list:
            if time.time() > deadline:
                break
            if u not in used_colors:
                continue
            
            for v in current_colors_list:
                if u == v:
                    continue
                if v not in used_colors:
                    continue
                if time.time() > deadline:
                    break
                
                # Create a temporary copy
                temp_colors = colors[:]
                # Find edges with color u and change to v
                for k in color_map.get(u, []):
                    temp_colors[k] = v
                        
                # Check validity
                temp_arr = np.array(temp_colors, dtype=np.int32)
                if is_valid_fast(temp_arr):
                    # Accept merge
                    colors = temp_colors
                    used_colors.discard(u)
                    color_map = build_color_map(colors)
                    merged_any = True
                    break
            
            if merged_any:
                break
            if time.time() > deadline:
                break
                
        if not merged_any:
            # No simple merge possible.
            # Try targeted perturbation: change edges in tight K4s
            restarts += 1
            if restarts < max_restarts and time.time() < deadline:
                col_arr = np.array(colors, dtype=np.int32)
                tight_k4s = get_tight_k4s(col_arr)
                
                if len(tight_k4s) > 0:
                    # Pick a random tight K4 and perturb one of its edges
                    tight_idx = tight_k4s[random.randint(0, len(tight_k4s) - 1)]
                    k4_edges = k4_edge_indices[tight_idx]
                    edge_idx = k4_edges[random.randint(0, 5)]
                    old_color = colors[edge_idx]
                    
                    # Try changing to a color that doesn't create a 4-color K4
                    # Pick from existing colors or introduce a new one
                    candidates = list(used_colors)
                    if random.random() < 0.1:
                        # Sometimes try a new color
                        new_color = max(used_colors) + 1
                        candidates.append(new_color)
                    
                    new_color = random.choice(candidates)
                    if new_color != old_color:
                        colors[edge_idx] = new_color
                        used_colors.add(new_color)
                        if old_color not in colors:
                            used_colors.discard(old_color)
                        color_map = build_color_map(colors)
                else:
                    # No tight K4s, do random perturbation
                    num_perturb = random.randint(1, 3)
                    for _ in range(num_perturb):
                        edge_idx = random.randint(0, 119)
                        old_color = colors[edge_idx]
                        new_color = random.choice(list(used_colors))
                        if new_color != old_color:
                            colors[edge_idx] = new_color
                            used_colors.add(new_color)
                            if old_color not in colors:
                                used_colors.discard(old_color)
                    color_map = build_color_map(colors)
                
                # Try to merge again after perturbation
                current_colors_list = sorted(used_colors)
                color_freq = {c: len(color_map.get(c, [])) for c in current_colors_list}
                current_colors_list.sort(key=lambda c: color_freq.get(c, 0))
                
                for u in current_colors_list:
                    if time.time() > deadline:
                        break
                    if u not in used_colors:
                        continue
                    
                    for v in current_colors_list:
                        if u == v:
                            continue
                        if v not in used_colors:
                            continue
                        if time.time() > deadline:
                            break
                        
                        temp_colors = colors[:]
                        for k in color_map.get(u, []):
                            temp_colors[k] = v
                            
                        temp_arr = np.array(temp_colors, dtype=np.int32)
                        if is_valid_fast(temp_arr):
                            colors = temp_colors
                            used_colors.discard(u)
                            color_map = build_color_map(colors)
                            merged_any = True
                            break
                    
                    if merged_any:
                        break
                    if time.time() > deadline:
                        break
        
        # Update best
        current_count = len(used_colors)
        if current_count < best_count:
            best_count = current_count
            best_colors = colors[:]
            best_col_arr = np.array(colors, dtype=np.int32)
        
        if not merged_any and restarts >= max_restarts:
            break

    # Final check
    if best_colors is not None:
        final_arr = np.array(best_colors, dtype=np.int32)
        if is_valid_fast(final_arr):
            return best_colors
    
    # Fallback
    return colors
