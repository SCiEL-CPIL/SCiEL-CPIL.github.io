import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate base solution (17 colors)
    # Edge index mapping: (i, j) -> idx
    def get_idx(i, j):
        return i * 15 - i * (i - 1) // 2 + (j - i - 1)
        
    # Precompute edges
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            edges.append((i, j))
            
    # Map edge index to (u, v)
    edge_uv = [None] * 120
    for idx, (i, j) in enumerate(edges):
        edge_uv[idx] = (i, j)

    # Map edge (u, v) to index
    edge_idx_map = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx_map[(i, j)] = idx
        edge_idx_map[(j, i)] = idx

    # Precompute for each edge, the list of K4s it belongs to
    edge_k4s = [[] for _ in range(120)]
    k4s_all = []
    
    # Generate all K4s (combinations of 4 vertices)
    for quad in combinations(range(16), 4):
        a, b, c, d = quad
        e_indices = [
            edge_idx_map[(a, b)],
            edge_idx_map[(a, c)],
            edge_idx_map[(a, d)],
            edge_idx_map[(b, c)],
            edge_idx_map[(b, d)],
            edge_idx_map[(c, d)]
        ]
        k4s_all.append(e_indices)
        for ei in e_indices:
            edge_k4s[ei].append(len(k4s_all) - 1)

    # Initialize colors with the seed construction: (i + j) % 17
    colors = [0] * 120
    for idx, (i, j) in enumerate(edges):
        colors[idx] = (i + j) % 17
        
    def is_valid_k4(k4_idx, cols):
        e_indices = k4s_all[k4_idx]
        c0 = cols[e_indices[0]]
        c1 = cols[e_indices[1]]
        c2 = cols[e_indices[2]]
        c3 = cols[e_indices[3]]
        c4 = cols[e_indices[4]]
        c5 = cols[e_indices[5]]
        
        # Check if at least 5 distinct
        # Fast path: if all 6 are distinct, return True
        # If 5 are distinct, return True
        # We can just use a set for simplicity first, optimize if slow
        # But 1820 K4s * 91 edges is too many checks if we do full scan.
        # We only check affected K4s.
        
        # Optimization: manual check for 5 distinct
        # If c0==c1==c2==c3==c4==c5 -> 1 color (Fail)
        # ...
        # Let's stick to set for correctness, but maybe optimize later.
        return len({c0, c1, c2, c3, c4, c5}) >= 5

    def is_valid_all(cols):
        for k4_idx in range(len(k4s_all)):
            if not is_valid_k4(k4_idx, cols):
                return False
        return True
        
    # Local Search + Color Merging Strategy
    best_colors = colors[:]
    best_num_colors = len(set(colors))
    
    # Helper to check if merging color 'from_c' to 'to_c' is valid
    # We only need to check edges that have color 'from_c'
    def try_merge_color(from_c, to_c, cols):
        # Find all edges with color from_c
        affected_edges = []
        for i in range(120):
            if cols[i] == from_c:
                affected_edges.append(i)
                
        if not affected_edges:
            return True, cols
            
        # Temporarily change colors
        changed_indices = []
        for e in affected_edges:
            changed_indices.append(e)
            cols[e] = to_c
            
        # Check validity for all K4s that contain any of these edges
        # A K4 is invalid if it has < 5 distinct colors
        # We need to check K4s that intersect the set of affected edges.
        # Actually, a K4 is affected if it contains at least one edge with color from_c (now to_c).
        # But changing a color might *fix* a K4 (if it had 4 colors and now has 5) or *break* it (if it had 5 and now has 4).
        # So we must check all K4s that contain any of the affected edges.
        
        affected_k4s = set()
        for e in affected_edges:
            for k4_idx in edge_k4s[e]:
                affected_k4s.add(k4_idx)
                
        for k4_idx in affected_k4s:
            if not is_valid_k4(k4_idx, cols):
                # Revert
                for e in changed_indices:
                    cols[e] = from_c
                return False, cols
                
        return True, cols

    # Main Loop
    while time.time() < deadline:
        # Strategy 1: Try to merge colors
        current_colors = set(colors)
        if len(current_colors) > 1:
            # Pick two distinct colors to merge
            # Prefer merging a color that appears less frequently?
            # Or just random.
            color_list = list(current_colors)
            random.shuffle(color_list)
            from_c = color_list[0]
            to_c = color_list[1]
            
            # Try merge
            new_colors = colors[:]
            success, _ = try_merge_color(from_c, to_c, new_colors)
            
            if success:
                colors = new_colors
                num_c = len(set(colors))
                if num_c < best_num_colors:
                    best_num_colors = num_c
                    best_colors = colors[:]
                continue # Restart merge attempt with new state
        
        # Strategy 2: Local Search (Random Walk)
        # If we can't merge, perturb the solution to allow future merges
        e_idx = random.randint(0, 119)
        old_color = colors[e_idx]
        
        # Pick a new color from the current palette
        current_palette = list(set(colors))
        new_color = random.choice(current_palette)
        
        if new_color == old_color:
            continue
            
        colors[e_idx] = new_color
        
        # Check validity
        valid = True
        for k4_idx in edge_k4s[e_idx]:
            if not is_valid_k4(k4_idx, colors):
                valid = False
                break
        
        if not valid:
            colors[e_idx] = old_color

    # Final compression
    unique_colors = sorted(list(set(best_colors)))
    mapping = {c: i for i, c in enumerate(unique_colors)}
    final_colors = [mapping[c] for c in best_colors]
    return final_colors

# EVOLVE-BLOCK-END
