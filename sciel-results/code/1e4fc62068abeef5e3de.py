import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # 1. Generate base solution (17 colors)
    # Edge index mapping: (i, j) -> idx
    # i, j in 0..15, i < j
    def get_idx(i, j):
        # Number of edges with first vertex < i: i * (15 - i) / 2 ? No.
        # Sum_{k=0}^{i-1} (15 - k) = 15i - i(i-1)/2
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
    # A K4 is defined by 4 vertices {a, b, c, d}. 
    # It contains 6 edges.
    # For a given edge (u, v), the other 2 vertices (w, x) must be chosen from the remaining 14 vertices.
    # There are C(14, 2) = 91 such K4s.
    # We need to check if these K4s have >= 5 distinct colors.
    
    # To speed up checking, we can store the K4s as tuples of 6 edge indices.
    edge_k4s = [[] for _ in range(120)]
    k4s_all = []
    
    # Generate all K4s (combinations of 4 vertices)
    # 1820 K4s
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
        
    # Check if base is valid (it should be)
    def is_valid_k4(k4_idx, cols):
        e_indices = k4s_all[k4_idx]
        c_set = set()
        for ei in e_indices:
            c_set.add(cols[ei])
            if len(c_set) >= 5:
                return True
        return len(c_set) >= 5

    # Full validation (for debugging or initial check)
    def is_valid_all(cols):
        for k4_idx in range(len(k4s_all)):
            if not is_valid_k4(k4_idx, cols):
                return False
        return True
        
    # Local Search
    # Objective: Minimize number of distinct colors.
    
    best_colors = colors[:]
    best_num_colors = len(set(colors))
    
    # Maintain color counts
    from collections import defaultdict
    color_counts = defaultdict(int)
    for c in colors:
        color_counts[c] += 1
    
    # Helper to update counts
    def update_counts(e_idx, old_c, new_c):
        color_counts[old_c] -= 1
        if color_counts[old_c] == 0:
            del color_counts[old_c]
        color_counts[new_c] += 1

    # Time loop
    while time.time() < deadline:
        # Pick a random edge
        e_idx = random.randint(0, 119)
        old_color = colors[e_idx]
        
        # Strategy:
        # 1. If we want to reduce colors, we should try to map an edge to an existing color.
        # 2. Specifically, try to map edges with high-frequency or high-index colors to others.
        
        # Get list of currently used colors
        used_colors = list(color_counts.keys())
        
        if len(used_colors) <= 1:
            break
            
        # Choose a candidate new color
        # Bias towards colors that are already used to consolidate.
        # We pick a random used color different from old_color.
        if len(used_colors) == 1:
            new_color = used_colors[0]
        else:
            # Pick a random color from used colors, ensuring it's not old_color if possible
            # But if old_color is the only one, we can't change.
            # We want to try to *remove* old_color eventually.
            # So we pick a target color from the set of used colors.
            # If we pick old_color, we do nothing.
            # So we pick from used_colors excluding old_color?
            # If we exclude, we force a change. This might be too aggressive if no change is valid.
            # Let's pick a random used color. If it's old_color, skip.
            new_color = random.choice(used_colors)
            if new_color == old_color:
                # Try another one? Or just skip?
                # Let's just pick a different one if possible
                if len(used_colors) > 1:
                    new_color = random.choice([c for c in used_colors if c != old_color])
                else:
                    continue

        # Apply change
        colors[e_idx] = new_color
        
        # Check validity
        valid = True
        for k4_idx in edge_k4s[e_idx]:
            if not is_valid_k4(k4_idx, colors):
                valid = False
                break
        
        if valid:
            # Accept
            update_counts(e_idx, old_color, new_color)
            
            # Check if we reduced distinct colors
            current_num_colors = len(color_counts)
            if current_num_colors < best_num_colors:
                best_num_colors = current_num_colors
                best_colors = colors[:]
                
                # Optional: Try to compact immediately? No, keep raw for now.
                # We can also try to aggressively eliminate the max color next.
        else:
            # Revert
            colors[e_idx] = old_color

    # Final compression
    unique_colors = sorted(list(set(best_colors)))
    mapping = {c: i for i, c in enumerate(unique_colors)}
    final_colors = [mapping[c] for c in best_colors]
    return final_colors

# EVOLVE-BLOCK-END
