import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Edge index mapping: (i, j) -> idx
    def get_idx(i, j):
        return i * 15 - i * (i - 1) // 2 + (j - i - 1)
        
    # Precompute edges
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            edges.append((i, j))
            
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
        
    # For penalty-based local search:
    # For each K4, track the number of distinct colors (0-6)
    # A K4 is "satisfied" if it has >= 5 distinct colors
    # Penalty = number of unsatisfied K4s
    
    # Precompute for each K4, the 6 edge indices
    k4_edges = k4s_all  # k4_edges[k4_idx] = list of 6 edge indices
    
    # Initialize color counts per K4
    # For each K4, maintain a dict: color -> count
    k4_color_counts = [{} for _ in range(len(k4s_all))]
    k4_distinct = [0] * len(k4s_all)
    k4_satisfied = [False] * len(k4s_all)
    
    def init_k4_state():
        for k4_idx in range(len(k4s_all)):
            e_indices = k4_edges[k4_idx]
            counts = {}
            for ei in e_indices:
                c = colors[ei]
                counts[c] = counts.get(c, 0) + 1
            k4_color_counts[k4_idx] = counts
            k4_distinct[k4_idx] = len(counts)
            k4_satisfied[k4_idx] = k4_distinct[k4_idx] >= 5
    
    def update_k4(k4_idx):
        """Recompute distinct count for a K4 from scratch (used for verification)"""
        e_indices = k4_edges[k4_idx]
        counts = {}
        for ei in e_indices:
            c = colors[ei]
            counts[c] = counts.get(c, 0) + 1
        k4_color_counts[k4_idx] = counts
        k4_distinct[k4_idx] = len(counts)
        k4_satisfied[k4_idx] = k4_distinct[k4_idx] >= 5
    
    def update_k4_incremental(k4_idx, old_color, new_color):
        """Update K4 state when one edge's color changes from old_color to new_color"""
        counts = k4_color_counts[k4_idx]
        # Remove old color
        if old_color in counts:
            counts[old_color] -= 1
            if counts[old_color] == 0:
                del counts[old_color]
        # Add new color
        if new_color not in counts:
            k4_distinct[k4_idx] += 1
        counts[new_color] = counts.get(new_color, 0) + 1
        k4_satisfied[k4_idx] = k4_distinct[k4_idx] >= 5
    
    def get_penalty():
        """Total penalty = number of unsatisfied K4s"""
        return sum(1 for s in k4_satisfied if not s)
    
    init_k4_state()
    
    # Track best solution
    best_colors = colors[:]
    best_penalty = get_penalty()
    best_num_colors = len(set(colors))
    
    # Local search with penalty-based acceptance
    # We allow moves that increase penalty if they reduce distinct colors
    # This is a multi-objective: minimize penalty, then minimize distinct colors
    
    while time.time() < deadline:
        # Pick a random edge
        e_idx = random.randint(0, 119)
        old_color = colors[e_idx]
        
        # Determine candidate colors
        # Bias towards colors that are already used to reduce distinct count
        used_colors = list(set(colors))
        # With some probability, try a new low color
        if random.random() < 0.3:
            new_color = random.randint(0, min(12, len(used_colors) + 2))
        else:
            new_color = random.choice(used_colors)
        
        if new_color == old_color:
            continue
            
        colors[e_idx] = new_color
        
        # Update all affected K4s
        old_penalty = get_penalty()
        for k4_idx in edge_k4s[e_idx]:
            update_k4_incremental(k4_idx, old_color, new_color)
        new_penalty = get_penalty()
        
        # Acceptance criterion:
        # 1. If penalty decreases, always accept
        # 2. If penalty same, accept if distinct colors decreases
        # 3. If penalty increases, accept with small probability (simulated annealing style)
        
        cur_distinct = len(set(colors))
        
        if new_penalty < old_penalty:
            # Better penalty, accept
            pass
        elif new_penalty == old_penalty:
            if cur_distinct < best_num_colors:
                # Same penalty, fewer colors, accept
                pass
            else:
                # Revert
                colors[e_idx] = old_color
                for k4_idx in edge_k4s[e_idx]:
                    update_k4_incremental(k4_idx, new_color, old_color)
                continue
        else:
            # Penalty increased
            # Accept with small probability to escape local optima
            if random.random() < 0.05:
                pass
            else:
                # Revert
                colors[e_idx] = old_color
                for k4_idx in edge_k4s[e_idx]:
                    update_k4_incremental(k4_idx, new_color, old_color)
                continue
        
        # Update best if improved
        if new_penalty == 0:
            # Valid coloring
            if cur_distinct < best_num_colors:
                best_num_colors = cur_distinct
                best_colors = colors[:]
        elif new_penalty < best_penalty:
            best_penalty = new_penalty
            # Also track colors for this state
            if cur_distinct <= best_num_colors:
                best_colors = colors[:]
                best_num_colors = cur_distinct

    # Final: use best_colors if it's valid, otherwise use current
    # Verify best_colors
    # Recompute penalty for best_colors
    saved_colors = colors[:]
    colors[:] = best_colors[:]
    init_k4_state()
    final_penalty = get_penalty()
    
    if final_penalty == 0:
        final_colors = best_colors
    else:
        # best_colors not valid, try current
        colors[:] = saved_colors
        init_k4_state()
        if get_penalty() == 0:
            final_colors = saved_colors
        else:
            # Neither is valid, use best_colors and hope (shouldn't happen if base is valid)
            final_colors = best_colors
    
    # Compact labels
    unique_colors = sorted(list(set(final_colors)))
    mapping = {c: i for i, c in enumerate(unique_colors)}
    final_compact = [mapping[c] for c in final_colors]
    return final_compact

# EVOLVE-BLOCK-END
