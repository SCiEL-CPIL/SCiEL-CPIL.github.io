import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n = 16
    # Generate all edges
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Initial coloring: (i + j) % 17
    init_colors = [(i + j) % 17 for i, j in edges]
    current_colors = init_colors[:]
    
    # Precompute K4s and their edge indices
    # k4s is a list of tuples of 4 vertices
    # k4_edge_indices is a list of lists of edge indices
    # We need a map from (u, v) to edge index
    edge_index_map = {}
    for idx, (u, v) in enumerate(edges):
        edge_index_map[(u, v)] = idx
        edge_index_map[(v, u)] = idx
        
    k4s = []
    k4_edge_indices = []
    for quad in itertools.combinations(range(n), 4):
        k4s.append(quad)
        idxs = []
        for u, v in itertools.combinations(quad, 2):
            idxs.append(edge_index_map[(u, v)])
        k4_edge_indices.append(idxs)
        
    num_k4s = len(k4s)
    
    # Initialize color counts for each K4
    # k4_color_counts[k] is a dictionary {color: count}
    k4_color_counts = [{} for _ in range(num_k4s)]
    
    for k_idx, edge_idxs in enumerate(k4_edge_indices):
        counts = {}
        for e_idx in edge_idxs:
            c = current_colors[e_idx]
            counts[c] = counts.get(c, 0) + 1
        k4_color_counts[k_idx] = counts

    def is_valid(k_idx):
        counts = k4_color_counts[k_idx]
        distinct = sum(1 for c in counts.values() if c > 0)
        return distinct >= 5

    def check_all_valid():
        for k_idx in range(num_k4s):
            if not is_valid(k_idx):
                return False
        return True

    # Helper to update counts when edge e_idx changes from old_c to new_c
    def update_counts(e_idx, old_c, new_c):
        # Find all K4s containing edge e_idx
        # This is slow if done from scratch. 
        # Precompute edge_to_k4s
        # But we can just iterate all K4s? 1820 * 6 = 10920 ops per change. 
        # 10000 changes * 10920 = 1e8 ops. Might be slow in Python.
        # Better to precompute edge_to_k4s.
        pass

    # Precompute edge_to_k4s
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k_idx, edge_idxs in enumerate(k4_edge_indices):
        for e_idx in edge_idxs:
            edge_to_k4s[e_idx].append(k_idx)

    def update_counts_fast(e_idx, old_c, new_c):
        for k_idx in edge_to_k4s[e_idx]:
            counts = k4_color_counts[k_idx]
            # Decrement old
            if old_c in counts:
                counts[old_c] -= 1
                if counts[old_c] == 0:
                    del counts[old_c]
            # Increment new
            if new_c not in counts:
                counts[new_c] = 1
            else:
                counts[new_c] += 1

    # Local Search to minimize colors
    # Strategy: Try to eliminate the color with the fewest edges.
    # Or just random hill climbing on the number of colors used.
    
    best_colors = current_colors[:]
    best_num_colors = len(set(best_colors))
    
    def get_num_colors(colors):
        return len(set(colors))
    
    while time.time() < deadline:
        # Try to improve
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        old_c = current_colors[e_idx]
        
        # Try to change to a color that is already used, or a new color?
        # To minimize, we want to reduce the set of used colors.
        # So we should try to change an edge of a "rare" color to a "common" color.
        
        # Let's try a move: change color of e_idx to a random color from the current palette.
        palette = list(set(current_colors))
        if not palette:
            break
            
        new_c = random.choice(palette)
        if new_c == old_c:
            continue
            
        # Apply change
        update_counts_fast(e_idx, old_c, new_c)
        current_colors[e_idx] = new_c
        
        # Check validity of affected K4s
        valid = True
        for k_idx in edge_to_k4s[e_idx]:
            if not is_valid(k_idx):
                valid = False
                break
                
        if valid:
            # Accept move
            new_num_colors = get_num_colors(current_colors)
            if new_num_colors < best_num_colors:
                best_colors = current_colors[:]
                best_num_colors = new_num_colors
        else:
            # Revert
            update_counts_fast(e_idx, new_c, old_c)
            current_colors[e_idx] = old_c
            
        # Occasionally, try to eliminate a color completely
        # Pick a color with minimum frequency
        # Recolor all its edges to random other colors
        # This is a bigger move.
        
    # Ensure we return the best found
    if get_num_colors(best_colors) < get_num_colors(current_colors):
        return best_colors
    else:
        return current_colors
