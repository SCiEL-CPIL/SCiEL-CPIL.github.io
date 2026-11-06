"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Precompute all K4 subgraphs (indices of the 6 edges for each 4-tuple)
    n = 16
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[(i,j)] = idx
            idx += 1
    total_edges = 120
    
    # Precompute K4 edge index lists
    k4_edges = []
    for quad in itertools.combinations(range(n), 4):
        a,b,c,d = quad
        k4_edges.append([
            edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)],
            edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]
        ])
    num_k4 = len(k4_edges)
    
    # Convert to numpy for fast validation
    k4_arr = np.array(k4_edges, dtype=np.int32)  # (num_k4, 6)
    
    def count_colors(coloring):
        """Count distinct colors used."""
        return len(set(coloring))
    
    def is_valid(coloring):
        """Check if every K4 has at least 5 distinct colors."""
        col_arr = np.array(coloring, dtype=np.int32)
        # For each K4, get the 6 edge colors
        colors_in_k4 = col_arr[k4_arr]  # (num_k4, 6)
        # Count distinct colors per K4: sort and count unique
        sorted_colors = np.sort(colors_in_k4, axis=1)
        # A K4 is invalid if it has < 5 distinct colors, i.e., >= 2 duplicates
        # Check: for each row, count number of positions where value equals previous
        # If any row has >= 2 such positions, it's invalid
        # Actually, we need < 5 distinct, meaning at least 2 pairs sharing a color
        # More precisely: distinct count < 5 means at least 2 edges share colors with others
        # Let's compute distinct count properly
        # For each row of 6 sorted values, count distinct
        distinct_counts = np.ones(num_k4, dtype=np.int32)
        for k in range(1, 6):
            distinct_counts += (sorted_colors[:, k] != sorted_colors[:, k-1]).astype(np.int32)
        return np.all(distinct_counts >= 5)
    
    def get_violations(coloring):
        """Return list of K4 indices that violate the condition."""
        col_arr = np.array(coloring, dtype=np.int32)
        colors_in_k4 = col_arr[k4_arr]
        sorted_colors = np.sort(colors_in_k4, axis=1)
        distinct_counts = np.ones(num_k4, dtype=np.int32)
        for k in range(1, 6):
            distinct_counts += (sorted_colors[:, k] != sorted_colors[:, k-1]).astype(np.int32)
        return np.where(distinct_counts < 5)[0]
    
    # Start with the known valid solution: (i+j) % 17
    best_coloring = [(i + j) % 17 for i in range(n) for j in range(i+1, n)]
    best_colors = count_colors(best_coloring)
    
    # Try to reduce colors by local search
    # Strategy: try merging colors (reassigning some edges to reduce palette)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        if best_colors <= 13:
            break
        
        # Start from best coloring
        coloring = list(best_coloring)
        current_colors = best_colors
        
        # Try random local moves: pick an edge and change its color to another used color
        for step in range(2000):
            if time.time() > deadline:
                break
            
            # Pick a random edge
            e = random.randint(0, total_edges - 1)
            old_color = coloring[e]
            
            # Try to change to a different color that's already in use
            used_colors = list(set(coloring))
            if len(used_colors) < 2:
                continue
            
            # Pick a target color different from old_color
            candidates = [c for c in used_colors if c != old_color]
            if not candidates:
                continue
            new_color = random.choice(candidates)
            
            coloring[e] = new_color
            
            # Check validity
            if is_valid(coloring):
                new_num_colors = count_colors(coloring)
                if new_num_colors < current_colors:
                    current_colors = new_num_colors
                    if current_colors < best_colors:
                        best_coloring = list(coloring)
                        best_colors = current_colors
                        if best_colors <= 13:
                            break
                # Keep the change even if not better (for exploration)
            else:
                # Revert
                coloring[e] = old_color
        
        if best_colors <= 13:
            break
    
    # If we still have > 15 colors, fall back to the guaranteed valid solution
    if not is_valid(best_coloring):
        best_coloring = [(i + j) % 17 for i in range(n) for j in range(i+1, n)]
    
    return best_coloring
# EVOLVE-BLOCK-END
