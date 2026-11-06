"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Precompute all edges in lexicographic order
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Build edge index lookup
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
    
    # Precompute all K4s and their edge sets
    k4s = list(combinations(range(n), 4))
    k4_edges = []
    for k4 in k4s:
        e_list = []
        for a, b in combinations(k4, 2):
            e_list.append(edge_idx[(a, b)])
        k4_edges.append(e_list)
    
    # Start with the cyclic construction
    colors = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    
    def count_distinct_colors(c):
        return len(set(c.tolist()))
    
    def is_valid(c):
        # Check every K4 has at least 5 distinct colors
        for e_list in k4_edges:
            distinct = set(c[e_list].tolist())
            if len(distinct) < 5:
                return False
        return True
    
    def try_merge(c, color_a, color_b):
        """Try to merge color_b into color_a, return new array or None if invalid."""
        if color_a == color_b:
            return None
        new_c = c.copy()
        new_c[new_c == color_b] = color_a
        if is_valid(new_c):
            return new_c
        return None
    
    best_colors = colors.copy()
    best_count = count_distinct_colors(best_colors)
    
    # Try to reduce colors by merging
    # Strategy: repeatedly try to merge the least-used color into another
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        current_colors = set(best_colors.tolist())
        if len(current_colors) <= 13:
            break
        
        # Count frequency of each color
        color_freq = {}
        for c in best_colors.tolist():
            color_freq[c] = color_freq.get(c, 0) + 1
        
        # Sort colors by frequency (try to merge rarest first)
        sorted_colors = sorted(color_freq.keys(), key=lambda x: color_freq[x])
        
        merged = False
        for target in sorted_colors:
            if time.time() > deadline:
                break
            if target not in color_freq:
                continue
            # Try to merge target into each other color
            for donor in sorted_colors:
                if donor == target:
                    continue
                if donor not in color_freq:
                    continue
                new_c = try_merge(best_colors, donor, target)
                if new_c is not None:
                    best_colors = new_c
                    best_count = count_distinct_colors(best_colors)
                    merged = True
                    break
            if merged:
                break
        
        if not merged:
            # Try random perturbations
            if random.random() < 0.5:
                # Randomly reassign a few edges to existing colors
                num_reassign = random.randint(1, 5)
                new_c = best_colors.copy()
                existing = set(best_colors.tolist())
                for _ in range(num_reassign):
                    edge_pos = random.randint(0, num_edges - 1)
                    if existing:
                        new_c[edge_pos] = random.choice(list(existing))
                if is_valid(new_c):
                    best_colors = new_c
                    best_count = count_distinct_colors(best_colors)
    
    # Final check
    if not is_valid(best_colors):
        # Fallback to original
        best_colors = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    
    return best_colors.tolist()
# EVOLVE-BLOCK-END
