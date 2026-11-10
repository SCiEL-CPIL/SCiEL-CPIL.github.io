"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    """
    Construct an edge-coloring of K16 minimizing colors used
    such that every K4 has at least 5 distinct colors.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Precompute all K4s and their edge indices
    k4_edges = []
    for quad in combinations(range(n), 4):
        edges = []
        for a, b in combinations(quad, 2):
            idx = a * (n - 1) - a * (a + 1) // 2 + (b - a - 1)
            edges.append(idx)
        k4_edges.append(edges)
    
    num_k4s = len(k4_edges)
    
    # Helper to convert edge index to (i, j)
    def edge_to_pair(idx):
        # Inverse of the indexing scheme:
        # For fixed i, j ranges from i+1 to n-1
        # idx = i*(n-1) - i*(i+1)//2 + (j-i-1)
        # Let's find i such that i*(n-1) - i*(i+1)//2 <= idx < (i+1)*(n-1) - (i+1)*(i+2)//2
        i = 0
        while i < n - 1:
            start_i = i * (n - 1) - i * (i + 1) // 2
            end_i = (i + 1) * (n - 1) - (i + 1) * (i + 2) // 2
            if start_i <= idx < end_i:
                j = i + 1 + (idx - start_i)
                return (i, j)
            i += 1
        return (0, 1)  # fallback
    
    # Helper to check validity of a coloring
    def is_valid(coloring):
        for edges in k4_edges:
            colors_in_k4 = set(coloring[e] for e in edges)
            if len(colors_in_k4) < 5:
                return False
        return True
    
    # Helper to count distinct colors used
    def count_colors(coloring):
        return len(set(coloring))
    
    # Start with the seed solution: (i+j) % 17
    def seed_coloring():
        return np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    
    best_coloring = seed_coloring()
    best_score = count_colors(best_coloring)  # 17
    
    # Try to improve by reducing colors
    # Strategy: try to merge colors one by one
    
    def try_merge_colors(coloring, max_colors, max_iters=1000):
        """Try to reduce number of colors by merging."""
        coloring = coloring.copy()
        current_colors = sorted(set(coloring))
        
        if len(current_colors) <= max_colors:
            return coloring, len(current_colors)
        
        # Build color-to-edges mapping
        for _ in range(max_iters):
            current_colors = sorted(set(coloring))
            if len(current_colors) <= max_colors:
                break
            
            # Try to merge the two least frequent colors
            from collections import Counter
            color_counts = Counter(coloring)
            # Sort by count, try merging the two least frequent
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1])
            
            merged = False
            for c1, count1 in sorted_colors[:5]:
                for c2, count2 in sorted_colors[5:10]:
                    if c1 == c2:
                        continue
                    
                    # Try merging c2 into c1
                    test_coloring = coloring.copy()
                    test_coloring[test_coloring == c2] = c1
                    
                    if is_valid(test_coloring):
                        coloring = test_coloring
                        merged = True
                        break
                if merged:
                    break
            
            if not merged:
                break
        
        return coloring, count_colors(coloring)
    
    # Start with seed and try to reduce
    current = best_coloring.copy()
    
    # Try reducing from 17 down
    for target in range(best_score - 1, 12, -1):  # Try 16, 15, 14, 13
        if time.time() > deadline - 30:
            break
        
        # Reset to best known
        current = best_coloring.copy()
        
        # Try to merge colors to reach target
        improved, score = try_merge_colors(current, target, max_iters=500)
        
        if score < best_score:
            best_coloring = improved
            best_score = score
    
    # If we have time, try local search
    if time.time() < deadline - 20 and best_score > 13:
        # Local search: randomly change colors and see if we can improve
        rng = random.Random(42)
        current = best_coloring.copy()
        current_score = best_score
        
        for _ in range(10000):
            if time.time() > deadline - 5:
                break
            
            # Pick a random edge and try a random color
            edge_idx = rng.randint(0, num_edges - 1)
            old_color = current[edge_idx]
            new_color = rng.randint(0, current_score + 2)
            
            if new_color == old_color:
                continue
            
            current[edge_idx] = new_color
            
            # Check if still valid (quick check: only affected K4s)
            # For simplicity, do full check but with early exit
            valid = True
            for edges in k4_edges:
                if edge_idx in edges:
                    colors_in_k4 = set(current[e] for e in edges)
                    if len(colors_in_k4) < 5:
                        valid = False
                        break
            
            if valid:
                new_score = count_colors(current)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_coloring = current.copy()
                        best_score = current_score
            else:
                current[edge_idx] = old_color
    
    # Final check
    if not is_valid(best_coloring):
        best_coloring = seed_coloring()
        best_score = count_colors(best_coloring)
    
    return [int(c) for c in best_coloring]
# EVOLVE-BLOCK-END
