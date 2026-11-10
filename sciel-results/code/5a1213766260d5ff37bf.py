"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from collections import Counter

def construct():
    """
    Construct an edge-coloring of K16 minimizing colors used
    such that every K4 has at least 5 distinct colors.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
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
    
    # Precompute which K4s contain each edge for fast validity checking
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
    
    # Helper to check validity of a coloring using precomputed K4-edge mapping
    def is_valid(coloring):
        for edges in k4_edges:
            colors_in_k4 = set(coloring[e] for e in edges)
            if len(colors_in_k4) < 5:
                return False
        return True
    
    # Helper to check if changing edge_idx to new_color keeps validity
    # Only checks K4s that contain edge_idx
    def is_valid_after_change(coloring, edge_idx, new_color):
        for k4_idx in edge_to_k4s[edge_idx]:
            edges = k4_edges[k4_idx]
            colors_in_k4 = set()
            for e in edges:
                if e == edge_idx:
                    colors_in_k4.add(new_color)
                else:
                    colors_in_k4.add(coloring[e])
            if len(colors_in_k4) < 5:
                return False
        return True
    
    # Faster validity check using color count arrays
    # color_count[k4_idx][c] = number of edges in K4 k4_idx with color c
    def build_color_counts(coloring):
        counts = [np.zeros(120, dtype=np.int32) for _ in range(num_k4s)]
        for k4_idx, edges in enumerate(k4_edges):
            for e in edges:
                counts[k4_idx][coloring[e]] += 1
        return counts
    
    def is_valid_after_change_fast(counts, coloring, edge_idx, old_color, new_color):
        if old_color == new_color:
            return True
        for k4_idx in edge_to_k4s[edge_idx]:
            counts[k4_idx][old_color] -= 1
            counts[k4_idx][new_color] += 1
            # Count distinct colors in this K4
            distinct = 0
            for c in range(120):
                if counts[k4_idx][c] > 0:
                    distinct += 1
                    if distinct >= 5:
                        break
            if distinct < 5:
                # Undo
                counts[k4_idx][old_color] += 1
                counts[k4_idx][new_color] -= 1
                return False
        return True
    
    def count_colors(coloring):
        return len(set(coloring))
    
    # Track which colors are actually used for faster iteration
    def get_used_colors(coloring):
        return sorted(set(coloring))
    
    # Try multiple algebraic constructions
    def construction_sum_mod(k):
        """(i+j) % k"""
        return np.array([(i + j) % k for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    
    def construction_diff_mod(k):
        """(i-j) % k (adjusted for i<j)"""
        return np.array([(i - j) % k for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    
    def construction_product_mod(k):
        """(i*j) % k"""
        return np.array([(i * j) % k for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    
    def construction_mixed(i, j, k1, k2):
        """(i*a + j*b) % k"""
        a, b = k1, k2
        return np.array([(i * a + j * b) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    
    # Try various constructions and pick the best valid one
    candidates = []
    
    # (i+j) % k for various k
    for k in range(13, 18):
        coloring = construction_sum_mod(k)
        if is_valid(coloring):
            candidates.append(coloring)
    
    # (i-j) % k
    for k in range(13, 18):
        coloring = construction_diff_mod(k)
        if is_valid(coloring):
            candidates.append(coloring)
    
    # (i*j) % k
    for k in range(13, 18):
        coloring = construction_product_mod(k)
        if is_valid(coloring):
            candidates.append(coloring)
    
    # If no valid construction found, use (i+j)%17 as fallback
    if not candidates:
        candidates.append(construction_sum_mod(17))
    
    # Pick the one with fewest colors
    best_coloring = min(candidates, key=count_colors)
    best_score = count_colors(best_coloring)
    
    # Local search to try to reduce colors further
    if best_score > 13 and time.time() < deadline - 20:
        rng = random.Random(42)
        current = best_coloring.copy()
        current_score = best_score
        counts = build_color_counts(current)
        
        # Systematic color elimination: try to remove colors with fewest edges first
        used = get_used_colors(current)
        # Sort by frequency (ascending) to try eliminating rare colors first
        color_freq = {c: int(np.sum(current == c)) for c in used}
        for target_color in sorted(used, key=lambda c: color_freq[c]):
            if time.time() > deadline - 15:
                break
            
            edges_with_color = np.where(current == target_color)[0]
            if len(edges_with_color) == 0:
                continue
            
            # Try to reassign each edge of this color to another color
            success = True
            for edge_idx in edges_with_color:
                if time.time() > deadline - 15:
                    success = False
                    break
                
                old_color = current[edge_idx]
                if old_color != target_color:
                    continue
                
                # Try all other colors that are currently in use
                other_colors = [c for c in get_used_colors(current) if c != target_color]
                rng.shuffle(other_colors)
                
                reassigned = False
                for new_color in other_colors:
                    if is_valid_after_change_fast(counts, current, edge_idx, old_color, new_color):
                        current[edge_idx] = new_color
                        reassigned = True
                        break
                
                if not reassigned:
                    success = False
                    break
            
            if success:
                new_score = count_colors(current)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_coloring = current.copy()
                        best_score = current_score
                    # Rebuild counts for next iteration
                    counts = build_color_counts(current)
        
        # Random local search for further improvement
        if time.time() < deadline - 5 and best_score > 13:
            counts = build_color_counts(current)
            for iteration in range(50000):
                if time.time() > deadline - 3:
                    break
                
                edge_idx = rng.randint(0, num_edges - 1)
                old_color = current[edge_idx]
                
                # Try colors from 0 to current_score-1
                new_color = rng.randint(0, current_score - 1)
                
                if new_color == old_color:
                    continue
                
                if is_valid_after_change_fast(counts, current, edge_idx, old_color, new_color):
                    current[edge_idx] = new_color
                    new_score = count_colors(current)
                    if new_score < current_score:
                        current_score = new_score
                        if current_score < best_score:
                            best_coloring = current.copy()
                            best_score = current_score
                        # Rebuild counts when score improves
                        counts = build_color_counts(current)
                else:
                    pass  # counts already restored by is_valid_after_change_fast
    
    # Final validation
    if not is_valid(best_coloring):
        # Fallback to a known valid coloring
        best_coloring = construction_sum_mod(17)
        best_score = count_colors(best_coloring)
    
    return [int(c) for c in best_coloring]
# EVOLVE-BLOCK-END
