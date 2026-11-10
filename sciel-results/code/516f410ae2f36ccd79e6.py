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
    
    Strategy:
    1. Start with (i+j) % 17 coloring (known valid, 17 colors)
    2. Try structured variants with fewer colors (13-16)
    3. Use vectorized validity checks for speed
    4. Aggressively eliminate rare colors
    5. Local search to reduce color count
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n = 16
    num_edges = n * (n - 1) // 2  # 120

    # Precompute edge index mapping
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    # Precompute all K4s and their edge indices as numpy arrays
    k4_list = list(combinations(range(n), 4))
    num_k4s = len(k4_list)  # 1820
    
    # For each K4, store the 6 edge indices
    k4_edges_arr = np.zeros((num_k4s, 6), dtype=np.int32)
    for ki, quad in enumerate(k4_list):
        a, b, c, d = quad
        k4_edges_arr[ki, 0] = edge_idx(a, b)
        k4_edges_arr[ki, 1] = edge_idx(a, c)
        k4_edges_arr[ki, 2] = edge_idx(a, d)
        k4_edges_arr[ki, 3] = edge_idx(b, c)
        k4_edges_arr[ki, 4] = edge_idx(b, d)
        k4_edges_arr[ki, 5] = edge_idx(c, d)

    # For each edge, which K4s contain it (for incremental checks)
    edge_to_k4s = [[] for _ in range(num_edges)]
    for ki in range(num_k4s):
        for e in k4_edges_arr[ki]:
            edge_to_k4s[e].append(ki)

    def is_valid_fast(coloring):
        """Vectorized validity check: every K4 must have >= 5 distinct colors."""
        k4_colors = coloring[k4_edges_arr]  # shape (num_k4s, 6)
        sorted_k4 = np.sort(k4_colors, axis=1)  # (num_k4s, 6)
        diffs = np.diff(sorted_k4, axis=1)  # (num_k4s, 5)
        distinct_counts = 1 + np.sum(diffs != 0, axis=1)  # (num_k4s,)
        return np.all(distinct_counts >= 5)

    def count_colors(coloring):
        return len(set(coloring.tolist()))

    # Seed coloring: (i+j) % 17
    def seed_coloring():
        return np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)

    best_coloring = seed_coloring()
    best_score = count_colors(best_coloring)  # 17

    # Try alternative structured colorings with fewer colors
    structured_starts = []
    
    # Try (i+j+offset) % k for k in 13..16
    for k in range(13, 17):
        for offset in range(k):
            if time.time() > deadline - 60:
                break
            colors = np.array([((i + j) + offset) % k for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
            if is_valid_fast(colors):
                sc = count_colors(colors)
                structured_starts.append((sc, colors))
    
    # Try (j-i+offset) % k
    for k in range(13, 17):
        for offset in range(k):
            if time.time() > deadline - 60:
                break
            colors = np.array([((j - i) + offset) % k for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
            if is_valid_fast(colors):
                sc = count_colors(colors)
                structured_starts.append((sc, colors))
    
    # Try (i*j + i + j) % k
    for k in range(13, 17):
        for offset in range(k):
            if time.time() > deadline - 60:
                break
            colors = np.array([((i * j + i + j) + offset) % k for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
            if is_valid_fast(colors):
                sc = count_colors(colors)
                structured_starts.append((sc, colors))
    
    # Try (i^2 + j) % k
    for k in range(13, 17):
        for offset in range(k):
            if time.time() > deadline - 60:
                break
            colors = np.array([((i * i + j) + offset) % k for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
            if is_valid_fast(colors):
                sc = count_colors(colors)
                structured_starts.append((sc, colors))
    
    # Try (j^2 + i) % k
    for k in range(13, 17):
        for offset in range(k):
            if time.time() > deadline - 60:
                break
            colors = np.array([((j * j + i) + offset) % k for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
            if is_valid_fast(colors):
                sc = count_colors(colors)
                structured_starts.append((sc, colors))
    
    # Try (i + 2*j) % k
    for k in range(13, 17):
        for offset in range(k):
            if time.time() > deadline - 60:
                break
            colors = np.array([((i + 2 * j) + offset) % k for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
            if is_valid_fast(colors):
                sc = count_colors(colors)
                structured_starts.append((sc, colors))
    
    # Try (2*i + j) % k
    for k in range(13, 17):
        for offset in range(k):
            if time.time() > deadline - 60:
                break
            colors = np.array([((2 * i + j) + offset) % k for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
            if is_valid_fast(colors):
                sc = count_colors(colors)
                structured_starts.append((sc, colors))
    
    # Use the best structured start if better
    if structured_starts:
        structured_starts.sort(key=lambda x: x[0])
        if structured_starts[0][0] < best_score:
            best_score = structured_starts[0][0]
            best_coloring = structured_starts[0][1].copy()

    # Now try to reduce colors from the best starting point
    def try_reduce_colors(coloring, max_iters=3000):
        """Aggressively try to reduce the number of colors used."""
        coloring = coloring.copy()
        current_score = count_colors(coloring)
        
        for _ in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Get current color usage
            color_counts = Counter(coloring.tolist())
            current_colors = sorted(color_counts.keys())
            
            if len(current_colors) <= 13:
                break
            
            # Try to eliminate the rarest color
            rarest_color = min(current_colors, key=lambda c: color_counts[c])
            edges_with_rarest = np.where(coloring == rarest_color)[0]
            
            if len(edges_with_rarest) == 0:
                continue
            
            # Try to reassign all edges of rarest_color to other colors
            success = True
            for edge_idx in edges_with_rarest:
                found = False
                # Try colors in order of frequency (most common first for stability)
                for new_color in sorted(current_colors, key=lambda c: -color_counts[c]):
                    if new_color == rarest_color:
                        continue
                    test_coloring = coloring.copy()
                    test_coloring[edge_idx] = new_color
                    if is_valid_fast(test_coloring):
                        coloring[edge_idx] = new_color
                        found = True
                        break
                if not found:
                    success = False
                    break
            
            if success:
                current_score = count_colors(coloring)
            else:
                # Try a more targeted approach: reassign one edge at a time
                # to a color that creates the least "pressure"
                found_any = False
                for edge_idx in edges_with_rarest:
                    found = False
                    for new_color in current_colors:
                        if new_color == rarest_color:
                            continue
                        test_coloring = coloring.copy()
                        test_coloring[edge_idx] = new_color
                        if is_valid_fast(test_coloring):
                            coloring[edge_idx] = new_color
                            found = True
                            found_any = True
                            break
                    if found:
                        # Check if we can now eliminate the color
                        remaining = np.where(coloring == rarest_color)[0]
                        if len(remaining) == 0:
                            current_score = count_colors(coloring)
                            break
                if not found_any:
                    break
        
        return coloring, current_score

    # Try reducing from our best starting point
    current = best_coloring.copy()
    reduced, score = try_reduce_colors(current, max_iters=2000)
    if score < best_score:
        best_coloring = reduced
        best_score = score

    # Local search: try to further reduce by swapping colors
    if time.time() < deadline - 25 and best_score > 13:
        rng = random.Random(42)
        current = best_coloring.copy()
        current_score = best_score
        
        for _ in range(15000):
            if time.time() > deadline - 5:
                break
            
            # Pick a random edge and try a random different color
            edge_idx = rng.randint(0, num_edges - 1)
            old_color = current[edge_idx]
            
            # Try colors that are currently in use
            current_colors = set(current.tolist())
            other_colors = [c for c in current_colors if c != old_color]
            
            if not other_colors:
                continue
            
            new_color = rng.choice(other_colors)
            
            test_coloring = current.copy()
            test_coloring[edge_idx] = new_color
            
            if is_valid_fast(test_coloring):
                new_score = count_colors(test_coloring)
                if new_score < current_score:
                    current = test_coloring
                    current_score = new_score
                    if current_score < best_score:
                        best_coloring = current.copy()
                        best_score = current_score
        
        # Also try targeted elimination
        current = best_coloring.copy()
        current_score = best_score
        
        for _ in range(3000):
            if time.time() > deadline - 5:
                break
            
            color_counts = Counter(current.tolist())
            current_colors = sorted(color_counts.keys())
            
            if len(current_colors) <= 13:
                break
            
            # Try to eliminate each color
            eliminated = False
            for target_color in current_colors:
                if time.time() > deadline - 5:
                    break
                
                edges_with_color = np.where(current == target_color)[0]
                if len(edges_with_color) == 0:
                    continue
                
                # Try to reassign all edges of this color
                success = True
                for edge_idx in edges_with_color:
                    found = False
                    for new_color in current_colors:
                        if new_color == target_color:
                            continue
                        test_coloring = current.copy()
                        test_coloring[edge_idx] = new_color
                        if is_valid_fast(test_coloring):
                            current[edge_idx] = new_color
                            found = True
                            break
                    if not found:
                        success = False
                        break
                
                if success:
                    current_score = count_colors(current)
                    eliminated = True
                    if current_score < best_score:
                        best_coloring = current.copy()
                        best_score = current_score
                    break
            
            if not eliminated:
                break

    # Final validation
    if not is_valid_fast(best_coloring):
        best_coloring = seed_coloring()
        best_score = count_colors(best_coloring)

    return [int(c) for c in best_coloring]
# EVOLVE-BLOCK-END
