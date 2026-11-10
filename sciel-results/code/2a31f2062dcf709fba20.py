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
    deadline = start_time + 145  # Safety margin before 150s limit

    n = 16
    num_edges = n * (n - 1) // 2

    # Precompute edge index mapping
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    # Precompute all K4s and their edge indices as numpy arrays
    k4_list = list(combinations(range(n), 4))
    num_k4s = len(k4_list)
    
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
        # coloring is a numpy array of length num_edges
        # Get colors for all K4 edges at once
        k4_colors = coloring[k4_edges_arr]  # shape (num_k4s, 6)
        # Count distinct colors per K4: for each row, check if < 5 unique
        # A K4 is valid iff not (all 6 same) and not (only 4 or fewer distinct)
        # Fast check: sort each row and count distinct
        # Alternative: for each K4, the number of distinct colors = 6 - (number of pairs that are equal)
        # But that's not quite right. Let's use a different approach.
        # A K4 has < 5 distinct colors iff it has a "repeated" pattern that limits uniqueness.
        # Specifically, < 5 distinct means at most 4 distinct, which means at least 2 pairs share a color.
        # Actually, let's just compute distinct count efficiently.
        # For 6 elements, distinct count = 6 - sum over pairs of [they are equal]... no that's wrong.
        # Let's use: distinct = number of unique values. For small arrays, we can check:
        # A K4 is INVALID if the 6 colors have < 5 distinct values.
        # This happens iff there exists a value appearing at least twice AND the total distinct < 5.
        # Simplest vectorized approach: for each K4, check if all 6 are same (1 distinct),
        # or if there's a pattern with 4 or fewer distinct.
        # Let me use a different trick: a K4 is valid iff for every pair of "opposite" edges 
        # in the K4... no, that's not right either.
        # Let's just do: for each K4, compute the set size by checking all pairs.
        # Actually, the fastest way for 6 elements: 
        # distinct_count >= 5 iff NOT (there are at least 2 pairs of equal elements that "cover" enough)
        # Let me just use: invalid iff len(set) < 5.
        # Vectorized: for each row, check if any value appears >= 2 times AND the number of distinct values < 5.
        # A simpler sufficient condition for invalidity: 
        #   - all 6 same: 1 distinct
        #   - 5 same, 1 different: 2 distinct  
        #   - 4 same, 2 same: 2 distinct
        #   - 4 same, 1, 1: 3 distinct
        #   - 3 same, 3 same: 2 distinct
        #   - 3 same, 2 same, 1: 3 distinct
        #   - 3 same, 1, 1, 1: 4 distinct
        #   - 2 same, 2 same, 2 same: 3 distinct
        #   - 2 same, 2 same, 1, 1: 4 distinct
        #   - 2 same, 1, 1, 1, 1: 5 distinct (VALID)
        #   - all distinct: 6 distinct (VALID)
        # So INVALID iff: some value appears >= 3 times, OR (two values each appear >= 2 times and total distinct <= 4)
        # This is getting complex. Let me use a simpler approach:
        # For each K4, the coloring is invalid iff the 6 edge colors span < 5 colors.
        # I'll compute this by: for each K4, check if there's a color that appears on all 6 edges (1 distinct),
        # or check pairwise. Actually, let me just use a lookup approach.
        
        # Most practical vectorized approach: 
        # A K4 is invalid iff min(color) == max(color) (1 distinct) 
        # or (max-min is small and ...) - this doesn't work for arbitrary color labels.
        
        # Let me use: for each K4, count distinct by checking all 15 pairs.
        # Two colors are "equal" if they match. The number of distinct values in 6 elements
        # can be computed as: 6 - (number of equivalence classes - 1) ... no.
        # 
        # Actually, the cleanest vectorized approach: 
        # For each K4, invalid iff there exist colors c1, c2 such that... 
        # Let me just compute: for each K4, is the number of distinct colors < 5?
        # I'll use: distinct < 5 iff there's a "collision" pattern.
        # 
        # Practical approach: use the fact that with 6 elements, distinct < 5 means 
        # at least one value appears at least twice, AND the structure limits uniqueness.
        # 
        # Let me just do it the straightforward way but vectorized:
        # For each K4, check if all 6 are equal, or if there's a specific pattern.
        
        # Actually, let me use a different strategy: precompute which K4s are "tight" 
        # (exactly 5 colors) vs "loose" (6 colors). A K4 with exactly 5 colors has one 
        # pair of edges sharing a color. If we change that shared color, the K4 might become valid or invalid.
        
        # For now, let me use a simple but effective vectorized check:
        # A K4 is valid iff NOT (the 6 colors have a "deficit" of uniqueness)
        # I'll compute: for each K4, the number of distinct colors.
        # Use: sort each row (6 elements), then count transitions.
        
        sorted_k4 = np.sort(k4_colors, axis=1)  # (num_k4s, 6)
        # Count distinct: 1 + number of positions where sorted_k4[i, k] != sorted_k4[i, k-1]
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

    # Try alternative structured colorings
    structured_starts = []
    # (i+j) % 17 is our base
    # Try (i*j) % k for various k
    for k in [13, 14, 15, 16]:
        colors = np.array([(i * j) % k for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        if is_valid_fast(colors):
            sc = count_colors(colors)
            structured_starts.append((sc, colors))
    
    # Try (i+j+offset) % k
    for k in [13, 14, 15, 16]:
        for offset in range(k):
            if time.time() > deadline - 50:
                break
            colors = np.array([((i + j) + offset) % k for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
            if is_valid_fast(colors):
                sc = count_colors(colors)
                structured_starts.append((sc, colors))
    
    # Try (j-i) % k
    for k in [13, 14, 15, 16]:
        for offset in range(k):
            if time.time() > deadline - 50:
                break
            colors = np.array([((j - i) + offset) % k for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
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
    def try_reduce_colors(coloring, max_iters=5000):
        """Aggressively try to reduce the number of colors used."""
        coloring = coloring.copy()
        current_score = count_colors(coloring)
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Get current color usage
            color_counts = Counter(coloring.tolist())
            current_colors = sorted(color_counts.keys())
            
            if len(current_colors) <= 13:
                break
            
            # Try to eliminate the rarest color
            # Find the least frequent color
            rarest_color = min(current_colors, key=lambda c: color_counts[c])
            edges_with_rarest = np.where(coloring == rarest_color)[0]
            
            if len(edges_with_rarest) == 0:
                continue
            
            # Try to reassign all edges of rarest_color to other colors
            # Strategy: for each edge, try to assign it to the "best" alternative color
            # that maintains validity
            
            success = True
            for edge_idx in edges_with_rarest:
                # Try assigning to each other color
                found = False
                for new_color in current_colors:
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
                # We eliminated rarest_color
                current_score = count_colors(coloring)
            else:
                # Try a more aggressive approach: randomly reassign some edges
                # and see if we can still reduce
                break
        
        return coloring, current_score

    # Try reducing from our best starting point
    current = best_coloring.copy()
    reduced, score = try_reduce_colors(current, max_iters=2000)
    if score < best_score:
        best_coloring = reduced
        best_score = score

    # Local search: try to further reduce by swapping colors
    if time.time() < deadline - 20 and best_score > 13:
        rng = random.Random(42)
        current = best_coloring.copy()
        current_score = best_score
        
        for _ in range(10000):
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
        
        for _ in range(2000):
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
