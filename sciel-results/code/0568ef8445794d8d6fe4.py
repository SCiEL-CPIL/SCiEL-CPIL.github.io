import itertools
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    n = 16
    
    # Precompute all 4-combinations
    quads = list(itertools.combinations(range(n), 4))
    
    # Precompute edge indices for each quad
    # Edge (i, j) with i < j.
    # We will use a 16x16 matrix for lookup.
    
    def count_colors(coloring):
        return max(coloring) + 1 if coloring else 0

    def build_edge_matrix(coloring):
        """
        coloring: list of 120 ints
        returns: 16x16 numpy array
        """
        M = np.zeros((n, n), dtype=np.int32)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = coloring[idx]
                M[i, j] = c
                M[j, i] = c
                idx += 1
        return M

    def is_valid_fast(M):
        """
        Check if every K4 in M has at least 5 distinct colors.
        M is a 16x16 numpy array.
        """
        for a, b, c, d in quads:
            # 6 edges: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
            # Using a set of integers
            # numpy indexing returns scalars, convert to int for set
            c1 = int(M[a, b])
            c2 = int(M[a, c])
            c3 = int(M[a, d])
            c4 = int(M[b, c])
            c5 = int(M[b, d])
            c6 = int(M[c, d])
            # Check distinct count
            # Manual check is faster than set() for small numbers sometimes, 
            # but set() is fine.
            if len({c1, c2, c3, c4, c5, c6}) < 5:
                return False
        return True

    def make_coloring(formula):
        """
        formula: function(i, j) -> int
        returns: list of 120 ints
        """
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(formula(i, j))
        return colors

    candidates = []
    
    # 1. Modular Linear and Polynomial Families
    # Try moduli 13 to 16
    for m in range(13, 17):
        # Linear: (a*i + b*j) % m
        for a in range(1, m):
            for b in range(1, m):
                if a == b:
                    # (i+i) = 2i, depends only on i, might be bad
                    # But let's include some
                    pass 
                candidates.append(lambda i, j, a=a, b=b, m=m: (a*i + b*j) % m)
        
        # Difference: (i-j)^2 % m (symmetric)
        candidates.append(lambda i, j, m=m: ((i-j)*(i-j)) % m)
        
        # Product: (i*j) % m
        candidates.append(lambda i, j, m=m: (i*j) % m)
        
        # Sum of squares: (i^2 + j^2) % m
        candidates.append(lambda i, j, m=m: (i*i + j*j) % m)
        
        # Mixed: (i^2 + i*j + j^2) % m
        candidates.append(lambda i, j, m=m: (i*i + i*j + j*j) % m)
        
        # (i*j + i + j) % m
        candidates.append(lambda i, j, m=m: (i*j + i + j) % m)
        
        # (i*j - i - j) % m
        candidates.append(lambda i, j, m=m: ((i*j - i - j) % m + m) % m)
        
        # (i+j)^2 % m
        candidates.append(lambda i, j, m=m: ((i+j)*(i+j)) % m)
        
        # (i-j)^3 % m (handle negative)
        candidates.append(lambda i, j, m=m: ((i-j)**3 % m + m) % m)
        
        # (i*j*(i+j)) % m
        candidates.append(lambda i, j, m=m: (i*j*(i+j)) % m)
        
        # Bitwise XOR % m
        candidates.append(lambda i, j, m=m: (i ^ j) % m)
        
        # Bitwise AND % m
        candidates.append(lambda i, j, m=m: (i & j) % m)
        
        # (i | j) % m
        candidates.append(lambda i, j, m=m: (i | j) % m)
        
        # Randomized linear combinations with small coefficients
        for _ in range(20):
            a = random.randint(-5, 5)
            b = random.randint(-5, 5)
            c = random.randint(-5, 5)
            if a == 0 and b == 0 and c == 0:
                continue
            candidates.append(lambda i, j, a=a, b=b, c=c, m=m: (a*i*i + b*i*j + c*j*j) % m)
            
    # 2. Random Permutations of Vertices applied to Seed-like structures
    # The seed (i+j) % 17 works. What if we permute vertices?
    # This preserves the property.
    # But we want to reduce colors.
    # Let's stick to direct formulas first.
    
    # 3. Specific known good structures?
    # If m=13, 14, 15, 16 are valid, we found a better solution.
    
    # Evaluate candidates
    best_coloring = None
    best_num_colors = float('inf')
    
    # Sort candidates by estimated complexity or just iterate
    # To speed up, we can check a subset first? No, 1820 checks is fast.
    # Number of candidates: 
    # m in 13..16 (4 mods)
    # Linear: ~13*13*4 = ~600
    # Others: ~10*4 = 40
    # Random: 20*4 = 80
    # Total ~700 candidates.
    # 700 * 1820 * (6 accesses + set) is very fast.
    
    for idx, formula in enumerate(candidates):
        if time.time() > deadline:
            break
            
        try:
            colors = make_coloring(formula)
            # Quick check: if max color >= best_num_colors, skip?
            # Actually, we want to MINIMIZE colors.
            # If max(color) + 1 >= best_num_colors, it can't be strictly better, 
            # but it might be equal. We only care about improvement.
            # However, a coloring with 13 colors is better than 14.
            # If we find 13, we stop? No, we want the min.
            # If max(c)+1 >= best_num_colors, we can skip validation if we only care about < best.
            # But we need to know if it's valid to update best.
            # If max(c)+1 >= best_num_colors, it's not an improvement.
            # So we can skip validation if max(c)+1 >= best_num_colors.
            
            max_c = max(colors)
            if max_c + 1 >= best_num_colors:
                continue
                
            M = build_edge_matrix(colors)
            if is_valid_fast(M):
                best_num_colors = max_c + 1
                best_coloring = colors
                # If we hit the theoretical lower bound 13, we can stop?
                # The bound is 13 <= f(16). So 13 is optimal.
                if best_num_colors == 13:
                    break
        except Exception:
            continue
    
    # If no candidate from the list worked, or we want to try to improve a valid one
    # Let's try a local search on the best found, or on the seed if none found.
    
    current_coloring = best_coloring
    if current_coloring is None:
        # Fallback to seed
        current_coloring = make_coloring(lambda i, j: (i + j) % 17)
        best_num_colors = 17
    else:
        best_num_colors = max(current_coloring) + 1
    
    # Local Search: Merge colors
    # We have a valid coloring. Can we merge two colors to reduce the count?
    # This is a hard problem, but greedy merging might work.
    
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            M = build_edge_matrix(current_coloring)
            colors_used = list(set(current_coloring))
            
            # Try to merge each pair
            for c1 in colors_used:
                for c2 in colors_used:
                    if c1 >= c2:
                        continue
                    if time.time() > deadline - 2:
                        break
                    
                    # Create a test coloring where c2 -> c1
                    test_colors = []
                    for c in current_coloring:
                        if c == c2:
                            test_colors.append(c1)
                        else:
                            test_colors.append(c)
                    
                    # Check if valid
                    # Optimization: Only check K4s that contain an edge of color c2
                    # But building the full matrix and checking all is simpler and fast enough
                    # 1820 checks is < 1ms usually.
                    test_M = build_edge_matrix(test_colors)
                    if is_valid_fast(test_M):
                        # Success
                        current_coloring = test_colors
                        best_num_colors = max(current_coloring) + 1
                        improved = True
                        break
                if time.time() > deadline - 2:
                    break
    
    return current_coloring
