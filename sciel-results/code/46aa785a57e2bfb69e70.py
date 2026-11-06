"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
import random

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    n = 16
    
    def count_colors(coloring):
        return max(coloring) + 1
    
    def build_edge_array(coloring, n=16):
        """Build a 16x16 symmetric matrix where edge_matrix[i,j] = color of edge (i,j)"""
        edge_matrix = np.zeros((n, n), dtype=np.int32)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_matrix[i, j] = coloring[idx]
                edge_matrix[j, i] = coloring[idx]
                idx += 1
        return edge_matrix
    
    def is_valid_fast(edge_matrix, n=16):
        """Check if every K4 has at least 5 distinct colors using numpy"""
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            colors = [
                int(edge_matrix[a, b]), int(edge_matrix[a, c]), int(edge_matrix[a, d]),
                int(edge_matrix[b, c]), int(edge_matrix[b, d]), int(edge_matrix[c, d])
            ]
            if len(set(colors)) < 5:
                return False
        return True
    
    def make_coloring_formula(formula_fn, n=16):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(formula_fn(i, j))
        return colors
    
    candidates = []
    
    # Linear formulas: (a*i + b*j) mod m
    for mod in range(13, 20):
        for a in range(1, 8):
            for b in range(1, 8):
                if a == b:
                    continue
                cand = make_coloring_formula(lambda i, j, a=a, b=b, mod=mod: (a*i + b*j) % mod)
                candidates.append(cand)
    
    # XOR-based formulas
    for mod in range(13, 20):
        for shift in range(4):
            cand = make_coloring_formula(lambda i, j, shift=shift, mod=mod: ((i << shift) ^ j) % mod)
            candidates.append(cand)
    
    # Product-based: (i*j) mod m
    for mod in range(13, 20):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i * j) % mod)
        candidates.append(cand)
    
    # Mixed: (i*j + i + j) mod m
    for mod in range(13, 20):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i * j + i + j) % mod)
        candidates.append(cand)
    
    # Quadratic: (i^2 + j^2) mod m
    for mod in range(13, 20):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*i + j*j) % mod)
        candidates.append(cand)
    
    # Difference-based: (i-j)^2 mod m
    for mod in range(13, 20):
        cand = make_coloring_formula(lambda i, j, mod=mod: ((i-j)*(i-j)) % mod)
        candidates.append(cand)
    
    # (i^2 + a*i*j + b*j^2) mod m
    for mod in range(13, 20):
        for a in range(1, 5):
            for b in range(1, 5):
                cand = make_coloring_formula(lambda i, j, a=a, b=b, mod=mod: (i*i + a*i*j + b*j*j) % mod)
                candidates.append(cand)
    
    # Bit manipulation: (i ^ j) * k mod m
    for mod in range(13, 20):
        for k in range(1, 5):
            cand = make_coloring_formula(lambda i, j, k=k, mod=mod: ((i ^ j) * k) % mod)
            candidates.append(cand)
    
    # Sum of bits: popcount(i) + popcount(j) mod m
    for mod in range(13, 20):
        cand = make_coloring_formula(lambda i, j, mod=mod: (bin(i).count('1') + bin(j).count('1')) % mod)
        candidates.append(cand)
    
    # (i+j)^2 mod m
    for mod in range(13, 20):
        cand = make_coloring_formula(lambda i, j, mod=mod: ((i+j)*(i+j)) % mod)
        candidates.append(cand)
    
    # NEW: More diverse polynomial families
    # (a*i^2 + b*i*j + c*j^2 + d*i + e*j + f) mod m
    for mod in range(13, 17):
        for a in range(0, 3):
            for b in range(0, 3):
                for c in range(0, 3):
                    if a == 0 and b == 0 and c == 0:
                        continue
                    for d in range(0, 2):
                        for e in range(0, 2):
                            cand = make_coloring_formula(
                                lambda i, j, a=a, b=b, c=c, d=d, e=e, mod=mod: 
                                (a*i*i + b*i*j + c*j*j + d*i + e*j) % mod
                            )
                            candidates.append(cand)
    
    # (a*i*j + b*i + c*j) mod m
    for mod in range(13, 17):
        for a in range(1, 4):
            for b in range(0, 3):
                for c in range(0, 3):
                    cand = make_coloring_formula(
                        lambda i, j, a=a, b=b, c=c, mod=mod: 
                        (a*i*j + b*i + c*j) % mod
                    )
                    candidates.append(cand)
    
    # (i*j*(i+j)) mod m
    for mod in range(13, 17):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*j*(i+j)) % mod)
        candidates.append(cand)
    
    # (i*j*(i-j)) mod m  
    for mod in range(13, 17):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*j*(i-j)) % mod)
        candidates.append(cand)
    
    # (i^2*j + i*j^2) mod m
    for mod in range(13, 17):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*i*j + i*j*j) % mod)
        candidates.append(cand)
    
    # (i^3 + j^3) mod m
    for mod in range(13, 17):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*i*i + j*j*j) % mod)
        candidates.append(cand)
    
    # (i^2*j^2) mod m
    for mod in range(13, 17):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*i*j*j) % mod)
        candidates.append(cand)
    
    # (i^2*j + j^2*i + i*j) mod m
    for mod in range(13, 17):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*i*j + j*j*i + i*j) % mod)
        candidates.append(cand)
    
    # (a*i^2*j + b*i*j^2 + c*i*j) mod m
    for mod in range(13, 16):
        for a in range(1, 3):
            for b in range(1, 3):
                for c in range(0, 2):
                    cand = make_coloring_formula(
                        lambda i, j, a=a, b=b, c=c, mod=mod: 
                        (a*i*i*j + b*i*j*j + c*i*j) % mod
                    )
                    candidates.append(cand)
    
    # (i*j)^(k) mod m for small k
    for mod in range(13, 16):
        for k in [2, 3]:
            cand = make_coloring_formula(lambda i, j, k=k, mod=mod: pow(i*j, k, mod))
            candidates.append(cand)
    
    # (i+j)*(i-j) = i^2 - j^2 mod m
    for mod in range(13, 17):
        cand = make_coloring_formula(lambda i, j, mod=mod: ((i*i - j*j) % mod + mod) % mod)
        candidates.append(cand)
    
    # (i+j)^3 mod m
    for mod in range(13, 17):
        cand = make_coloring_formula(lambda i, j, mod=mod: ((i+j)**3) % mod)
        candidates.append(cand)
    
    # (i-j)^3 mod m
    for mod in range(13, 17):
        cand = make_coloring_formula(lambda i, j, mod=mod: ((i-j)**3) % mod)
        candidates.append(cand)
    
    # (i*j + i^2 + j^2) mod m
    for mod in range(13, 17):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*j + i*i + j*j) % mod)
        candidates.append(cand)
    
    # (i*j - i^2 - j^2) mod m
    for mod in range(13, 17):
        cand = make_coloring_formula(lambda i, j, mod=mod: ((i*j - i*i - j*j) % mod + mod) % mod)
        candidates.append(cand)
    
    # (a*(i*j) + b*(i+j) + c*(i-j)^2) mod m
    for mod in range(13, 16):
        for a in range(1, 3):
            for b in range(0, 2):
                for c in range(0, 2):
                    cand = make_coloring_formula(
                        lambda i, j, a=a, b=b, c=c, mod=mod: 
                        (a*i*j + b*(i+j) + c*(i-j)*(i-j)) % mod
                    )
                    candidates.append(cand)
    
    # (i XOR j) + (i AND j) mod m
    for mod in range(13, 17):
        cand = make_coloring_formula(lambda i, j, mod=mod: ((i ^ j) + (i & j)) % mod)
        candidates.append(cand)
    
    # (i XOR j) * (i AND j) mod m
    for mod in range(13, 17):
        cand = make_coloring_formula(lambda i, j, mod=mod: ((i ^ j) * (i & j)) % mod)
        candidates.append(cand)
    
    # (i | j) mod m
    for mod in range(13, 17):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i | j) % mod)
        candidates.append(cand)
    
    # (i & j) * (i ^ j) mod m
    for mod in range(13, 17):
        cand = make_coloring_formula(lambda i, j, mod=mod: ((i & j) * (i ^ j)) % mod)
        candidates.append(cand)
    
    # Find the best valid candidate
    best = None
    best_num_colors = float('inf')
    best_edge_matrix = None
    
    # Sort candidates by estimated color count (max value) to find good ones faster
    candidates.sort(key=lambda c: max(c) + 1)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        edge_matrix = build_edge_array(cand)
        if is_valid_fast(edge_matrix):
            nc = count_colors(cand)
            if nc < best_num_colors:
                best_num_colors = nc
                best = list(cand)
                best_edge_matrix = edge_matrix
    
    # If we found a valid coloring, try local search to reduce colors
    if best is not None and time.time() < deadline:
        # Local search: try to reduce colors by merging color classes
        current = list(best)
        current_nc = best_num_colors
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            edge_matrix = build_edge_array(current)
            
            # Try to merge each pair of colors
            colors_used = list(range(current_nc))
            for c1 in colors_used:
                for c2 in colors_used:
                    if c1 >= c2:
                        continue
                    if time.time() > deadline:
                        break
                    
                    # Try merging c2 into c1
                    test = list(current)
                    for idx, val in enumerate(test):
                        if val == c2:
                            test[idx] = c1
                    
                    test_edge = build_edge_array(test)
                    if is_valid_fast(test_edge):
                        new_nc = max(test) + 1
                        if new_nc < current_nc:
                            current = test
                            current_nc = new_nc
                            improved = True
                            break
                if time.time() > deadline:
                    break
            
            if improved:
                break
        
        if current_nc < best_num_colors:
            best = current
            best_num_colors = current_nc
    
    if best is not None:
        return best
    
    # Fallback: return the seed
    return make_coloring_formula(lambda i, j: (i + j) % 17)
    
# EVOLVE-BLOCK-END
