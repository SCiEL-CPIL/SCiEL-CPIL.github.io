"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n = 16
    
    quads = list(itertools.combinations(range(n), 4))
    
    def build_ec(coloring):
        """Build edge color lookup as a 2D array."""
        ec = np.zeros((n, n), dtype=np.int32)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                ec[i, j] = coloring[idx]
                ec[j, i] = coloring[idx]
                idx += 1
        return ec
    
    def is_valid_fast(coloring):
        """Fast validation using a 2D array."""
        ec = build_ec(coloring)
        for a, b, c, d in quads:
            c1 = ec[a, b]
            c2 = ec[a, c]
            c3 = ec[a, d]
            c4 = ec[b, c]
            c5 = ec[b, d]
            c6 = ec[c, d]
            colors = {c1, c2, c3, c4, c5, c6}
            if len(colors) < 5:
                return False
        return True
    
    def make_coloring_formula(formula_fn):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(formula_fn(i, j))
        return colors
    
    candidates = []
    
    # 1. Try linear forms (a*i + b*j) % m
    for m in range(13, 18):
        for a in range(1, 8):
            for b in range(1, 8):
                if a == b:
                    continue
                cand = make_coloring_formula(lambda i, j, a=a, b=b, m=m: (a * i + b * j) % m)
                candidates.append(cand)
                
    # 2. Try (i + j) % m specifically
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i + j) % m)
        candidates.append(cand)
        
    # 3. Try (i*j) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i * j) % m)
        candidates.append(cand)
        
    # 4. Try (i*i + j*j) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i*i + j*j) % m)
        candidates.append(cand)
    
    # 5. Try XOR-based: (i ^ j) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i ^ j) % m)
        candidates.append(cand)
    
    # 6. Try (i + 2*j) % m and (2*i + j) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i + 2*j) % m)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, m=m: (2*i + j) % m)
        candidates.append(cand)
    
    # 7. Try (i*j + i + j) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i*j + i + j) % m)
        candidates.append(cand)
    
    # 8. Try (i*j + i - j) % m (need to handle negative, use (i*j + i + (m-j)) % m)
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i*j + i + (m-j)) % m)
        candidates.append(cand)
    
    # 9. Try (i^2 + j^2 + i*j) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i*i + j*j + i*j) % m)
        candidates.append(cand)
    
    # 10. Try (i^2 - j^2) % m = (i-j)(i+j) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: ((i-j)*(i+j)) % m)
        candidates.append(cand)
    
    # 11. Try (i*j + i^2 + j^2) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i*j + i*i + j*j) % m)
        candidates.append(cand)
    
    # 12. Try (i + j + i*j) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i + j + i*j) % m)
        candidates.append(cand)
    
    # 13. Try (i^3 + j^3) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i*i*i + j*j*j) % m)
        candidates.append(cand)
    
    # 14. Try (i^2 * j + i * j^2) % m = i*j*(i+j) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i*j*(i+j)) % m)
        candidates.append(cand)
    
    # 15. Try (i^2 + i*j + j^2) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i*i + i*j + j*j) % m)
        candidates.append(cand)
    
    # 16. Try (i + 3*j) % m and (3*i + j) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i + 3*j) % m)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, m=m: (3*i + j) % m)
        candidates.append(cand)
    
    # 17. Try (i + 4*j) % m and (4*i + j) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i + 4*j) % m)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, m=m: (4*i + j) % m)
        candidates.append(cand)
    
    # 18. Try (i + 5*j) % m and (5*i + j) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i + 5*j) % m)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, m=m: (5*i + j) % m)
        candidates.append(cand)
    
    # 19. Try (i + 6*j) % m and (6*i + j) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i + 6*j) % m)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, m=m: (6*i + j) % m)
        candidates.append(cand)
    
    # 20. Try (i + 7*j) % m and (7*i + j) % m
    for m in range(13, 18):
        cand = make_coloring_formula(lambda i, j, m=m: (i + 7*j) % m)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, m=m: (7*i + j) % m)
        candidates.append(cand)
    
    # Sort candidates by number of colors used (ascending) to find minimum faster
    candidates.sort(key=lambda c: max(c) + 1)
    
    best = None
    best_num_colors = float('inf')
    
    for cand in candidates:
        if time.time() > deadline:
            break
        nc = max(cand) + 1
        if nc >= best_num_colors:
            break  # No need to check candidates with more colors
        if is_valid_fast(cand):
            best_num_colors = nc
            best = cand
    
    if best is not None:
        return best
    
    # Fallback: return a known valid coloring with 17 colors
    return make_coloring_formula(lambda i, j: (i + j) % 17)
    
# EVOLVE-BLOCK-END
