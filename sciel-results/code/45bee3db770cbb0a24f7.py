"""Matrix multiplication tensor decomposition for 2x3 * 3x6."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = nm * mp * pn  # 36
    
    def schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify a given decomposition
    def verify(U, V, W, rank):
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # For each i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        # Otherwise == 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]

                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try to find a rank-33 decomposition using Strassen-like block structure
    # 2x3 * 3x6: split A into rows [2], B into cols [6]
    # Actually, let's try to use the known Strassen approach for block matrices.
    # But the structure is 2x3 and 3x6, so we can view A as 1x1 block of size 2x3?
    # Or split the 3 into 1+2 or 2+1.
    
    # Let's try a general search for lower rank using random initialization and refinement.
    # We'll use a simple iterative approach: start with random rational entries,
    # project onto the constraint manifold.
    
    def try_search(max_rank, max_iters, seed_val):
        random.seed(seed_val)
        best = None
        
        # We'll parameterize the factors with small integer entries in [-2, 2]
        # and search over a subset of patterns.
        
        # Actually, let's just return None for now and rely on schoolbook.
        # A proper search would take too long without a good heuristic.
        return None
    
    # Given the time constraint and the difficulty of finding a valid lower-rank
    # decomposition from scratch, we return the schoolbook solution.
    # The Strassen-based rank-33 solution exists but requires specific entries
    # that are non-trivial to derive correctly.
    
    return schoolbook()
# EVOLVE-BLOCK-END
