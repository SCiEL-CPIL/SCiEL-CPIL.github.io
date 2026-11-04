import time
import sys
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook fallback (Rank 48)
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build the rank-44 solution
    rank44 = 44
    U44 = [[0] * rank44 for _ in range(n * m)]
    V44 = [[0] * rank44 for _ in range(m * p)]
    W44 = [[0] * rank44 for _ in range(p * n)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U44[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V44[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W44[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    for i in range(2):
        for k in range(8):
            U44[i * m + 2][col] = 1
            V44[2 * p + k][col] = 1
            W44[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate44 = {"U": U44, "V": V44, "W": W44}
    if not _validate(candidate44):
        return _schoolbook()

    # Try to find rank 43 by eliminating one term
    # Strategy: try to express one of the 16 "third row" terms as a combination
    # of other terms, or find a linear dependency
    
    # The 16 terms for the third row of A are at columns 28..43
    # Each such term c has: U[i*3+2][c]=1, V[2*8+k][c]=1, W[k*2+i][c]=1
    # for i in {0,1}, k in {0..7}
    
    # Try: can we replace two of these 16 terms with one term plus adjustments?
    # Or can we find a rank-43 solution by perturbing?
    
    # Approach: Try to find a vector (u, v, w) in the span of existing columns
    # such that adding it creates a dependency that allows elimination
    
    # Simpler: try random rank-43 constructions by merging two terms
    # For each pair of columns (c1, c2) from the 16 schoolbook terms,
    # check if we can replace them with a single column
    
    # Actually, let's try a different construction:
    # Use 3 Strassen blocks (21 terms) for columns 0-5 of B
    # Then handle columns 6-7 differently
    
    # Or: try rank 42 with a different grouping
    
    # Let's try: 4 Strassen blocks but with one block using a rank-6 
    # decomposition of 2x2x2 instead of rank-7 (impossible, min is 7)
    
    # Alternative: overlap the Strassen blocks with the third-row terms
    # The third row of A (index 2) only appears in U at positions (0,2) and (1,2)
    # These don't appear in any Strassen block, so there's no overlap possible
    # in the current construction.
    
    # Let's try a completely different partition:
    # Treat A as 3 rows of 2 columns each, and B as 8 columns of 3 rows
    # Use a "strip" method
    
    # For now, return the rank-44 solution as best found
    return candidate44
