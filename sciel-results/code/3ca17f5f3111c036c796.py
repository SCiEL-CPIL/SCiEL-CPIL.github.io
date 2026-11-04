import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _verify(U, V, W, n=4, m=4, p=7, tol=1e-9):
    r = len(U[0]) if U and U[0] else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wi = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vk[c] * wi[c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n=4, m=4, p=7):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

# 2x2x2 Strassen: 7 rank-1 tensor terms for (2x2)(2x2)->(2x2)
# Each term: (X_dict, Y_dict) where X_dict maps (r,a)->coeff for A,
# Y_dict maps (a,c)->coeff for B. The output C = X*Y is handled via
# the standard Strassen reconstruction.
#
# The 7 Strassen products:
# P1 = (A00+A11)(B00+B11)
# P2 = (A10+A11)B00
# P3 = A00(B01-B11)
# P4 = A11(B10-B00)
# P5 = (A00+A01)B11
# P6 = (A10-A00)(B00+B01)
# P7 = (A01-A11)(B10+B11)
#
# C00 = P1 + P4 - P5 + P7
# C01 = P3 + P5
# C10 = P2 + P4
# C11 = P1 - P2 + P3 + P6

STRASSEN_2X2 = [
    # (X: {(r,a): coeff}, Y: {(a,c): coeff})
    # P1: X = A00+A11, Y = B00+B11
    ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    # P2: X = A10+A11, Y = B00
    ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
    # P3: X = A00, Y = B01-B11
    ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
    # P4: X = A11, Y = B10-B00
    ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
    # P5: X = A00+A01, Y = B11
    ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
    # P6: X = A10-A00, Y = B00+B01
    ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
    # P7: X = A01-A11, Y = B10+B11
    ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
]

# C reconstruction coefficients: C[r,c] = sum_t w[t][r,c] * P_t
C_COEFFS = [
    # P1
    {(0,0): 1, (1,1): 1},
    # P2
    {(1,0): 1, (1,1): -1},
    # P3
    {(0,1): 1, (1,1): 1},
    # P4
    {(0,0): 1, (1,0): 1},
    # P5
    {(0,0): -1, (0,1): 1},
    # P6
    {(1,1): 1},
    # P7
    {(0,0): 1},
]

def _verify_2x2_strassen():
    """Verify that the 7 Strassen terms correctly decompose 2x2x2 matmul."""
    n = m = p = 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for t in range(7):
        X, Y = STRASSEN_2X2[t]
        for (r, a), cx in X.items():
            U[r * m + a][t] += cx
        for (a, c), cy in Y.items():
            V[a * p + c][t] += cy
        for (r, c), wc in C_COEFFS[t].items():
            W[c * n + r][t] += wc
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_hybrid(n=4, m=4, p=7):
    """Build a hybrid decomposition:
    - 4x4x4 part (cols 0-3): global 2x2 block Strassen, rank 49
    - 4x4x3 part (cols 4-6): 2x2x3 Strassen per 2x2 block, rank 44
    Total: 93
    """
    # Verify 2x2x2 Strassen first
    if not _verify_2x2_strassen():
        return None
    
    total_rank = 49 + 44
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Part 1: 4x4x4 using global 2x2 block Strassen (rank 49)
    # 7 Strassen products, each decomposed into 7 sub-terms = 49 terms
    for t in range(7):
        X, Y = STRASSEN_2X2[t]
        for st in range(7):
            Xs, Ys = STRASSEN_2X2[st]
            r = current_rank
            current_rank += 1
            
            # Build U: for each (r,a) in Xs, for each (r,a) in X,
            # the block index determines the global position
            for (lr, la), cx in Xs.items():
                for (br, ba), cx2 in X.items():
                    # Global A position: row = 2*br + lr, col = 2*ba + la
                    gi = 2 * br + lr
                    gj = 2 * ba + la
                    U[gi * m + gj][r] += cx * cx2
            
            for (la, lc), cy in Ys.items():
                for (ba, bc), cy2 in Y.items():
                    # Global B position: row = 2*ba + la, col = 2*bc + lc
                    gj = 2 * ba + la
                    gk = 2 * bc + lc
                    V[gj * p + gk][r] += cy * cy2
            
            # W: C reconstruction
            for (lr, lc), wc in C_COEFFS[st].items():
                for (br, bc), wc2 in C_COEFFS[t].items():
                    gi = 2 * br + lr
                    gk = 2 * bc + lc
                    W[gk * n + gi][r] += wc * wc2
    
    # Part 2: 4x4x3 using 2x2x3 Strassen (rank 44)
    # For each pair of rows in A (2 pairs: rows 0-1, rows 2-3),
    # and for the 3 columns of B (cols 4-6), use a 2x2x3 decomposition.
    # 2x2x3 can be done with 2x2x2 Strassen (7 terms) + 1 schoolbook col (2x2x1 = 4 terms)
    # = 11 terms per 2x2x3. Two 2x2 row blocks: 2 * 11 = 22... 
    # Wait, that's only 22. Let me reconsider.
    #
    # Actually for 4x4x3: A is 4x4, B is 4x3, C is 4x3.
    # We can partition A into 2x2 blocks (4 blocks), B into 2x2+2x1 or similar.
    # Simpler: use 2x2x3 for each pair of A rows.
    # A rows 0-1 (2x4) times B (4x3) -> C rows 0-1 (2x3): this is 2x4x3
    # A rows 2-3 (2x4) times B (4x3) -> C rows 2-3 (2x3): this is 2x4x3
    #
    # 2x4x3: partition A into 2x2 blocks (2 blocks), B into 2x2 + 2x1
    # Block 1: A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6] (2x2x3)
    # Block 2: A[0:2,2:4] * B[2:4,4:6] -> C[0:2,4:6] (2x2x3)
    # Similarly for rows 2-3.
    #
    # 2x2x3 rank: use Strassen 2x2x2 (7) + schoolbook 2x2x1 (4) = 11
    # 4 blocks of 2x2x3: 4 * 11 = 44. 
    
    # 2x2x3 decomposition:
    # For 2x2x2 part (cols 0-1 of the 3): use 7 Strassen terms
    # For 2x2x1 part (col 2 of the 3): use 4 schoolbook terms
    # Total: 11 terms per 2x2x3
    
    for a_row_block in range(2):  # A rows 0-1 or 2-3
        for a_col_block in range(2):  # A cols 0-1 or 2-3
            # B rows match A cols: B rows 2*a_col_block to 2*a_col_block+1
            b_row_block = a_col_block
            # C rows: 2*a_row_block to 2*a_row_block+1
            c_row_block = a_row_block
            # C cols: 4,5,6
            
            # 2x2x2 Strassen for B cols 4,5 (first 2 of the 3)
            for t in range(7):
                X, Y = STRASSEN_2X2[t]
                r = current_rank
                current_rank += 1
                
                for (lr, la), cx in X.items():
                    gi = 2 * a_row_block + lr
                    gj = 2 * a_col_block + la
                    U[gi * m + gj][r] += cx
                
                for (la, lc), cy in Y.items():
                    gj = 2 * b_row_block + la
                    gk = 4 + lc  # B cols 4,5
                    V[gj * p + gk][r] += cy
                
                for (lr, lc), wc in C_COEFFS[t].items():
                    gi = 2 * c_row_block + lr
                    gk = 4 + lc  # C cols 4,5
                    W[gk * n + gi][r] += wc
            
            # 2x2x1 schoolbook for B col 6
            for i in range(2):
                for j in range(2):
                    r = current_rank
                    current_rank += 1
                    gi = 2 * a_row_block + i
                    gj = 2 * a_col_block + j
                    gj_b = 2 * b_row_block + j
                    gk = 6  # B col 6, C col 6
                    U[gi * m + gj][r] = 1.0
                    V[gj_b * p + gk][r] = 1.0
                    W[gk * n + gi][r] = 1.0
    
    return {"U": U, "V": V, "W": W, "rank": current_rank}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Try the Strassen hybrid decomposition
    try:
        result = _build_strassen_hybrid(n, m, p)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return {"U": result["U"], "V": result["V"], "W": result["W"]}
    except Exception:
        pass
    
    # Fallback to schoolbook
    result = _schoolbook(n, m, p)
    return result
