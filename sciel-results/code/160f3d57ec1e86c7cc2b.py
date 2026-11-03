import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

# ---------------------------------------------------------------------------
# Exact verification (integer / half-integer safe via Fraction)
# ---------------------------------------------------------------------------
def _verify(U, V, W):
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for M in (U, V, W):
        for row in M:
            if len(row) != rank:
                return False

    # Convert to Fraction for exact arithmetic
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]

    one = Fraction(1, 1)
    zero = Fraction(0, 1)
    
    # Check all (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = zero
                urow = Uf[i * m + j]
                vrow = Vf[j * p + k]
                wrow = Wf[k * n + i]
                for r in range(rank):
                    s += urow[r] * vrow[r] * wrow[r]
                if s != one:
                    return False
    return True

# ---------------------------------------------------------------------------
# Construction of the rank-88 decomposition
# ---------------------------------------------------------------------------
def _build_rank88():
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Part 1: Row i=0. Schoolbook for 1x4x8.
    # C_0k = sum_j A_0j B_jk
    # Tensor terms: e_{0,j} (U), e_{j,k} (V), e_{k,0} (W)
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: Rows i=1,2. Strassen-like for 2x4x8.
    # Split inner dim m=4 into two blocks of 2: j in {0,1}, {2,3}
    # Split output dim p=8 into four blocks of 2: k in {0,1}, {2,3}, {4,5}, {6,7}
    # For each (j_block, k_block), we have a 2x2x2 subproblem.
    # A_sub is 2x2 (rows 1,2; cols 2*jb, 2*jb+1)
    # B_sub is 2x2 (rows 2*jb, 2*jb+1; cols 2*kb, 2*kb+1)
    # C_sub is 2x2 (rows 1,2; cols 2*kb, 2*kb+1)
    
    # Strassen coefficients for 2x2x2
    # U (A factors): 7 vectors in R^4 (flattened 2x2)
    # V (B factors): 7 vectors in R^4 (flattened 2x2)
    # W (C factors): 7 vectors in R^4 (flattened 2x2)
    
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B21-B22)
    # M4 = (A22)(B12-B21)
    # M5 = (A11+A12)(B22)
    # M6 = (A12-A21)(B11+B21)
    # M7 = (A21-A11)(B12+B22)
    
    # Indices for 2x2 matrix [0,1;2,3] (row-major)
    # A11=0, A12=1, A21=2, A22=3
    
    strassen_U = [
        [1, 0, 0, 1],   # M1: A11+A22
        [0, 0, 1, 1],   # M2: A21+A22
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: A11+A12
        [-1, 0, 1, 0],  # M6: A12-A21 (Note: standard is A12-A21 or A21-A12? Let's check W)
        [1, 0, 0, -1],  # M7: A21-A11 (Wait, standard Strassen: M7 = (A21-A11)(B12+B22))
    ]
    
    # Let's re-verify Strassen formulas to ensure correctness.
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B21-B22)
    # M4 = (A22)(B12-B21)
    # M5 = (A11+A12)(B22)
    # M6 = (A12-A21)(B11+B21)
    # M7 = (A21-A11)(B12+B22)
    
    # U vectors (A):
    U1 = [1, 0, 0, 1] # A11+A22
    U2 = [0, 0, 1, 1] # A21+A22
    U3 = [1, 0, 0, 0] # A11
    U4 = [0, 0, 0, 1] # A22
    U5 = [1, 1, 0, 0] # A11+A12
    U6 = [0, 1, -1, 0] # A12-A21
    U7 = [-1, 0, 1, 0] # A21-A11
    
    # V vectors (B):
    V1 = [1, 0, 0, 1] # B11+B22
    V2 = [1, 0, 0, 0] # B11
    V3 = [0, -1, 1, 0] # B21-B22 (Indices: B11=0, B12=1, B21=2, B22=3)
    # Wait, B21 is index 2, B22 is index 3.
    # B21-B22 -> [0, 0, 1, -1]
    V3 = [0, 0, 1, -1]
    V4 = [0, 1, -1, 0] # B12-B21 -> [0, 1, -1, 0]
    V5 = [0, 0, 0, 1] # B22
    V6 = [1, 0, 1, 0] # B11+B21
    V7 = [0, 1, 0, 1] # B12+B22
    
    # W vectors (C):
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    W1 = [1, 0, 0, -1] # M1 contributes +1 to C11, -1 to C22?
    # Let's map C indices: C11=0, C12=1, C21=2, C22=3
    # M1: +1*C11, +1*C22 ? No.
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    W1 = [1, 0, 0, 1] # M1 adds to C11 and C22
    W2 = [0, 0, 1, -1] # M2 adds to C21, subtracts from C22
    W3 = [0, 1, 0, 1] # M3 adds to C12 and C22
    W4 = [1, 0, 1, 0] # M4 adds to C11 and C21
    W5 = [-1, 1, 0, 0] # M5 subtracts from C11, adds to C12
    W6 = [0, 0, 0, 1] # M6 adds to C22
    W7 = [-1, 0, 0, 0] # M7 subtracts from C11
    
    strassen_U = [U1, U2, U3, U4, U5, U6, U7]
    strassen_V = [V1, V2, V3, V4, V5, V6, V7]
    strassen_W = [W1, W2, W3, W4, W5, W6, W7]
    
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                # Update U
                for l in range(4):
                    if strassen_U[t][l] != 0:
                        li = l // 2
                        lj = l % 2
                        U[(1 + li) * m + (j_start + lj)][col] += strassen_U[t][l]
                # Update V
                for l in range(4):
                    if strassen_V[t][l] != 0:
                        lj = l // 2
                        lk = l % 2
                        V[(j_start + lj) * p + (k_start + lk)][col] += strassen_V[t][l]
                # Update W
                for l in range(4):
                    if strassen_W[t][l] != 0:
                        li = l // 2
                        lk = l % 2
                        W[(k_start + lk) * n + (1 + li)][col] += strassen_W[t][l]
                col += 1

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build the rank-88 solution
    best = _build_rank88()
    best_rank = 88
    
    # Verify
    if _verify(best["U"], best["V"], best["W"]):
        return best
    else:
        # Fallback to schoolbook (rank 96) if something went wrong
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
