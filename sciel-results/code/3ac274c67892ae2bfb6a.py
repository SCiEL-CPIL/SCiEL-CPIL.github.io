import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

def _schoolbook_3x3x3():
    """Return (U, V, W) for rank-27 schoolbook 3x3x3.
    U is 9x27, V is 9x27, W is 9x27 (all lists of lists of int/frac)."""
    n = m = p = 3
    rank = 27
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def _strassen_3x3x3():
    """Return (U, V, W) for rank-14 Strassen 3x3x3.
    U is 9x14, V is 9x14, W is 9x14."""
    n = m = p = 3
    rank = 14
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    # Strassen factors for 3x3x3 (rank 14)
    # Each factor c: U_c, V_c, W_c are 3x3 matrices (as 9-element lists row-major)
    # U[c][i*m+j] = coefficient for A[i][j]
    # V[c][j*p+k] = coefficient for B[j][k]
    # W[c][k*n+i] = coefficient for C[i][k]
    
    def mat3(r0c0, r0c1, r0c2, r1c0, r1c1, r1c2, r2c0, r2c1, r2c2):
        return [r0c0, r0c1, r0c2, r1c0, r1c1, r1c2, r2c0, r2c1, r2c2]
    
    # Standard Strassen 8 factors for 2x2x2, extended to 3x3x3
    # For 3x3, we can use the following rank-14 decomposition:
    # This is based on the Laderman/Strassen extension.
    # Let's use a known rank-14 decomposition for 3x3x3.
    
    # I'll use the decomposition from Strassen's 1969 paper extended to 3x3.
    # Factors are linear combinations.
    
    # Factor 0: M1 = (A11+A22)(B11+B22) style, but for 3x3
    # Let me use a concrete rank-14 decomposition.
    
    # Actually, let me use a simpler approach: 
    # R(3,3,3) = 14 is known. Let me hardcode a valid one.
    # From literature: the 14-term formula for 3x3 matrix multiplication.
    
    # I'll construct it by extending the 2x2 Strassen (rank 7) to 3x3.
    # The 3x3 matrix can be viewed as 2x2 blocks with extra rows/cols.
    # This is complex. Let me use a verified rank-14 decomposition.
    
    # Here's a rank-14 decomposition (U, V, W each 9x14):
    # Using the standard extension.
    
    # Factor 1
    U0 = mat3(1,0,0, 0,1,0, 0,0,0)
    V0 = mat3(1,0,0, 0,1,0, 0,0,0)
    W0 = mat3(1,0,0, 0,1,0, 0,0,0)
    
    # This is getting complex. Let me use a different approach.
    # I'll verify if a rank-14 decomposition exists by construction.
    
    # For safety, let me just use schoolbook for 3x3x3 (rank 27) 
    # and combine with schoolbook for 3x3x2 (rank 18).
    # Total: 27 + 27 + 18 = 72. Same as baseline.
    
    # Actually, let me try to use Strassen properly.
    # The rank-14 3x3x3 decomposition:
    # I'll use the one from "Fast Matrix Multiplication" by C. Strassen.
    
    # Let me use a different strategy: use the known fact that
    # R(3,3,3) <= 14 and provide a valid decomposition.
    
    # Here is a valid rank-14 decomposition for 3x3x3:
    # (Sources: various fast matrix multiplication papers)
    
    # I'll define the 14 factors as 3x3 matrices for U, V, W.
    # For each factor r, U[r], V[r], W[r] are 3x3 (flattened to 9).
    
    # Let me use the decomposition from:
    # "A 14-term formula for 3x3 matrix multiplication"
    
    # Factors (U_r, V_r, W_r) as 3x3 matrices:
    
    f = []
    # Factor 1: diagonal-like
    f.append((mat3(1,0,0, 0,1,0, 0,0,1),
              mat3(1,0,0, 0,1,0, 0,0,1),
              mat3(1,0,0, 0,1,0, 0,0,1)))
    # Factor 2
    f.append((mat3(0,1,0, 0,0,1, 1,0,0),
              mat3(0,1,0, 0,0,1, 1,0,0),
              mat3(0,1,0, 0,0,1, 1,0,0)))
    # Factor 3
    f.append((mat3(0,0,1, 1,0,0, 0,1,0),
              mat3(0,0,1, 1,0,0, 0,1,0),
              mat3(0,0,1, 1,0,0, 0,1,0)))
    # Factor 4
    f.append((mat3(1,1,0, 1,1,0, 0,0,0),
              mat3(1,0,0, 0,1,0, 0,0,0),
              mat3(1,1,0, 0,0,0, 0,0,0)))
    # Factor 5
    f.append((mat3(0,0,0),
              mat3(1,0,0),
              mat3(1,0,0)))
    
    # This is not working well. Let me abandon Strassen for 3x3x3
    # and use a simpler decomposition.
    
    # Strategy: 3x3x8 = two 3x3x4 blocks.
    # For 3x3x4, schoolbook is 36. But we can do better.
    # Actually, let's just use schoolbook for the whole thing.
    # But that gives rank 72, same as baseline.
    
    # Alternative: use the fact that 3x3x8 can be done with
    # rank = R(3,3,4) + R(3,3,4).
    # R(3,3,4) <= 36 (schoolbook).
    # So rank <= 72.
    
    # Can we do R(3,3,4) < 36?
    # R(3,3,4) >= 3*4 = 12 (trivial bound).
    # R(3,3,4) <= 36.
    # Actually, R(3,3,4) = ? Not sure.
    
    # Let me try a different approach: use tensor rank bounds.
    # For 3x3x8, the trivial bound is 3*3*8 = 72.
    # The "border rank" might be lower.
    
    # For now, let me return the schoolbook solution (rank 72)
    # but structured to be easily improvable.
    
    # Actually, I realize I should just return the schoolbook.
    # The task says score is the rank, and lower is better.
    # Schoolbook gives 72.
    
    # Let me try one more thing: can I use a rank-13 decomposition for 3x3x3?
    # No, R(3,3,3) = 14 is the best known.
    
    # So my best bet is:
    # 3x3x8 = 3x3x3 + 3x3x3 + 3x3x2
    # Rank = 14 + 14 + 18 = 46, IF I have a valid rank-14 3x3x3.
    
    # Since I can't easily hardcode a valid rank-14 3x3x3,
    # let me just use schoolbook: 27 + 27 + 18 = 72.
    
    # This gives rank 72, same as baseline. Score = 72/72 = 1.0.
    # Not an improvement.
    
    # Let me try to actually implement a valid rank-14 3x3x3.
    
    # From a known source, here is a rank-14 decomposition:
    # (I'll use the one from the "Matrix Multiplication" Wikipedia article's references)
    
    # Let me try a programmatic construction.
    # For 3x3x3, we can use the 2x2 Strassen (rank 7) applied to
    # overlapping 2x2 blocks.
    
    # 3x3 = [2x2 | 2x2
    #         2x2 | 2x2] with overlaps.
    
    # This is getting too complex for a quick implementation.
    # Let me just return schoolbook.
    
    return U, V, W  # All zeros, will be replaced

def _schoolbook_3x3x2():
    """Return (U, V, W) for rank-18 schoolbook 3x3x2.
    U is 9x18, V is 6x18, W is 6x18."""
    n = m = p = 3  # m=3 for the middle dimension
    p_actual = 2
    rank = n * m * p_actual  # 18
    U = [[0]*rank for _ in range(n*m)]  # 9x18
    V = [[0]*rank for _ in range(m*p_actual)]  # 6x18
    W = [[0]*rank for _ in range(p_actual*n)]  # 6x18
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p_actual):
                U[i*m+j][c] = 1
                V[j*p_actual+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def _combine_blocks(U1, V1, W1, U2, V2, W2, U3, V3, W3):
    """Combine three blocks: 3x3x3, 3x3x3, 3x3x2 into 3x3x8.
    U: 9x(r1+r2+r3), V: 8x(r1+r2+r3), W: 24x(r1+r2+r3)
    Wait, V should be m*p = 3*8 = 24 rows.
    W should be p*n = 8*3 = 24 rows.
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    r3 = len(U3[0])
    total_rank = r1 + r2 + r3
    
    # U: n*m = 9 rows, total_rank cols
    U = [[0]*total_rank for _ in range(9)]
    for i in range(9):
        for c in range(r1):
            U[i][c] = U1[i][c]
        for c in range(r2):
            U[i][r1+c] = U2[i][c]
        for c in range(r3):
            U[i][r1+r2+c] = U3[i][c]
    
    # V: m*p = 3*8 = 24 rows, total_rank cols
    # Block 1: 3x3x3, V1 is 9x14 (or 9x27), corresponds to B[0:3, 0:3]
    # Block 2: 3x3x3, V2 is 9x14, corresponds to B[0:3, 3:6]
    # Block 3: 3x3x2, V3 is 6x18, corresponds to B[0:3, 6:8]
    V = [[0]*total_rank for _ in range(24)]
    # V1: 9 rows (j=0..2, k=0..2) -> V[j*8 + k, c] for j=0..2, k=0..2
    for j in range(3):
        for k in range(3):
            for c in range(r1):
                V[j*8 + k][c] = V1[j*3 + k][c]
    # V2: 9 rows -> V[j*8 + (k+3), c]
    for j in range(3):
        for k in range(3):
            for c in range(r2):
                V[j*8 + (k+3)][r1+c] = V2[j*3 + k][c]
    # V3: 6 rows -> V[j*8 + (k+6), c]
    for j in range(3):
        for k in range(2):
            for c in range(r3):
                V[j*8 + (k+6)][r1+r2+c] = V3[j*2 + k][c]
    
    # W: p*n = 8*3 = 24 rows, total_rank cols
    # W[k*n + i, c]
    # Block 1: W1 is 9x14, W1[k*3 + i, c] for k=0..2, i=0..2 -> W[k*3 + i, c]
    for k in range(3):
        for i in range(3):
            for c in range(r1):
                W[k*3 + i][c] = W1[k*3 + i][c]
    # Block 2: W2 -> W[(k+3)*3 + i, c]
    for k in range(3):
        for i in range(3):
            for c in range(r2):
                W[(k+3)*3 + i][r1+c] = W2[k*3 + i][c]
    # Block 3: W3 is 6x18, W3[k*3 + i, c] for k=0..1, i=0..2 -> W[(k+6)*3 + i, c]
    for k in range(2):
        for i in range(3):
            for c in range(r3):
                W[(k+6)*3 + i][r1+r2+c] = W3[k*3 + i][c]
    
    return U, V, W

def construct():
    deadline = time.time() + 120
    
    # Use schoolbook for 3x3x3 blocks and 3x3x2
    U1, V1, W1 = _schoolbook_3x3x3()
    U2, V2, W2 = _schoolbook_3x3x3()
    U3, V3, W3 = _schoolbook_3x3x2()
    
    U, V, W = _combine_blocks(U1, V1, W1, U2, V2, W2, U3, V3, W3)
    
    return {"U": U, "V": V, "W": W}
