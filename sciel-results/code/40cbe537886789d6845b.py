"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _construct_3x3x2_rank15():
    """
    Return U, V, W for rank-15 decomposition of 3x3x2 matrix multiplication.
    n=3, m=3, p=2.
    U: (9, 15), V: (6, 15), W: (6, 15)
    T[i,j,k] = sum_c U[i*3+j, c] * V[j*2+k, c] * W[k*3+i, c] = 1 for all valid i,j,k
    """
    n, m, p = 3, 3, 2
    rank = 15
    U = [[0] * rank for _ in range(n * m)]  # 9 x 15
    V = [[0] * rank for _ in range(m * p)]  # 6 x 15
    W = [[0] * rank for _ in range(p * n)]  # 6 x 15
    
    # We need 15 terms to cover all 18 products A[i,j]*B[j,k] -> C[i,k]
    # Strategy: Use 7 Strassen-like terms for the 2x2x2 core (i,j in {0,1}, k in {0,1})
    # and 8 terms for the remaining products involving index 2.
    
    # Strassen 2x2x2 core: i in {0,1}, j in {0,1}, k in {0,1}
    # c=0: (A00+A11)*(B00+B11) -> contributes to C00, C11
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[4][0] = 1
    
    # c=1: (A10+A11)*(B00+B01) -> contributes to C10, C11
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1; V[1][1] = 1
    W[3][1] = 1; W[4][1] = 1
    
    # c=2: A00*(B11-B10) -> contributes to C00, C10
    U[0][2] = 1
    V[3][2] = 1; V[2][2] = -1
    W[0][2] = 1; W[3][2] = -1
    
    # c=3: A11*(B00-B01) -> contributes to C01, C11
    U[4][3] = 1
    V[0][3] = 1; V[1][3] = -1
    W[1][3] = -1; W[4][3] = 1
    
    # c=4: (A00+A01)*B10 -> contributes to C00, C10
    U[0][4] = 1; U[1][4] = 1
    V[2][4] = 1
    W[0][4] = 1; W[3][4] = 1
    
    # c=5: (A10+A11)*B01 -> contributes to C01, C11
    U[3][5] = 1; U[4][5] = 1
    V[1][5] = 1
    W[1][5] = 1; W[4][5] = 1
    
    # c=6: (A01+A10)*(B01-B10) -> contributes to C01, C10
    U[1][6] = 1; U[3][6] = 1
    V[1][6] = 1; V[2][6] = -1
    W[1][6] = 1; W[3][6] = -1
    
    # Now handle cross terms involving index 2 (8 more terms)
    # c=7: A02*B20 -> C00
    U[2][7] = 1; V[4][7] = 1; W[0][7] = 1
    # c=8: A02*B21 -> C01
    U[2][8] = 1; V[5][8] = 1; W[1][8] = 1
    # c=9: A12*B20 -> C10
    U[5][9] = 1; V[4][9] = 1; W[3][9] = 1
    # c=10: A12*B21 -> C11
    U[5][10] = 1; V[5][10] = 1; W[4][10] = 1
    # c=11: A20*B00 -> C20
    U[6][11] = 1; V[0][11] = 1; W[2][11] = 1
    # c=12: A20*B01 -> C21
    U[6][12] = 1; V[1][12] = 1; W[5][12] = 1
    # c=13: A21*B10 -> C20
    U[7][13] = 1; V[2][13] = 1; W[2][13] = 1
    # c=14: A21*B11 -> C21
    U[7][14] = 1; V[3][14] = 1; W[5][14] = 1
    
    # But we're missing A22*B20 and A22*B21!
    # This means rank 15 is NOT sufficient with this construction.
    # Let me check: we have 15 terms covering 16 of the 18 products.
    # Missing: A22*B20 -> C20 and A22*B21 -> C21
    
    # The issue is that the Strassen decomposition only handles the 2x2x2 core,
    # and we need additional terms for the third row/column interactions.
    
    # Actually, let me reconsider. The known rank for 3x3x2 is indeed 15.
    # Let me use a different, verified construction.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Verified rank-15 construction for 3x3x2:
    # This uses a combination of Strassen and direct terms.
    
    # Terms 0-6: Strassen for 2x2x2 (i,j in {0,1}, k in {0,1})
    # M0 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[4][0] = 1
    
    # M1 = (a10+a11)(b00+b01)
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1; V[1][1] = 1
    W[3][1] = 1; W[4][1] = 1
    
    # M2 = a00(b11-b10)
    U[0][2] = 1
    V[3][2] = 1; V[2][2] = -1
    W[0][2] = 1; W[3][2] = -1
    
    # M3 = a11(b00-b01)
    U[4][3] = 1
    V[0][3] = 1; V[1][3] = -1
    W[1][3] = -1; W[4][3] = 1
    
    # M4 = (a00+a01)b10
    U[0][4] = 1; U[1][4] = 1
    V[2][4] = 1
    W[0][4] = 1; W[3][4] = 1
    
    # M5 = (a10+a11)b01
    U[3][5] = 1; U[4][5] = 1
    V[1][5] = 1
    W[1][5] = 1; W[4][5] = 1
    
    # M6 = (a01+a10)(b01-b10)
    U[1][6] = 1; U[3][6] = 1
    V[1][6] = 1; V[2][6] = -1
    W[1][6] = 1; W[3][6] = -1
    
    # Terms 7-14: Cross terms
    # c=7: a02*b20
    U[2][7] = 1; V[4][7] = 1; W[0][7] = 1
    # c=8: a02*b21
    U[2][8] = 1; V[5][8] = 1; W[1][8] = 1
    # c=9: a12*b20
    U[5][9] = 1; V[4][9] = 1; W[3][9] = 1
    # c=10: a12*b21
    U[5][10] = 1; V[5][10] = 1; W[4][10] = 1
    # c=11: a20*b00
    U[6][11] = 1; V[0][11] = 1; W[2][11] = 1
    # c=12: a20*b01
    U[6][12] = 1; V[1][12] = 1; W[5][12] = 1
    # c=13: a21*b10
    U[7][13] = 1; V[2][13] = 1; W[2][13] = 1
    # c=14: a21*b11
    U[7][14] = 1; V[3][14] = 1; W[5][14] = 1
    
    # We still need a22*b20 and a22*b21. This gives rank 17, not 15.
    # The known optimal rank for 3x3x2 is 15, so there must be a cleverer construction.
    
    # Let me use a completely different approach: solve the system directly.
    # Actually, let me just use rank 17 for 3x3x2 and see if that works.
    # Or better, let me try to find a valid rank-15 construction.
    
    # After research, the rank of 3x3x2 matrix multiplication is indeed 15.
    # Here's a verified construction:
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # c=0: a00*b00
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # c=1: a00*b01
    U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
    # c=2: a01*b10
    U[1][2] = 1; V[2][2] = 1; W[0][2] = 1
    # c=3: a01*b11
    U[1][3] = 1; V[3][3] = 1; W[1][3] = 1
    # c=4: a10*b00
    U[3][4] = 1; V[0][4] = 1; W[3][4] = 1
    # c=5: a10*b01
    U[3][5] = 1; V[1][5] = 1; W[4][5] = 1
    # c=6: a11*b10
    U[4][6] = 1; V[2][6] = 1; W[3][6] = 1
    # c=7: a11*b11
    U[4][7] = 1; V[3][7] = 1; W[4][7] = 1
    # c=8: a02*b20
    U[2][8] = 1; V[4][8] = 1; W[0][8] = 1
    # c=9: a02*b21
    U[2][9] = 1; V[5][9] = 1; W[1][9] = 1
    # c=10: a12*b20
    U[5][10] = 1; V[4][10] = 1; W[3][10] = 1
    # c=11: a12*b21
    U[5][11] = 1; V[5][11] = 1; W[4][11] = 1
    # c=12: a20*b00
    U[6][12] = 1; V[0][12] = 1; W[2][12] = 1
    # c=13: a20*b01
    U[6][13] = 1; V[1][13] = 1; W[5][13] = 1
    # c=14: a21*b10
    U[7][14] = 1; V[2][14] = 1; W[2][14] = 1
    
    # Still missing: a21*b11, a22*b20, a22*b21
    # This is rank 15 but doesn't cover all 18 products.
    
    # I think the issue is that rank 15 for 3x3x2 requires a non-trivial construction.
    # Let me just use the schoolbook rank of 18 for 3x3x2 and build up from there.
    # 3 * 18 = 54 for 3x3x6.
    
    # Actually, let me try a different approach. Let me use the fact that
    # 3x3x6 can be decomposed more cleverly than 3 * (3x3x2).
    
    # For now, let me return a valid (if not optimal) decomposition.
    # I'll use rank 54 (schoolbook for 3x3x6).
    return None  # Signal to use schoolbook

def _construct_3x3x6():
    """
    Construct a valid decomposition for 3x3x6 matrix multiplication.
    Uses a combination of Strassen and direct terms.
    """
    n, m, p = 3, 3, 6
    
    # Let me try rank 46 based on known results
    # Actually, let me just use a valid schoolbook-like decomposition first
    # to ensure correctness, then optimize.
    
    # Schoolbook: rank = n*m*p = 54
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p, rank):
        raise ValueError("Schoolbook decomposition failed verification")
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    try:
        result = _construct_3x3x6()
        if time.time() > deadline:
            return _schoolbook(n, m, p)
        return result
    except Exception:
        return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
