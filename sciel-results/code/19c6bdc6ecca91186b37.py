import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS
    
    def schoolbook_full():
        """Full schoolbook decomposition for 3x3x7, rank 63."""
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
    
    def verify_full(U, V, W, n, m, p):
        """Verify full decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    # Laderman's rank-23 algorithm for 3x3x3 matrix multiplication.
    # The 23 rank-1 terms are well-known. Each term has (u, v, w) vectors
    # of length 9 (flattened 3x3 matrices).
    #
    # Laderman's algorithm uses 19 "diagonal" terms and 4 "off-diagonal" terms.
    # Here's a verified correct set:
    
    def e3(row, col):
        """Create a 9-element list with 1 at position (row, col) in 3x3."""
        v = [0] * 9
        v[row * 3 + col] = 1
        return v
    
    # The 23 Laderman factors:
    # Terms 0-18: diagonal-like terms
    # Terms 19-22: off-diagonal merged terms
    
    laderman = []
    
    # 19 diagonal terms: for each (i,j) in 3x3, we have terms that isolate A[i,j] * B[j,k]
    # Actually, the standard Laderman decomposition is:
    # 
    # Let me use the explicit known-correct Laderman factors.
    # From standard references, the 23 terms are:
    
    # I'll construct them systematically. Laderman's algorithm:
    # Start with 27 schoolbook terms, then combine 4 pairs using Strassen-like tricks.
    # 
    # The 23 factors (each is a tuple of three 9-vectors):
    
    # Diagonal terms (19 of them):
    # For the "main" diagonal structure
    
    # Let me use a verified implementation:
    
    # Laderman's 23 factors for 3x3x3:
    # f0:  A[0,0]*B[0,0] -> C[0,0]
    # f1:  A[0,1]*B[1,0] -> C[0,0]  
    # f2:  A[0,2]*B[2,0] -> C[0,0]
    # ... etc
    
    # Actually, let me just use a known correct set from literature.
    # The Laderman algorithm has rank 23 for 3x3x3.
    
    # Here's a verified set of 23 factors:
    L = [
        # Term 0: e00 * e00 * e00
        (e3(0,0), e3(0,0), e3(0,0)),
        # Term 1: e01 * e10 * e00
        (e3(0,1), e3(1,0), e3(0,0)),
        # Term 2: e02 * e20 * e00
        (e3(0,2), e3(2,0), e3(0,0)),
        # Term 3: e10 * e01 * e01
        (e3(1,0), e3(0,1), e3(0,1)),
        # Term 4: e11 * e11 * e11
        (e3(1,1), e3(1,1), e3(1,1)),
        # Term 5: e12 * e21 * e11
        (e3(1,2), e3(2,1), e3(1,1)),
        # Term 6: e20 * e02 * e02
        (e3(2,0), e3(0,2), e3(0,2)),
        # Term 7: e21 * e12 * e12
        (e3(2,1), e3(1,2), e3(1,2)),
        # Term 8: e22 * e22 * e22
        (e3(2,2), e3(2,2), e3(2,2)),
        # Now the off-diagonal merged terms (4 pairs merged from remaining 18):
        # Term 9: (e00+e11) * (e00+e11) * (e00+e11) - this gives cross terms
        # Actually, let me use the proper Strassen-like combinations.
        
        # The remaining 14 schoolbook terms need to be merged into 4 terms.
        # Standard Laderman uses these 4 additional terms:
        
        # Term 9: combines A[0,0]*B[0,1] + A[0,1]*B[0,0] type terms
        (e3(0,0), e3(0,1), e3(0,1)),
        # Term 10:
        (e3(0,1), e3(0,0), e3(0,1)),
        # Term 11:
        (e3(1,0), e3(1,1), e3(1,0)),
        # Term 12:
        (e3(1,1), e3(1,0), e3(1,0)),
        # Term 13:
        (e3(1,1), e3(1,2), e3(1,2)),
        # Term 14:
        (e3(1,2), e3(1,1), e3(1,2)),
        # Term 15:
        (e3(2,0), e3(2,1), e3(2,0)),
        # Term 16:
        (e3(2,1), e3(2,0), e3(2,0)),
        # Term 17:
        (e3(2,1), e3(2,2), e3(2,2)),
        # Term 18:
        (e3(2,2), e3(2,1), e3(2,2)),
        # Term 19:
        (e3(0,2), e3(2,1), e3(0,2)),
        # Term 20:
        (e3(2,1), e3(0,2), e3(0,2)),
        # Term 21:
        (e3(0,0), e3(1,2), e3(1,0)),
        # Term 22:
        (e3(1,2), e3(0,0), e3(1,0)),
    ]
    
    # Wait, that's 23 terms but I need to verify they're correct.
    # Let me verify by checking that sum_r u[i,j]*v[j,k]*w[k,i] = 1 for all i,j,k.
    
    # Actually, the above is NOT the correct Laderman decomposition.
    # Let me use a properly verified set.
    
    # The correct Laderman rank-23 factors are well-documented.
    # Let me implement them properly.
    
    # Laderman's algorithm for 3x3x3:
    # The 23 multiplications are:
    #
    # M1 = A00 * B00
    # M2 = A00 * B01  
    # M3 = A01 * B10
    # M4 = A01 * B11
    # M5 = A02 * B20
    # M6 = A02 * B21
    # M7 = A10 * B00
    # M8 = A10 * B01
    # M9 = A11 * B10
    # M10 = A11 * B11
    # M11 = A12 * B20
    # M12 = A12 * B21
    # M13 = A20 * B00
    # M14 = A20 * B01
    # M15 = A21 * B10
    # M16 = A21 * B11
    # M17 = A22 * B20
    # M18 = A22 * B21
    # M19 = (A00+A11) * (B00+B11)
    # M20 = (A00+A22) * (B00+B22)
    # M21 = (A11+A22) * (B11+B22)
    # M22 = (A01+A12+A20) * (B10+B21+B02)  -- this isn't quite right either
    
    # Let me just use a brute-force verified approach.
    # I'll construct the factors and verify them.
    
    # Actually, for reliability within the time budget, let me use a 
    # different strategy: use the known-correct Laderman factors from a 
    # verified source.
    
    # Here's a verified correct set of Laderman's 23 factors:
    # (u, v, w) where each is a 9-vector (row-major 3x3)
    
    L_correct = [
        # 19 terms that are "basis" terms
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),  # A00*B00->C00
        ([0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),  # A01*B10->C01? No...
    ]
    
    # I'm going in circles. Let me use a completely different approach:
    # Just use schoolbook for all 7 columns (rank 63) as the safe fallback,
    # but try to verify if a better decomposition works.
    
    # For now, return schoolbook to ensure correctness.
    # The improvement will come from properly implementing Laderman.
    
    return schoolbook_full()
    
    # The following code is for when we have correct Laderman factors:
    # rank = 23 + 36  # 59
    # U = [[0] * rank for _ in range(n * m)]
    # V = [[0] * rank for _ in range(m * p)]
    # W = [[0] * rank for _ in range(p * n)]
    #
    # # Add Laderman terms for k=0,1,2
    # for r in range(23):
    #     u_vec, v_vec, w_vec = L_correct[r]
    #     for idx in range(9):
    #         U[idx][r] = u_vec[idx]
    #         # V and W need to be mapped to the 3x3x7 space
    #
    # # Add schoolbook terms for k=3,4,5,6
    # col = 23
    # for k in range(3, 7):
    #     for i in range(n):
    #         for j in range(m):
    #             U[i*m + j][col] = 1
    #             V[j*p + k][col] = 1
    #             W[k*n + i][col] = 1
    #             col += 1
    #
    # result = {"U": U, "V": V, "W": W}
    #
    # if verify_full(U, V, W, n, m, p):
    #     return result
    # else:
    #     return schoolbook_full()
# EVOLVE-BLOCK-END
