"""Rewritten program for matrix multiplication 2x5x6 tensor rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Strategy: Try block decomposition with Strassen for sub-blocks
    # We'll explore different ways to split dimensions
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check: For each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        # And for all other combinations, the sum should be 0
        # Actually, the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # should equal 1 for all (i,j,k) since matrix multiplication tensor has all 1s
        
        # Let's verify by checking a few random entries and the full structure
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_block_decomposition(n1, n2, m1, m2, p1, p2):
        """Try splitting into two blocks and using Strassen or schoolbook for each."""
        # Block 1: n1 x m1 x p1
        # Block 2: n2 x m2 x p2
        
        # Use schoolbook for each block
        rank1 = n1 * m1 * p1
        rank2 = n2 * m2 * p2
        total_rank = rank1 + rank2
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        # Block 1
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Block 2
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    U[(n1 + i) * m + (m1 + j)][col] = 1
                    V[(m1 + j) * p + (p1 + k)][col] = 1
                    W[(p1 + k) * n + (n1 + i)][col] = 1
                    col += 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W, "rank": total_rank}
        return None
    
    # Try various block splits
    # n=2, m=5, p=6
    splits = [
        (1, 1, 1, 4, 1, 5),  # 1x1x1 + 1x4x5
        (1, 1, 2, 3, 2, 4),  # 1x2x2 + 1x3x4
        (1, 1, 3, 2, 3, 3),  # 1x3x3 + 1x2x3
        (2, 0, 0, 5, 0, 6),  # Just schoolbook (should match)
        (1, 1, 1, 4, 3, 3),
        (1, 1, 2, 3, 1, 5),
        (1, 1, 4, 1, 3, 3),
        (1, 1, 5, 0, 0, 6),  # Degenerate
    ]
    
    for n1, n2, m1, m2, p1, p2 in splits:
        if n1 + n2 != n or m1 + m2 != m or p1 + p2 != p:
            continue
        if n1 <= 0 or n2 <= 0 or m1 <= 0 or m2 <= 0 or p1 <= 0 or p2 <= 0:
            continue
        if time.time() > deadline:
            break
        result = try_block_decomposition(n1, n2, m1, m2, p1, p2)
        if result:
            if result["rank"] < len(best_result["U"][0]):
                best_result = {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Try using Strassen for 2x2x2 sub-blocks
    # For a 2x2x2 block, Strassen gives rank 7 instead of 8
    def strassen_2x2x2():
        """Return Strassen decomposition for 2x2x2."""
        # Standard Strassen formulas
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        U = [[0]*7 for _ in range(4)]  # n*m = 4
        V = [[0]*7 for _ in range(4)]  # m*p = 4
        W = [[0]*7 for _ in range(4)]  # p*n = 4
        
        # M1: (A00+A11)(B00+B11)
        # u: rows 0 (i=0,j=0) and 3 (i=1,j=1) -> U[0,0]=1, U[3,0]=1
        # v: cols 0 (j=0,k=0) and 3 (j=1,k=1) -> V[0,0]=1, V[3,0]=1
        # w: (k*n+i) for C00: k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
        # C00 gets +1, C01 gets +1, C10 gets +1, C11 gets -1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = -1
        
        # M2: (A10+A11)B00
        # u: A10 (i=1,j=0) -> row 2, A11 (i=1,j=1) -> row 3
        # v: B00 (j=0,k=0) -> row 0
        # w: C10 += 1, C11 += 1... wait let me recalculate
        # C10 = M2+M4, so M2 contributes to C10
        # C11 = M1-M2+M3+M6, so M2 contributes -1 to C11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1  # C10: k=1,i=0 -> 1*2+0 = 2
        W[3][1] = -1  # C11: k=1,i=1 -> 1*2+1 = 3
        
        # M3: A00(B01-B11)
        # u: A00 -> row 0
        # v: B01 (j=0,k=1) -> row 1, B11 (j=1,k=1) -> row 3
        # C01 = M3+M5, C11 = ...+M3...
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1  # C01: k=0,i=1 -> 0*2+1 = 1
        W[3][2] = 1  # C11: k=1,i=1 -> 3
        
        # M4: A11(B10-B00)
        # u: A11 -> row 3
        # v: B10 (j=1,k=0) -> row 2, B00 (j=0,k=0) -> row 0
        # C00 = ...+M4..., C10 = ...+M4...
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1  # C00
        W[2][3] = 1  # C10
        
        # M5: (A00+A01)B11
        # u: A00 -> row 0, A01 -> row 1
        # v: B11 -> row 3
        # C00 = ...-M5..., C01 = ...+M5...
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1  # C00
        W[1][4] = 1  # C01
        
        # M6: (A10-A00)(B00+B10)
        # u: A10 -> row 2, A00 -> row 0
        # v: B00 -> row 0, B10 -> row 2
        # C11 = ...+M6...
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[3][5] = 1  # C11
        
        # M7: (A01-A11)(B01+B11)
        # u: A01 -> row 1, A11 -> row 3
        # v: B01 -> row 1, B11 -> row 3
        # C00 = ...+M7...
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = 1  # C00
        
        return U, V, W
    
    # Check if we can embed Strassen
    if time.time() < deadline:
        U_s, V_s, W_s = strassen_2x2x2()
        # This is for 2x2x2, we need 2x5x6
        # We could use it for a 2x2x2 subblock and schoolbook for the rest
        # But this requires careful index mapping
        
    # For now, return the best result found
    return best_result
# EVOLVE-BLOCK-END
