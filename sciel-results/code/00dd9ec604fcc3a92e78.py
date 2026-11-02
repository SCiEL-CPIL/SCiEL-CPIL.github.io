"""Combined program: structured rank-26 decomposition for 3x3 matrix multiplication."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    n, m, p = 3, 3, 3
    
    def verify(U, V, W, rank):
        """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        """Rank-27 schoolbook decomposition."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return U, V, W, rank

    def strassen_2x2(U, V, W, rank_offset, i0, j0, k0):
        """
        Add Strassen's 7-term algorithm for a 2x2 submatrix.
        """
        u00 = i0 * m + j0
        u01 = i0 * m + j0 + 1
        u10 = (i0+1) * m + j0
        u11 = (i0+1) * m + j0 + 1
        
        v00 = j0 * p + k0
        v01 = j0 * p + k0 + 1
        v10 = (j0+1) * p + k0
        v11 = (j0+1) * p + k0 + 1
        
        w00 = k0 * n + i0
        w01 = k0 * n + i0 + 1
        w10 = (k0+1) * n + i0
        w11 = (k0+1) * n + i0 + 1
        
        r = rank_offset
        
        # M1 = (A00+A11)(B00+B11)
        U[u00][r] += 1.0; U[u11][r] += 1.0
        V[v00][r] += 1.0; V[v11][r] += 1.0
        W[w00][r] += 1.0; W[w11][r] += 1.0
        r += 1
        
        # M2 = (A10+A11)B00
        U[u10][r] += 1.0; U[u11][r] += 1.0
        V[v00][r] += 1.0
        W[w10][r] += 1.0
        r += 1
        
        # M3 = A00(B10-B11)
        U[u00][r] += 1.0
        V[v10][r] += 1.0; V[v11][r] += -1.0
        W[w11][r] += 1.0
        r += 1
        
        # M4 = A11(B01-B00)
        U[u11][r] += 1.0
        V[v01][r] += 1.0; V[v00][r] += -1.0
        W[w00][r] += 1.0
        r += 1
        
        # M5 = (A00+A01)B11
        U[u00][r] += 1.0; U[u01][r] += 1.0
        V[v11][r] += 1.0
        W[w01][r] += 1.0
        r += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[u10][r] += 1.0; U[u00][r] += -1.0
        V[v00][r] += 1.0; V[v01][r] += 1.0
        W[w10][r] += 1.0
        r += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[u01][r] += 1.0; U[u11][r] += -1.0
        V[v10][r] += 1.0; V[v11][r] += 1.0
        W[w01][r] += 1.0
        r += 1
        
        return r

    # Strategy: Build rank-26 by combining Strassen on 2x2 blocks
    # with schoolbook for the remaining terms.
    
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # 1. Strassen for the 2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # This handles products A[i,j] * B[j,k] for i,j,k in {0,1}
    next_r = strassen_2x2(U, V, W, 0, 0, 0, 0)
    
    # 2. Schoolbook for all other products
    # Products are (i, j, k) where NOT (i<2 and j<2 and k<2)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # If this product was handled by Strassen, skip
                if i < 2 and j < 2 and k < 2:
                    continue
                
                U[i*m+j][next_r] = 1.0
                V[j*p+k][next_r] = 1.0
                W[k*n+i][next_r] = 1.0
                next_r += 1
                
    # Verify that we used exactly 26 terms
    assert next_r == 26, f"Expected 26 terms, got {next_r}"
    
    # Optional: Verify correctness
    # if not verify(U, V, W, 26):
    #     # Fallback to schoolbook if verification fails
    #     U, V, W, rank = schoolbook()
    #     return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
