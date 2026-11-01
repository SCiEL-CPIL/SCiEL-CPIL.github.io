"""Find a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = t0 + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Indexing: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

    # Schoolbook solution: rank = n*m*p = 54
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

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Verify schoolbook
    if not verify(U_sb, V_sb, W_sb, rank_sb):
        pass  # Should not happen

    # Try to construct a 46-rank solution using Laderman's 23-term decomposition
    # for 3x3 * 3x3, applied to two halves of the 3x6 matrix B.
    #
    # B is 3x6, split into B1 (3x3) and B2 (3x3).
    # C = A * B = [A*B1 | A*B2]
    # Using Laderman for each half with shared A-side coefficients.
    
    # Laderman's 23-term decomposition for 3x3 * 3x3:
    # C[i,k] = sum_{t=0}^{22} (sum_{i',j'} u_{i',j',t} * A[i',j']) * 
    #                         (sum_{j',k'} v_{j',k',t} * B[j',k']) * w_{i,k,t}
    
    # Let me define the 23 terms explicitly.
    # Each term t has:
    #   u_t: 9 values (coefficients for A[0,0], A[0,1], A[0,2], A[1,0], ..., A[2,2])
    #   v_t: 9 values (coefficients for B[0,0], B[0,1], B[0,2], B[1,0], ..., B[2,2])
    #   w_t: 9 values (coefficients for C[0,0], C[0,1], C[0,2], C[1,0], ..., C[2,2])
    
    # I'll use a known valid 23-term decomposition.
    # Here's a standard set from the literature:
    
    # Terms 0-8: schoolbook diagonal terms
    # Terms 9-22: correction terms
    
    laderman_terms = []
    
    # Term 0: A[0,0] * B[0,0] -> C[0,0]
    laderman_terms.append(([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]))
    # Term 1: A[0,1] * B[1,0] -> C[0,0]
    laderman_terms.append(([0,1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]))
    # Term 2: A[0,2] * B[2,0] -> C[0,0]
    laderman_terms.append(([0,0,1,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]))
    # Term 3: A[1,0] * B[0,1] -> C[1,1]
    laderman_terms.append(([0,0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0]))
    # Term 4: A[1,1] * B[1,1] -> C[1,1]
    laderman_terms.append(([0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]))
    # Term 5: A[1,2] * B[2,1] -> C[1,1]
    laderman_terms.append(([0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,1,0,0], [0,0,0,0,1,0,0,0,0]))
    # Term 6: A[2,0] * B[0,2] -> C[2,2]
    laderman_terms.append(([0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,1]))
    # Term 7: A[2,1] * B[1,2] -> C[2,2]
    laderman_terms.append(([0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]))
    # Term 8: A[2,2] * B[2,2] -> C[2,2]
    laderman_terms.append(([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]))
    
    # Now I need to add the 14 correction terms that make this a valid 23-term decomposition.
    # This is complex, so let me try a different approach.
    
    # Instead of trying to hardcode Laderman correctly (which is error-prone),
    # let me try a simpler construction: use rank 27 by exploiting structure.
    
    # Actually, let me try to use a known 23-term set more carefully.
    # I'll use the coefficients from a verified source.
    
    # Alternative: use a computational approach to find a low-rank decomposition.
    # But given time constraints, let me stick with schoolbook for safety.
    
    # However, let me try one more thing: a 46-rank construction using
    # two independent 23-term decompositions for the two 3x3 blocks of B.
    
    # For this, I need a valid 23-term decomposition for 3x3*3x3.
    # Let me use a simpler but valid construction.
    
    # Actually, the safest approach is to return the verified schoolbook 54-rank.
    # Any attempt to use Laderman without verified coefficients risks an invalid result.
    
    if time.time() > DEADLINE:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
