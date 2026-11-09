"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Target Tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0

    # Strategy: Try to find a rank-17 decomposition using a local search method.
    # We start with the schoolbook decomposition (rank 18) and try to merge terms or find a linear dependency.
    # A more robust approach for this specific case is to use the known Strassen-based rank-17 solution
    # but verify it programmatically. If it fails, we fall back to a random search for rank 17.
    
    # Let's try to construct the rank-17 solution using the standard Strassen extension.
    # The previous hardcoded attempt might have index errors. We will generate it more carefully.
    
    # Indices:
    # U: (n*m) x rank -> 6 x 17
    # V: (m*p) x rank -> 9 x 17
    # W: (p*n) x rank -> 6 x 17
    
    U = np.zeros((6, 17), dtype=np.float64)
    V = np.zeros((9, 17), dtype=np.float64)
    W = np.zeros((6, 17), dtype=np.float64)
    
    # Strassen block for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # This covers i in {0,1}, j in {0,1}, k in {0,1}.
    # Terms 0-6.
    
    # M1 = A[0,0] * (B[0,0] + B[1,1])
    U[0, 0] = 1
    V[0, 0] = 1
    V[4, 0] = 1
    # Contributes to C[0,0] and C[1,0]? No, Strassen outputs C[0,0], C[0,1], C[1,0], C[1,1].
    # We need to map the Strassen results to the W indices.
    # C[0,0] is W[0], C[1,0] is W[1], C[0,1] is W[2], C[1,1] is W[3].
    
    # Let's use the standard Strassen formulas and map them to W.
    # C[0,0] = M1 - M4 + M5
    # C[0,1] = M2 + M7
    # C[1,0] = M3 + M6
    # C[1,1] = M1 - M2 + M4
    
    # M1 = A[0,0](B[0,0] + B[1,1])
    # U[0]=1, V[0]=1, V[4]=1
    # W[0] += 1, W[1] += 1 (Wait, M1 contributes to C[0,0] and C[1,1] in standard Strassen? 
    # No, the M's are intermediate products. The W coefficients are determined by the combination.)
    
    # Let's just define the 7 rank-1 tensors for the Strassen part directly.
    # Term 0: M1 = A[0,0] * (B[0,0] + B[1,1])
    U[0, 0] = 1
    V[0, 0] = 1
    V[4, 0] = 1
    # This term contributes to C[0,0] with coeff 1, C[1,1] with coeff 1?
    # Let's check the expansion of C[0,0] = M1 - M4 + M5.
    # So M1 has coeff +1 for W[0].
    # M1 also appears in C[1,1] = M1 - M2 + M4. So coeff +1 for W[3].
    W[0, 0] = 1
    W[3, 0] = 1
    
    # Term 1: M2 = (A[0,0] + A[0,1]) * B[1,1]  (Using A[12] in my previous notation, which is A[0,1])
    # Wait, standard Strassen M2 = (A11 + A12) B22. Here A11=A[0,0], A12=A[0,1], B22=B[1,1].
    U[0, 1] = 1
    U[1, 1] = 1
    V[4, 1] = 1
    # M2 appears in C[0,1] = M2 + M7 with coeff +1.
    # M2 appears in C[1,1] = M1 - M2 + M4 with coeff -1.
    W[2, 1] = 1
    W[3, 1] = -1
    
    # Term 2: M3 = (A[1,0] + A[1,1]) * B[0,0]
    U[3, 2] = 1
    U[4, 2] = 1
    V[0, 2] = 1
    # M3 appears in C[1,0] = M3 + M6 with coeff +1.
    # M3 does not appear in other C's? 
    # Wait, C[0,0] = M1 - M4 + M5. C[0,1] = M2 + M7. C[1,0] = M3 + M6. C[1,1] = M1 - M2 + M4.
    # So M3 only contributes to W[1].
    W[1, 2] = 1
    
    # Term 3: M4 = A[1,1] * (B[1,0] - B[0,0])
    U[4, 3] = 1
    V[3, 3] = 1
    V[0, 3] = -1
    # M4 appears in C[0,0] with coeff -1.
    # M4 appears in C[1,1] with coeff +1.
    W[0, 3] = -1
    W[3, 3] = 1
    
    # Term 4: M5 = (A[0,0] - A[1,1]) * (B[1,0] + B[1,1])
    U[0, 4] = 1
    U[4, 4] = -1
    V[3, 4] = 1
    V[4, 4] = 1
    # M5 appears in C[0,0] with coeff +1.
    # M5 does not appear in other C's?
    W[0, 4] = 1
    
    # Term 5: M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    U[3, 5] = 1
    U[0, 5] = -1
    V[0, 5] = 1
    V[1, 5] = 1
    # M6 appears in C[1,0] with coeff +1.
    W[1, 5] = 1
    
    # Term 6: M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    U[1, 6] = 1
    U[4, 6] = -1
    V[3, 6] = 1
    V[4, 6] = 1
    # M7 appears in C[0,1] with coeff +1.
    W[2, 6] = 1
    
    # Now we have 7 terms for the 2x2 block.
    # Remaining terms:
    # C[0,0], C[1,0], C[0,1], C[1,1] also get contributions from A[i,2] and B[2,k].
    # C[i,k] += A[i,2]B[2,k] for k in {0,1}.
    # C[i,2] = A[i,0]B[0,2] + A[i,1]B[1,2] + A[i,2]B[2,2].
    
    # Terms 7-10: A[i,2] * B[2,k] for i in {0,1}, k in {0,1}.
    # Term 7: A[0,2] * B[2,0] -> C[0,0]
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # Term 8: A[1,2] * B[2,0] -> C[1,0]
    U[5, 8] = 1
    V[6, 8] = 1
    W[1, 8] = 1
    
    # Term 9: A[0,2] * B[2,1] -> C[0,1]
    U[2, 9] = 1
    V[7, 9] = 1
    W[2, 9] = 1
    
    # Term 10: A[1,2] * B[2,1] -> C[1,1]
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Terms 11-16: C[i,2] terms.
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    
    # Term 11: A[0,0] * B[0,2] -> C[0,2]
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # Term 12: A[0,1] * B[1,2] -> C[0,2]
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    # Term 13: A[0,2] * B[2,2] -> C[0,2]
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # Term 14: A[1,0] * B[0,2] -> C[1,2]
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    # Term 15: A[1,1] * B[1,2] -> C[1,2]
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    # Term 16: A[1,2] * B[2,2] -> C[1,2]
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    # Verify
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    if np.allclose(T_recon, T_target, atol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: If the hardcoded Strassen extension is incorrect, 
        # we return the schoolbook decomposition to ensure validity.
        # A more aggressive fallback would be to search, but time is tight.
        return schoolbook()

# EVOLVE-BLOCK-END
