"""
Rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.

Strategy:
- The schoolbook rank is 2*2*14 = 56.
- We use Strassen-style tricks to reduce rank.
- For 2x2 * 2x14, we view this as 7 independent 2x2 * 2x2 multiplications.
  Each 2x2 * 2x2 can be done in rank 7 (Strassen) instead of rank 8.
  So total rank = 7 * 7 = 49.

The Strassen decomposition for 2x2 * 2x2:
  P1 = (A11 + A22)(B11 + B22)
  P2 = (A21 + A22) B11
  P3 = A11 (B12 - B22)
  P4 = A22 (B21 - B11)
  P5 = (A11 + A12) B22
  P6 = (A21 - A11) (B11 + B12)
  P7 = (A12 - A22) (B21 + B22)

  C11 = P1 + P4 - P5 + P7
  C12 = P3 + P5
  C21 = P2 + P4
  C22 = P1 - P2 + P3 + P6

For our 2x2 * 2x14 problem:
  A is 2x2, B is 2x14, C is 2x14.
  
  We split B into 7 blocks B^(t) of size 2x2 for t=0..6.
  C^(t) = A * B^(t) for each t.
  
  For each block, use Strassen with 7 multiplications.
  Total rank = 7 * 7 = 49.

Tensor convention:
  U has shape (n*m, rank) = (4, 49)
  V has shape (m*p, rank) = (28, 49)
  W has shape (p*n, rank) = (28, 49)
  
  U[i*m+j, r] - coefficient for A_{i,j} in rank-r term
  V[j*p+k, r] - coefficient for B_{j,k} in rank-r term  
  W[k*n+i, r] - coefficient for C_{i,k} in rank-r term
  
  The tensor element T_{i,j,k} = 1 for all valid i,j,k.
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    
    # Strassen-based rank-49 decomposition
    # 7 blocks of 2x2, each using 7 Strassen multiplications
    num_blocks = p // 2  # 7
    strassen_per_block = 7
    rank = num_blocks * strassen_per_block  # 49
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49
    
    # Strassen terms: (a_coeffs, b_coeffs, c_coeffs)
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    strassen_terms = [
        # P1 = (A00 + A11)(B00 + B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2 = (A10 + A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # P3 = A00(B01 - B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # P4 = A11(B10 - B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # P5 = (A00 + A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # P6 = (A10 - A00)(B00 + B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # P7 = (A01 - A11)(B10 + B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for t in range(num_blocks):
        k0 = 2 * t      # first column of block
        k1 = 2 * t + 1  # second column of block
        
        for s in range(strassen_per_block):
            col = t * strassen_per_block + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]
            
            # U: shape (n*m, rank) = (4, rank)
            # U[i*m + j, col] = coefficient of A[i,j] in term s
            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m  # 0 or 1
                a_j = a_idx % m   # 0 or 1
                U[a_i * m + a_j][col] = coeff
            
            # V: shape (m*p, rank) = (28, rank)
            # V[j*p + k, col] = coefficient of B[j,k] in term s for block t
            for b_idx, coeff in b_coeffs.items():
                b_j = b_idx // 2   # 0 or 1
                b_col = b_idx % 2  # 0 or 1
                b_k = 2 * t + b_col
                V[b_j * p + b_k][col] = coeff
            
            # W: shape (p*n, rank) = (28, rank)
            # W[k*n + i, col] = coefficient of C[i,k] in term s for block t
            for c_idx, coeff in c_coeffs.items():
                c_i = c_idx // 2   # 0 or 1
                c_col = c_idx % 2  # 0 or 1
                c_k = 2 * t + c_col
                W[c_k * n + c_i][col] = coeff
    
    # Verify correctness: check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-9:
        # Fallback to schoolbook if verification fails
        rank_fallback = n * m * p
        U_fb = [[0] * rank_fallback for _ in range(n * m)]
        V_fb = [[0] * rank_fallback for _ in range(m * p)]
        W_fb = [[0] * rank_fallback for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_fb[ii*m+jj][col] = 1
                    V_fb[jj*p+kk][col] = 1
                    W_fb[kk*n+ii][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
