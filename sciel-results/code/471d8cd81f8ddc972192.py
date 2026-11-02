import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target rank to try
    target_rank = 15
    
    # Time management
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Try to construct a rank-16 solution using Bini et al. algorithm
    # For 2x3x3 matrix multiplication, rank 16 is achievable
    R16 = 16
    
    # Construct U, V, W for rank 16
    # U: (n*m, R16) = (6, 16)
    # V: (m*p, R16) = (9, 16)
    # W: (p*n, R16) = (6, 16)
    
    U = np.zeros((n*m, R16))
    V = np.zeros((m*p, R16))
    W = np.zeros((p*n, R16))
    
    # Indices:
    # U[i*m+j, r] where i in [0,1], j in [0,2]
    # V[j*p+k, r] where j in [0,2], k in [0,2]
    # W[k*n+i, r] where k in [0,2], i in [0,1]
    
    # Bini et al. rank-16 construction for 2x3x3
    # We use the following structure based on the known rank-16 algorithm
    
    # Define the 16 rank-1 terms
    # Each term r contributes U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    
    # Let's build this systematically
    # The 2x3x3 multiplication tensor has 18 entries
    # We need to represent it as sum of 16 rank-1 tensors
    
    # Using the known rank-16 decomposition:
    # We'll use a combination of Strassen-like blocks
    
    # For the 2x3 case, we can think of it as:
    # Row 1 of C = [A11, A12] * B (where A11 is 1x1, A12 is 1x2)
    # Row 2 of C = [A21, A22] * B (where A21 is 1x1, A22 is 1x2)
    
    # Let's use the explicit rank-16 construction
    # Based on the Bini et al. paper, the rank-16 algorithm for 2x3x3
    
    # Term 0: A11 * B11
    U[0, 0] = 1  # i=0, j=0
    V[0, 0] = 1  # j=0, k=0
    W[0, 0] = 1  # k=0, i=0
    
    # Term 1: A11 * B21
    U[0, 1] = 1
    V[3, 1] = 1  # j=1, k=0
    W[0, 1] = 1
    
    # Term 2: A11 * B31
    U[0, 2] = 1
    V[6, 2] = 1  # j=2, k=0
    W[0, 2] = 1
    
    # Term 3: A12[0] * B12
    U[1, 3] = 1  # i=0, j=1
    V[1, 3] = 1  # j=0, k=1
    W[1, 3] = 1  # k=1, i=0
    
    # Term 4: A12[0] * B22
    U[1, 4] = 1
    V[4, 4] = 1  # j=1, k=1
    W[1, 4] = 1
    
    # Term 5: A12[0] * B32
    U[1, 5] = 1
    V[7, 5] = 1  # j=2, k=1
    W[1, 5] = 1
    
    # Term 6: A12[1] * B13
    U[2, 6] = 1  # i=0, j=2
    V[2, 6] = 1  # j=0, k=2
    W[2, 6] = 1  # k=2, i=0
    
    # Term 7: A12[1] * B23
    U[2, 7] = 1
    V[5, 7] = 1  # j=1, k=2
    W[2, 7] = 1
    
    # Term 8: A12[1] * B33
    U[2, 8] = 1
    V[8, 8] = 1  # j=2, k=2
    W[2, 8] = 1
    
    # Terms 9-15: For the second row, using Strassen-like combinations
    # We need 7 more terms to handle row 2 with sharing
    
    # Term 9: A21 * (B11 + B21) - A11*B21 (to get A21*B11)
    # Actually, let's use a cleaner approach
    
    # For row 2:
    # C[1,0] = A21*B11 + A22[0]*B21 + A22[1]*B31
    # C[1,1] = A21*B12 + A22[0]*B22 + A22[1]*B32
    # C[1,2] = A21*B13 + A22[0]*B23 + A22[1]*B33
    
    # We can reuse some of the B terms from row 1
    
    # Term 9: A21 * B11 (new)
    U[3, 9] = 1  # i=1, j=0
    V[0, 9] = 1
    W[3, 9] = 1  # k=0, i=1
    
    # Term 10: A21 * B21 (new)
    U[3, 10] = 1
    V[3, 10] = 1
    W[3, 10] = 1
    
    # Term 11: A21 * B31 (new)
    U[3, 11] = 1
    V[6, 11] = 1
    W[3, 11] = 1
    
    # Term 12: A22[0] * B12 (new)
    U[4, 12] = 1  # i=1, j=1
    V[1, 12] = 1
    W[4, 12] = 1  # k=1, i=1
    
    # Term 13: A22[0] * B22 (new)
    U[4, 13] = 1
    V[4, 13] = 1
    W[4, 13] = 1
    
    # Term 14: A22[0] * B32 (new)
    U[4, 14] = 1
    V[7, 14] = 1
    W[4, 14] = 1
    
    # Term 15: A22[1] * B13 (new)
    U[5, 15] = 1  # i=1, j=2
    V[2, 15] = 1
    W[5, 15] = 1  # k=2, i=1
    
    # Wait, this gives us 16 terms but we're missing A22[1]*B23 and A22[1]*B33
    # This simple approach gives rank 18, not 16
    
    # Let me use the actual Bini et al. rank-16 algorithm
    # The key is to use shared intermediate products
    
    # Let me restart with the proper rank-16 construction
    
    U = np.zeros((n*m, R16))
    V = np.zeros((m*p, R16))
    W = np.zeros((p*n, R16))
    
    # The rank-16 algorithm for 2x3x3 uses the following:
    # We define intermediate products that are shared between rows
    
    # Let A = [[a, b, c], [d, e, f]] (2x3)
    # Let B be 3x3
    
    # C[0,0] = a*B[0,0] + b*B[1,0] + c*B[2,0]
    # C[0,1] = a*B[0,1] + b*B[1,1] + c*B[2,1]
    # C[0,2] = a*B[0,2] + b*B[1,2] + c*B[2,2]
    # C[1,0] = d*B[0,0] + e*B[1,0] + f*B[2,0]
    # C[1,1] = d*B[0,1] + e*B[1,1] + f*B[2,1]
    # C[1,2] = d*B[0,2] + e*B[1,2] + f*B[2,2]
    
    # Rank-16 construction:
    # P0 = a * B[:,0]  (contributes to C[0,0])
    # P1 = b * B[:,1]  (contributes to C[0,1])
    # P2 = c * B[:,2]  (contributes to C[0,2])
    # P3 = d * B[:,0]  (contributes to C[1,0])
    # P4 = e * B[:,1]  (contributes to C[1,1])
    # P5 = f * B[:,2]  (contributes to C[1,2])
    # That's only 6, we need more for the cross terms
    
    # Actually, the standard schoolbook is 18. To get 16, we need to share.
    
    # Let me use a different approach: try to find a valid rank-16 solution
    # using optimization with a better initialization
    
    # Fallback: use the known rank-18 solution and try to optimize to rank 16
    # But first, let's just return rank 16 if we can verify it, else rank 18
    
    # Let me try a simpler approach: use the Bini et al. explicit construction
    
    # After research, the rank-16 algorithm for 2x3x3 is:
    # It uses the fact that we can write the 2x3 multiplication using
    # 16 bilinear forms
    
    # Let me construct it properly:
    
    # The 18 entries of the tensor T[i,j,k] for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # T[i,j,k] = delta(i,j,k) in the sense that the multiplication tensor has 1s at all 18 positions
    
    # We want to write this as sum_{r=1}^{16} u_r ⊗ v_r ⊗ w_r
    
    # Here's a valid rank-16 construction:
    
    # Row 0 (i=0):
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    
    # Row 1 (i=1):
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # Rank-16 uses shared products. Let me define:
    # M0 = A[0,0] * B[0,0]
    # M1 = A[0,1] * B[1,0]
    # M2 = A[0,2] * B[2,0]
    # M3 = A[0,0] * B[0,1]
    # M4 = A[0,1] * B[1,1]
    # M5 = A[0,2] * B[2,1]
    # M6 = A[0,0] * B[0,2]
    # M7 = A[0,1] * B[1,2]
    # M8 = A[0,2] * B[2,2]
    # M9 = A[1,0] * B[0,0]
    # M10 = A[1,1] * B[1,0]
    # M11 = A[1,2] * B[2,0]
    # M12 = A[1,0] * B[0,1]
    # M13 = A[1,1] * B[1,1]
    # M14 = A[1,2] * B[2,1]
    # M15 = A[1,0] * B[0,2]
    # M16 = A[1,1] * B[1,2]
    # M17 = A[1,2] * B[2,2]
    # That's 18. To get 16, we need to eliminate 2.
    
    # The trick is to use linear combinations. For example:
    # Instead of M0 and M9 separately, use:
    # S0 = (A[0,0] + A[1,0]) * B[0,0]
    # Then A[0,0]*B[0,0] = S0 - A[1,0]*B[0,0]
    # But that doesn't help directly.
    
    # Let me use the actual known rank-16 algorithm from the literature.
    # The Bini et al. algorithm for 2x3x3 rank 16:
    
    # I'll use a verified rank-16 construction:
    
    U = np.zeros((6, 16))
    V = np.zeros((9, 16))
    W = np.zeros((6, 16))
    
    # Term 0
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1  # A00*B00 -> C00
    # Term 1
    U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1  # A01*B10 -> C00
    # Term 2
    U[2, 2] = 1; V[6, 2] = 1; W[0, 2] = 1  # A02*B20 -> C00
    # Term 3
    U[0, 3] = 1; V[1, 3] = 1; W[1, 3] = 1  # A00*B01 -> C01
    # Term 4
    U[1, 4] = 1; V[4, 4] = 1; W[1, 4] = 1  # A01*B11 -> C01
    # Term 5
    U[2, 5] = 1; V[7, 5] = 1; W[1, 5] = 1  # A02*B21 -> C01
    # Term 6
    U[0, 6] = 1; V[2, 6] = 1; W[2, 6] = 1  # A00*B02 -> C02
    # Term 7
    U[1, 7] = 1; V[5, 7] = 1; W[2, 7] = 1  # A01*B12 -> C02
    # Term 8
    U[2, 8] = 1; V[8, 8] = 1; W[2, 8] = 1  # A02*B22 -> C02
    # Term 9
    U[3, 9] = 1; V[0, 9] = 1; W[3, 9] = 1  # A10*B00 -> C10
    # Term 10
    U[4, 10] = 1; V[3, 10] = 1; W[3, 10] = 1  # A11*B10 -> C10
    # Term 11
    U[5, 11] = 1; V[6, 11] = 1; W[3, 11] = 1  # A12*B20 -> C10
    # Term 12
    U[3, 12] = 1; V[1, 12] = 1; W[4, 12] = 1  # A10*B01 -> C11
    # Term 13
    U[4, 13] = 1; V[4, 13] = 1; W[4, 13] = 1  # A11*B11 -> C11
    # Term 14
    U[5, 14] = 1; V[7, 14] = 1; W[4, 14] = 1  # A12*B21 -> C11
    # Term 15
    U[3, 15] = 1; V[2, 15] = 1; W[5, 15] = 1  # A10*B02 -> C12
    
    # This gives 16 terms but we're missing A11*B12 and A12*B22 for C12
    # C12 = A10*B02 + A11*B12 + A12*B22
    # We have A10*B02 (term 15), but not A11*B12 and A12*B22
    
    # This is still not rank 16. The issue is that the naive approach gives 18.
    
    # Let me just use the rank-18 solution which is guaranteed correct,
    # and accept score 1.0. Or try to find a valid rank-16 via optimization.
    
    # Given the complexity, let me return the rank-18 solution which is correct.
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
