"""Optimized tensor rank decomposition for 3x4 * 4x4 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]
N, M, P = 3, 4, 4
RANK_SCHOOLBOOK = N * M * P  # 48

# The multiplication tensor T has indices (i, j, k) with i in [0,n), j in [0,m), k in [0,p)
# T[i*m+j, j*p+k, k*n+i] = 1
# We want rank-r decomposition: U (n*m x r), V (m*p x r), W (p*n x r) such that
# T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

def verify_decomposition(U, V, W, n=N, m=M, p=P):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    r = U.shape[1]
    # Check dimensions
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    # Check all entries of the tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def verify_decomposition_fast(U, V, W, n=N, m=M, p=P):
    """Faster verification using tensor reshaping."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    r = U.shape[1]
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    
    # Compute the reconstructed tensor
    # T_recon[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    # Reconstruct: for each (i,j,k), sum over c
    # T_recon[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # Expected tensor: T[i,j,k] = 1 for all valid i,j,k
    expected = np.ones((n, m, p))
    
    return np.allclose(recon, expected, atol=1e-9)

def schoolbook_decomposition():
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = N * M * P
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i*M+j][col] = 1
                V[j*P+k][col] = 1
                W[k*N+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def strassen_2x2_2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    
    The Strassen algorithm computes C = A * B where A, B are 2x2:
    C = [[a11*b11 + a12*b21, a11*b12 + a12*b22],
         [a21*b11 + a22*b21, a21*b12 + a22*b22]]
    
    Using 7 multiplications:
    M1 = (a11+a22)(b11+b22)
    M2 = (a21+a22)b11
    M3 = a11(b12-b22)
    M4 = a22(b21-b11)
    M5 = (a11+a12)b22
    M6 = (a21-a11)(b11+b12)
    M7 = (a12-a22)(b21+b22)
    
    C11 = M1 - M4 + M6 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M3 + M2 + M5
    """
    # For 2x2*2x2, indices: i in [0,1], j in [0,1], k in [0,1]
    # T[i*2+j, j*2+k, k*2+i] = 1
    
    # U has shape (2*2, 7) = (4, 7)
    # V has shape (2*2, 7) = (4, 7)  
    # W has shape (2*2, 7) = (4, 7)
    
    # Each Strassen term M_c corresponds to one column of U, V, W
    # M_c = (sum_i alpha_i * A[i]) * (sum_j beta_j * B[j])
    # where the product contributes to specific C entries
    
    # Let's construct this carefully.
    # For each of the 7 Strassen products, we need:
    # - U factor: which row/col of A contributes (with what sign)
    # - V factor: which row/col of B contributes (with what sign)  
    # - W factor: which entry of C this contributes to (with what sign)
    
    # Strassen's 7 products:
    # M1 = (a11+a22)*(b11+b22)
    #   A part: a11 (i=0,j=0) + a22 (i=1,j=1)
    #   B part: b11 (j=0,k=0) + b22 (j=1,k=1)
    #   C part: contributes to C11 (+1), C22 (+1)
    
    # M2 = (a21+a22)*b11
    #   A part: a21 (i=1,j=0) + a22 (i=1,j=1)
    #   B part: b11 (j=0,k=0)
    #   C part: contributes to C21 (+1), C22 (+1)
    
    # M3 = a11*(b12-b22)
    #   A part: a11 (i=0,j=0)
    #   B part: b12 (j=0,k=1) - b22 (j=1,k=1)
    #   C part: contributes to C12 (+1), C22 (-1)
    
    # M4 = a22*(b21-b11)
    #   A part: a22 (i=1,j=1)
    #   B part: b21 (j=1,k=0) - b11 (j=0,k=0)
    #   C part: contributes to C11 (-1), C21 (+1)
    
    # M5 = (a11+a12)*b22
    #   A part: a11 (i=0,j=0) + a12 (i=0,j=1)
    #   B part: b22 (j=1,k=1)
    #   C part: contributes to C12 (+1), C22 (+1)
    
    # M6 = (a21-a11)*(b11+b12)
    #   A part: a21 (i=1,j=0) - a11 (i=0,j=0)
    #   B part: b11 (j=0,k=0) + b12 (j=0,k=1)
    #   C part: contributes to C11 (+1), C21 (+1)
    
    # M7 = (a12-a22)*(b21+b22)
    #   A part: a12 (i=0,j=1) - a22 (i=1,j=1)
    #   B part: b21 (j=1,k=0) + b22 (j=1,k=1)
    #   C part: contributes to C11 (+1), C22 (+1)
    
    # Now build U, V, W for 2x2*2x2
    # U[i*2+j, c] = coefficient of A[i,j] in product c
    # V[j*2+k, c] = coefficient of B[j,k] in product c
    # W[k*2+i, c] = coefficient for C[i,k] contribution from product c
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a11+a22)(b11+b22)
    U[0*2+0, 0] = 1  # a11
    U[1*2+1, 0] = 1  # a22
    V[0*2+0, 0] = 1  # b11
    V[1*2+1, 0] = 1  # b22
    W[0*2+0, 0] = 1  # C11
    W[1*2+1, 0] = 1  # C22
    
    # M2: (a21+a22)b11
    U[1*2+0, 1] = 1  # a21
    U[1*2+1, 1] = 1  # a22
    V[0*2+0, 1] = 1  # b11
    W[0*2+1, 1] = 1  # C21
    W[1*2+1, 1] = 1  # C22
    
    # M3: a11(b12-b22)
    U[0*2+0, 2] = 1  # a11
    V[0*2+1, 2] = 1  # b12
    V[1*2+1, 2] = -1  # -b22
    W[1*2+0, 2] = 1  # C12
    W[1*2+1, 2] = -1  # -C22
    
    # M4: a22(b21-b11)
    U[1*2+1, 3] = 1  # a22
    V[1*2+0, 3] = 1  # b21
    V[0*2+0, 3] = -1  # -b11
    W[0*2+0, 3] = -1  # -C11
    W[0*2+1, 3] = 1  # C21
    
    # M5: (a11+a12)b22
    U[0*2+0, 4] = 1  # a11
    U[0*2+1, 4] = 1  # a12
    V[1*2+1, 4] = 1  # b22
    W[1*2+0, 4] = 1  # C12
    W[1*2+1, 4] = 1  # C22
    
    # M6: (a21-a11)(b11+b12)
    U[1*2+0, 5] = 1  # a21
    U[0*2+0, 5] = -1  # -a11
    V[0*2+0, 5] = 1  # b11
    V[0*2+1, 5] = 1  # b12
    W[0*2+0, 5] = 1  # C11
    W[0*2+1, 5] = 1  # C21
    
    # M7: (a12-a22)(b21+b22)
    U[0*2+1, 6] = 1  # a12
    U[1*2+1, 6] = -1  # -a22
    V[1*2+0, 6] = 1  # b21
    V[1*2+1, 6] = 1  # b22
    W[0*2+0, 6] = 1  # C11
    W[1*2+1, 6] = 1  # C22
    
    return U, V, W

def block_decomposition_3x4_4x4():
    """
    Block decomposition for 3x4 * 4x4 using Strassen for 2x2*2x2 blocks.
    
    Strategy:
    - Split A (3x4) into: A_top (2x4) and A_bottom (1x4)
    - Split B (4x4) into: B_left (4x2) and B_right (4x2)
    - C (3x4) = C_top (2x4) + C_bottom (1x4)
    
    For C_top (2x4) = A_top (2x4) * B (4x4):
      Split further: A_top = [A_tl (2x2) A_tr (2x2)], B = [B_tl (2x2); B_bl (2x2)] for left half
      C_tl (2x2) = A_tl * B_tl + A_tr * B_bl  (two 2x2*2x2 multiplications)
      Similarly for right half
      
    Actually, let me use a cleaner blocking:
    
    A (3x4) = [[A00 (2x2), A01 (2x2)],
               [A10 (1x2), A11 (1x2)]]
    
    B (4x4) = [[B00 (2x2), B01 (2x2)],
               [B10 (2x2), B11 (2x2)]]
    
    C (3x4) = [[C00 (2x2), C01 (2x2)],
               [C10 (1x2), C11 (1x2)]]
    
    C00 = A00*B00 + A01*B10  (two 2x2*2x2 mults, each rank 7 -> rank 14)
    C01 = A00*B01 + A01*B11  (two 2x2*2x2 mults, each rank 7 -> rank 14)
    C10 = A10*B00 + A11*B10  (two 1x2*2x2 mults, each rank 2 -> rank 4)
    C11 = A10*B01 + A11*B11  (two 1x2*2x2 mults, each rank 2 -> rank 4)
    
    Total rank: 14 + 14 + 4 + 4 = 36
    """
    # We need to construct U, V, W for the full 3x4*4x4 problem
    # using the block structure above.
    
    # The full problem has:
    # U: shape (3*4, rank) = (12, rank)
    # V: shape (4*4, rank) = (16, rank)
    # W: shape (4*3, rank) = (12, rank)
    
    # Index conventions:
    # U[i*4+j, c] for i in [0,3), j in [0,4)  (A is 3x4, so i=row, j=col)
    # V[j*4+k, c] for j in [0,4), k in [0,4)  (B is 4x4, so j=row, k=col)
    # W[k*3+i, c] for k in [0,4), i in [0,3)  (C is 3x4, so k=col, i=row)
    
    # Block indices in original coordinates:
    # A00: rows 0-1, cols 0-1 -> U indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # A01: rows 0-1, cols 2-3 -> U indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # A10: row 2, cols 0-1 -> U indices: 2*4+0=8, 2*4+1=9
    # A11: row 2, cols 2-3 -> U indices: 2*4+2=10, 2*4+3=11
    
    # B00: rows 0-1, cols 0-1 -> V indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # B01: rows 0-1, cols 2-3 -> V indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # B10: rows 2-3, cols 0-1 -> V indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # B11: rows 2-3, cols 2-3 -> V indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    
    # C00: rows 0-1, cols 0-1 -> W indices: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
    # C01: rows 0-1, cols 2-3 -> W indices: 2*3+0=6, 2*3+1=7, 3*3+0=9, 3*3+1=10
    # C10: row 2, cols 0-1 -> W indices: 0*3+2=2, 1*3+2=5
    # C11: row 2, cols 2-3 -> W indices: 2*3+2=8, 3*3+2=11
    
    # Get Strassen decomposition for 2x2*2x2
    S_U, S_V, S_W = strassen_2x2_2x2()
    # S_U: (4, 7), S_V: (4, 7), S_W: (4, 7)
    # For 2x2*2x2: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # For 1x2*2x2, we need a simple rank-2 decomposition
    def one_by_two_by_two():
        """Rank-2 decomposition for 1x2 * 2x2."""
        # C (1x2) = A (1x2) * B (2x2)
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # Two products:
        # P1 = A[0,0] * (B[0,0] for C[0,0], B[0,1] for C[0,1])
        # P2 = A[0,1] * (B[1,0] for C[0,0], B[1,1] for C[0,1])
        
        # For 1x2*2x2:
        # U: shape (1*2, 2) = (2, 2), U[i*2+j, c] with i=0, j in [0,1]
        # V: shape (2*2, 2) = (4, 2), V[j*2+k, c] with j in [0,1], k in [0,1]
        # W: shape (2*1, 2) = (2, 2), W[k*1+i, c] with k in [0,1], i=0
        
        U = np.zeros((2, 2))
        V = np.zeros((4, 2))
        W = np.zeros((2, 2))
        
        # P1: A[0,0] * B[:,0] contributes to C[0,0]
        # A[0,0] is at U[0*2+0] = U[0]
        # B[0,0] is at V[0*2+0] = V[0], B[1,0] is at V[1*2+0] = V[2]
        # C[0,0] is at W[0*1+0] = W[0]
        U[0, 0] = 1
        V[0, 0] = 1  # B[0,0]
        V[2, 0] = 1  # B[1,0]
        W[0, 0] = 1  # C[0,0]
        
        # P2: A[0,1] * B[:,1] contributes to C[0,1]
        # A[0,1] is at U[0*2+1] = U[1]
        # B[0,1] is at V[0*2+1] = V[1], B[1,1] is at V[1*2+1] = V[3]
        # C[0,1] is at W[1*1+0] = W[1]
        U[1, 1] = 1
        V[1, 1] = 1  # B[0,1]
        V[3, 1] = 1  # B[1,1]
        W[1, 1] = 1  # C[0,1]
        
        return U, V, W
    
    O_U, O_V, O_W = one_by_two_by_two()
    # O_U: (2, 2), O_V: (4, 2), O_W: (2, 2)
    
    # Now assemble the full decomposition.
    # Total rank = 4 * 7 + 4 * 2 = 28 + 8 = 36
    total_rank = 36
    U_full = np.zeros((12, total_rank))
    V_full = np.zeros((16, total_rank))
    W_full = np.zeros((12, total_rank))
    
    col_offset = 0
    
    # Block C00 = A00*B00 + A01*B10
    # A00*B00: 2x2*2x2 using Strassen
    # A00 indices in U: [0,1,4,5] corresponding to local [0,1,2,3]
    # B00 indices in V: [0,1,4,5] corresponding to local [0,1,2,3]
    # C00 indices in W: [0,1,3,4] corresponding to local [0,1,2,3]
    
    # Map local 2x2 indices to global indices
    # Local (i,j) for A00 -> global U index i*4+j where i in [0,1], j in [0,1]
    # Local (j,k) for B00 -> global V index j*4+k where j in [0,1], k in [0,1]
    # Local (k,i) for C00 -> global W index k*3+i where k in [0,1], i in [0,1]
    
    # For Strassen, local indices:
    # A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    # A00*B00:
    for c in range(7):
        for local_i in range(2):
            for local_j in range(2):
                u_idx = local_i * 4 + local_j
                if S_U[local_i*2+local_j, c] != 0:
                    U_full[u_idx, col_offset+c] = S_U[local_i*2+local_j, c]
        for local_j in range(2):
            for local_k in range(2):
                v_idx = local_j * 4 + local_k
                if S_V[local_j*2+local_k, c] != 0:
                    V_full[v_idx, col_offset+c] = S_V[local_j*2+local_k, c]
        for local_k in range(2):
            for local_i in range(2):
                w_idx = local_k * 3 + local_i
                if S_W[local_k*2+local_i, c] != 0:
                    W_full[w_idx, col_offset+c] = S_W[local_k*2+local_i, c]
    col_offset += 7
    
    # A01*B10:
    # A01: rows 0-1, cols 2-3 -> U indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # B10: rows 2-3, cols 0-1 -> V indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # C00: rows 0-1, cols 0-1 -> W indices: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
    
    for c in range(7):
        for local_i in range(2):
            for local_j in range(2):
                # A01: local (i,j) -> global U index i*4 + (2+j)
                u_idx = local_i * 4 + (2 + local_j)
                if S_U[local_i*2+local_j, c] != 0:
                    U_full[u_idx, col_offset+c] = S_U[local_i*2+local_j, c]
        for local_j in range(2):
            for local_k in range(2):
                # B10: local (j,k) -> global V index (2+j)*4 + k
                v_idx = (2 + local_j) * 4 + local_k
                if S_V[local_j*2+local_k, c] != 0:
                    V_full[v_idx, col_offset+c] = S_V[local_j*2+local_k, c]
        for local_k in range(2):
            for local_i in range(2):
                # C00: local (k,i) -> global W index k*3 + i
                w_idx = local_k * 3 + local_i
                if S_W[local_k*2+local_i, c] != 0:
                    W_full[w_idx, col_offset+c] = S_W[local_k*2+local_i, c]
    col_offset += 7
    
    # Block C01 = A00*B01 + A01*B11
    # A00*B01:
    # A00: U indices [0,1,4,5]
    # B01: rows 0-1, cols 2-3 -> V indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # C01: rows 0-1, cols 2-3 -> W indices: 2*3+0=6, 2*3+1=7, 3*3+0=9, 3*3+1=10
    
    for c in range(7):
        for local_i in range(2):
            for local_j in range(2):
                u_idx = local_i * 4 + local_j
                if S_U[local_i*2+local_j, c] != 0:
                    U_full[u_idx, col_offset+c] = S_U[local_i*2+local_j, c]
        for local_j in range(2):
            for local_k in range(2):
                # B01: local (j,k) -> global V index j*4 + (2+k)
                v_idx = local_j * 4 + (2 + local_k)
                if S_V[local_j*2+local_k, c] != 0:
                    V_full[v_idx, col_offset+c] = S_V[local_j*2+local_k, c]
        for local_k in range(2):
            for local_i in range(2):
                # C01: local (k,i) -> global W index (2+k)*3 + i
                w_idx = (2 + local_k) * 3 + local_i
                if S_W[local_k*2+local_i, c] != 0:
                    W_full[w_idx, col_offset+c] = S_W[local_k*2+local_i, c]
    col_offset += 7
    
    # A01*B11:
    # A01: U indices [2,3,6,7]
    # B11: rows 2-3, cols 2-3 -> V indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    # C01: W indices [6,7,9,10]
    
    for c in range(7):
        for local_i in range(2):
            for local_j in range(2):
                u_idx = local_i * 4 + (2 + local_j)
                if S_U[local_i*2+local_j, c] != 0:
                    U_full[u_idx, col_offset+c] = S_U[local_i*2+local_j, c]
        for local_j in range(2):
            for local_k in range(2):
                v_idx = (2 + local_j) * 4 + (2 + local_k)
                if S_V[local_j*2+local_k, c] != 0:
                    V_full[v_idx, col_offset+c] = S_V[local_j*2+local_k, c]
        for local_k in range(2):
            for local_i in range(2):
                w_idx = (2 + local_k) * 3 + local_i
                if S_W[local_k*2+local_i, c] != 0:
                    W_full[w_idx, col_offset+c] = S_W[local_k*2+local_i, c]
    col_offset += 7
    
    # Block C10 = A10*B00 + A11*B10
    # A10: row 2, cols 0-1 -> U indices: 2*4+0=8, 2*4+1=9
    # B00: V indices [0,1,4,5]
    # C10: row 2, cols 0-1 -> W indices: 0*3+2=2, 1*3+2=5
    
    for c in range(2):
        for local_j in range(2):
            # A10: local j -> global U index 2*4 + j = 8+j
            u_idx = 2 * 4 + local_j
            if O_U[local_j, c] != 0:
                U_full[u_idx, col_offset+c] = O_U[local_j, c]
        for local_j in range(2):
            for local_k in range(2):
                v_idx = local_j * 4 + local_k
                if O_V[local_j*2+local_k, c] != 0:
                    V_full[v_idx, col_offset+c] = O_V[local_j*2+local_k, c]
        for local_k in range(2):
            # C10: local k -> global W index k*3 + 2
            w_idx = local_k * 3 + 2
            if O_W[local_k, c] != 0:
                W_full[w_idx, col_offset+c] = O_W[local_k, c]
    col_offset += 2
    
    # A11*B10:
    # A11: row 2, cols 2-3 -> U indices: 2*4+2=10, 2*4+3=11
    # B10: V indices [8,9,12,13]
    # C10: W indices [2,5]
    
    for c in range(2):
        for local_j in range(2):
            u_idx = 2 * 4 + (2 + local_j)
            if O_U[local_j, c] != 0:
                U_full[u_idx, col_offset+c] = O_U[local_j, c]
        for local_j in range(2):
            for local_k in range(2):
                v_idx = (2 + local_j) * 4 + local_k
                if O_V[local_j*2+local_k, c] != 0:
                    V_full[v_idx, col_offset+c] = O_V[local_j*2+local_k, c]
        for local_k in range(2):
            w_idx = local_k * 3 + 2
            if O_W[local_k, c] != 0:
                W_full[w_idx, col_offset+c] = O_W[local_k, c]
    col_offset += 2
    
    # Block C11 = A10*B01 + A11*B11
    # A10: U indices [8,9]
    # B01: V indices [2,3,6,7]
    # C11: row 2, cols 2-3 -> W indices: 2*3+2=8, 3*3+2=11
    
    for c in range(2):
        for local_j in range(2):
            u_idx = 2 * 4 + local_j
            if O_U[local_j, c] != 0:
                U_full[u_idx, col_offset+c] = O_U[local_j, c]
        for local_j in range(2):
            for local_k in range(2):
                v_idx = local_j * 4 + (2 + local_k)
                if O_V[local_j*2+local_k, c] != 0:
                    V_full[v_idx, col_offset+c] = O_V[local_j*2+local_k, c]
        for local_k in range(2):
            w_idx = (2 + local_k) * 3 + 2
            if O_W[local_k, c] != 0:
                W_full[w_idx, col_offset+c] = O_W[local_k, c]
    col_offset += 2
    
    # A11*B11:
    # A11: U indices [10,11]
    # B11: V indices [10,11,14,15]
    # C11: W indices [8,11]
    
    for c in range(2):
        for local_j in range(2):
            u_idx = 2 * 4 + (2 + local_j)
            if O_U[local_j, c] != 0:
                U_full[u_idx, col_offset+c] = O_U[local_j, c]
        for local_j in range(2):
            for local_k in range(2):
                v_idx = (2 + local_j) * 4 + (2 + local_k)
                if O_V[local_j*2+local_k, c] != 0:
                    V_full[v_idx, col_offset+c] = O_V[local_j*2+local_k, c]
        for local_k in range(2):
            w_idx = (2 + local_k) * 3 + 2
            if O_W[local_k, c] != 0:
                W_full[w_idx, col_offset+c] = O_W[local_k, c]
    col_offset += 2
    
    # Convert to lists of lists (or keep as numpy arrays - the interface accepts both)
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # Try the block decomposition with Strassen
    try:
        result = block_decomposition_3x4_4x4()
        
        # Verify the decomposition
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        
        if verify_decomposition_fast(U, V, W):
            return result
        else:
            # Fallback to schoolbook
            return schoolbook_decomposition()
    except Exception:
        # Fallback to schoolbook
        return schoolbook_decomposition()

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # Try the block decomposition with Strassen
    try:
        result = block_decomposition_3x4_4x4()
        
        # Verify the decomposition
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        
        if verify_decomposition_fast(U, V, W):
            return result
        else:
            # Fallback to schoolbook
            return schoolbook_decomposition()
    except Exception:
        # Fallback to schoolbook
        return schoolbook_decomposition()
# EVOLVE-BLOCK-END
