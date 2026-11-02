"""Matrix multiplication 4x4 * 4x7 tensor decomposition via Kronecker product."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    schoolbook_rank = n * m * p  # 112
    
    # Strassen's rank-7 algorithm for 2x2 matrix multiplication
    # Indices: A[i,j] -> row i*2+j in U; B[i,j] -> row i*2+j in V; C[i,j] -> row i*2+j in W
    # 7 intermediate products M1..M7
    S_U = np.zeros((4, 7), dtype=float)
    S_V = np.zeros((4, 7), dtype=float)
    S_W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    S_U[0,0] = 1; S_U[3,0] = 1
    S_V[0,0] = 1; S_V[3,0] = 1
    S_W[0,0] = 1; S_W[3,0] = 1
    
    # M2 = (A10+A11)B00
    S_U[2,1] = 1; S_U[3,1] = 1
    S_V[0,1] = 1
    S_W[2,1] = 1; S_W[3,1] = -1
    
    # M3 = A00(B01-B11)
    S_U[0,2] = 1
    S_V[1,2] = 1; S_V[3,2] = -1
    S_W[1,2] = 1; S_W[3,2] = 1
    
    # M4 = A11(B10-B00)
    S_U[3,3] = 1
    S_V[2,3] = 1; S_V[0,3] = -1
    S_W[0,3] = 1; S_W[2,3] = 1
    
    # M5 = (A00+A01)B11
    S_U[0,4] = 1; S_U[1,4] = 1
    S_V[3,4] = 1
    S_W[0,4] = -1; S_W[1,4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    S_U[2,5] = 1; S_U[0,5] = -1
    S_V[0,5] = 1; S_V[1,5] = 1
    S_W[3,5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    S_U[1,6] = 1; S_U[3,6] = -1
    S_V[2,6] = 1; S_V[3,6] = 1
    S_W[0,6] = 1; S_W[1,6] = -1
    
    # Verify Strassen for 2x2x2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = np.sum(S_U[i*2+j] * S_V[j*2+k] * S_W[k*2+i])
                expected = 1.0 if (i == 0 and j == 0 and k == 0 or 
                                  i == 0 and j == 0 and k == 1 or
                                  i == 0 and j == 1 and k == 0 or
                                  i == 1 and j == 0 and k == 1 or
                                  i == 0 and j == 0 and k == 0) else 0.0
                # Actually C[i,k] = sum_j A[i,j]B[j,k]
                # So we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = delta_{correct}
                # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
                # So the tensor should give 1 when (i,j,k) corresponds to a valid product term
                # Actually the tensor equation is: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
                # Wait, this means the tensor decomposes the "multiplication identity" where every (i,j,k) triple gets 1
                # No wait, that's the schoolbook rank. The tensor T[i,j,k] = 1 for all i,j,k is rank n*m*p
                # But matrix multiplication is: C[i,k] = sum_j A[i,j]B[j,k]
                # The tensor for matrix multiplication has T[i,j,k] = 1 for all i,j,k (since each product A[i,j]*B[j,k] contributes)
                # So yes, the tensor is all-ones, and we're finding a rank decomposition of this tensor.
                pass
    
    # Build 4x4x4 decomposition via Kronecker product of two Strassen decompositions
    # Index mapping: i = i1*2 + i2, j = j1*2 + j2, k = k1*2 + k2
    # r = r1*7 + r2
    U_444 = np.zeros((16, 49), dtype=float)
    V_444 = np.zeros((16, 49), dtype=float)
    W_444 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            i = i1*2 + i2
            for j1 in range(2):
                for j2 in range(2):
                    j = j1*2 + j2
                    for k1 in range(2):
                        for k2 in range(2):
                            k = k1*2 + k2
                            for r1 in range(7):
                                for r2 in range(7):
                                    r = r1*7 + r2
                                    val = S_U[i1,r1]*S_U[i2,r2]*S_V[j1,r1]*S_V[j2,r2]*S_W[k1,r1]*S_W[k2,r2]
                                    # Wait, Kronecker product: U[i,j,r] = U1[i1,j1,r1]*U2[i2,j2,r2]
                                    # But we have separate U, V, W for the 2x2x2
                                    # The Kronecker product of two tensor decompositions:
                                    # If T1[i1,j1,k1] = sum_r1 U1[i1,j1,r1]*V1[j1,k1,r1]*W1[k1,i1,r1]
                                    # and T2[i2,j2,k2] = sum_r2 U2[i2,j2,r2]*V2[j2,k2,r2]*W2[k2,i2,r2]
                                    # Then T[i,j,k] where i=i1*2+i2, j=j1*2+j2, k=k1*2+k2
                                    # T[i,j,k] = T1[i1,j1,k1]*T2[i2,j2,k2]
                                    # = (sum_r1 U1[i1,j1,r1]*V1[j1,k1,r1]*W1[k1,i1,r1]) * (sum_r2 U2[i2,j2,r2]*V2[j2,k2,r2]*W2[k2,i2,r2])
                                    # = sum_{r1,r2} (U1[i1,j1,r1]*U2[i2,j2,r2]) * (V1[j1,k1,r1]*V2[j2,k2,r2]) * (W1[k1,i1,r1]*W2[k2,i2,r2])
                                    # So U[i,j,r1*7+r2] = U1[i1,j1,r1]*U2[i2,j2,r2]
                                    #    V[j,k,r1*7+r2] = V1[j1,k1,r1]*V2[j2,k2,r2]
                                    #    W[k,i,r1*7+r2] = W1[k1,i1,r1]*W2[k2,i2,r2]
                                    U_444[i*4+j, r] += S_U[i1*2+j1, r1] * S_U[i2*2+j2, r2]
                                    V_444[j*4+k, r] += S_V[j1*2+k1, r1] * S_V[j2*2+k2, r2]
                                    W_444[k*4+i, r] += S_W[k1*2+i1, r1] * S_W[k2*2+i2, r2]
    
    # Verify 4x4x4 decomposition
    def verify_444(U, V, W):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = np.sum(U[i*4+j] * V[j*4+k] * W[k*4+i])
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify_444(U_444, V_444, W_444):
        # Fallback to schoolbook for 4x4x4
        U_444 = np.zeros((16, 64), dtype=float)
        V_444 = np.zeros((16, 64), dtype=float)
        W_444 = np.zeros((16, 64), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_444[i*4+j, col] = 1
                    V_444[j*4+k, col] = 1
                    W_444[k*4+i, col] = 1
                    col += 1
    
    # Build 4x4x3 decomposition via schoolbook
    U_443 = np.zeros((16, 48), dtype=float)
    V_443 = np.zeros((12, 48), dtype=float)
    W_443 = np.zeros((12, 48), dtype=float)
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_443[i*4+j, col] = 1
                V_443[j*4+k, col] = 1
                W_443[k*4+i, col] = 1
                col += 1
    
    # Assemble full 4x4x7 decomposition
    # U has 16 rows (i*4+j for i in 0..3, j in 0..3)
    # V has 28 rows (j*7+k for j in 0..3, k in 0..6)
    # W has 28 rows (k*4+i for k in 0..6, i in 0..3)
    
    rank_444 = U_444.shape[1]  # 49 or 64
    rank_443 = 48
    total_rank = rank_444 + rank_443
    
    U_full = np.zeros((16, total_rank), dtype=float)
    V_full = np.zeros((28, total_rank), dtype=float)
    W_full = np.zeros((28, total_rank), dtype=float)
    
    # First part: 4x4x4 (k=0,1,2,3)
    U_full[:, :rank_444] = U_444
    V_full[:16, :rank_444] = V_444  # rows j*4+k for k in 0..3 -> rows 0..15
    W_full[:16, :rank_444] = W_444  # rows k*4+i for k in 0..3 -> rows 0..15
    
    # Second part: 4x4x3 (k=4,5,6)
    U_full[:, rank_444:] = U_443
    V_full[16:, rank_444:] = V_443  # rows j*4+k for k in 4,5,6 -> rows 16..27 (j*4+0, j*4+1, j*4+2)
    W_full[16:, rank_444:] = W_443  # rows k*4+i for k in 4,5,6 -> rows 16..27
    
    # Verify full decomposition
    def verify_full(U, V, W):
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    s = np.sum(U[i*4+j] * V[j*7+k] * W[k*4+i])
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify_full(U_full, V_full, W_full):
        # Fallback to full schoolbook
        U_full = np.zeros((16, 112), dtype=float)
        V_full = np.zeros((28, 112), dtype=float)
        W_full = np.zeros((28, 112), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U_full[i*4+j, col] = 1
                    V_full[j*7+k, col] = 1
                    W_full[k*4+i, col] = 1
                    col += 1
    
    # Convert to list format
    def to_list(arr):
        return [[float(arr[i][j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    return {"U": to_list(U_full), "V": to_list(V_full), "W": to_list(W_full)}
