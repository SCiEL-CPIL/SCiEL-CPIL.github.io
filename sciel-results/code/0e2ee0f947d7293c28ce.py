"""Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
DIMENSIONS = [4, 4, 5]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing a rank-r decomposition
    of the matrix multiplication tensor for 4x4 * 4x5.
    
    The decomposition satisfies:
      For all i in [0,n), j in [0,m), k in [0,p):
        sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    where n=4, m=4, p=5.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Strategy: Use recursive Strassen (2x2 rank 7) for the 4x4 * 4x4 part
    # and schoolbook for the 4x4 * 4x1 part.
    
    # First, define the Strassen 2x2x2 decomposition.
    # T[i,j,k] = 1 if i==k else 0 for i,j,k in {0,1}
    # Rank 7 decomposition:
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # r=0: (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1
    
    # r=1: (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[2,1]=1; W2[3,1]=-1
    
    # r=2: A00(B01-B11)
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[1,2]=1; W2[3,2]=1
    
    # r=3: A11(B10-B00)
    U2[3,3]=1
    V2[2,3]=1; V2[0,3]=-1
    W2[0,3]=1; W2[2,3]=1
    
    # r=4: (A00+A01)B11
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[1,4]=1
    
    # r=5: (A10-A00)(B00+B01)
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1
    
    # r=6: (A01-A11)(B10+B11)
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1
    
    # Verify 2x2x2 Strassen
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0
                for r in range(7):
                    val += U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
                expected = 1 if i == k else 0
                if val != expected:
                    # Fallback to schoolbook if Strassen verification fails
                    rank_schoolbook = n * m * p
                    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
                    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
                    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
                    col = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                U_school[i * m + j][col] = 1
                                V_school[j * p + k][col] = 1
                                W_school[k * n + i][col] = 1
                                col += 1
                    return {"U": U_school, "V": V_school, "W": W_school}
    
    # Now build the 4x4 * 4x4 part using recursive Strassen.
    # We view 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # The 4x4x4 multiplication tensor can be decomposed using:
    # T4[i,j,k] = sum_{s=0}^{6} sum_{t=0}^{6} U4[i,j,s*7+t] * V4[j,k,s*7+t] * W4[k,i,s*7+t]
    # where the coefficients come from the block-level Strassen structure.
    
    # For the 4x4 * 4x4 part (k in 0..3), we use rank 7*7 = 49.
    # For the 4x4 * 4x1 part (k=4), we use schoolbook rank 16.
    # Total rank: 49 + 16 = 65.
    
    rank_44 = 49
    rank_41 = 16
    total_rank = rank_44 + rank_41
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Build the 4x4 * 4x4 part using recursive Strassen.
    # The block-level Strassen coefficients for 2x2 block matrices:
    # s=0: L = A00+A11, R = B00+B11
    # s=1: L = A10+A11, R = B00
    # s=2: L = A00, R = B01-B11
    # s=3: L = A11, R = B10-B00
    # s=4: L = A00+A01, R = B11
    # s=5: L = A10-A00, R = B00+B01
    # s=6: L = A01-A11, R = B10+B11
    
    # For each s, we have block coefficients. Let's define them as 2x2 matrices.
    L_coeffs = [
        [[1, 0], [0, 1]],   # s=0: A00+A11
        [[0, 0], [1, 1]],   # s=1: A10+A11
        [[1, 0], [0, 0]],   # s=2: A00
        [[0, 0], [0, 1]],   # s=3: A11
        [[1, 1], [0, 0]],   # s=4: A00+A01
        [[-1, 0], [1, 0]],  # s=5: A10-A00
        [[0, 1], [0, -1]],  # s=6: A01-A11
    ]
    R_coeffs = [
        [[1, 0], [0, 1]],   # s=0: B00+B11
        [[1, 0], [0, 0]],   # s=1: B00
        [[0, 1], [0, -1]],  # s=2: B01-B11
        [[-1, 0], [1, 0]],  # s=3: B10-B00
        [[0, 0], [0, 1]],   # s=4: B11
        [[1, 1], [0, 0]],   # s=5: B00+B01
        [[0, 1], [1, 1]],   # s=6: B10+B11
    ]
    
    col = 0
    for s in range(7):
        for t in range(7):
            # For each (s, t), we add a rank-1 term.
            # U4[i,j] = sum_{I,J} L_coeffs[s][I,J] * U2[i0*2+j0, t] where i=2*I+i0, j=2*J+j0
            # V4[j,k] = sum_{J,K} R_coeffs[s][J,K] * V2[j0*2+k0, t] where j=2*J+j0, k=2*K+k0
            # W4[k,i] = W2[k0*2+i0, t] where k=2*K+k0, i=2*I+i0
            
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        for i0 in range(2):
                            for j0 in range(2):
                                for k0 in range(2):
                                    i = 2*I + i0
                                    j = 2*J + j0
                                    k = 2*K + k0
                                    
                                    # U contribution
                                    u_val = L_coeffs[s][I][J] * U2[i0*2+j0, t]
                                    if u_val != 0:
                                        U[i*m+j][col] += u_val
                                    
                                    # V contribution
                                    v_val = R_coeffs[s][J][K] * V2[j0*2+k0, t]
                                    if v_val != 0:
                                        V[j*p+k][col] += v_val
                                    
                                    # W contribution
                                    w_val = W2[k0*2+i0, t]
                                    if w_val != 0:
                                        W[k*n+i][col] += w_val
            
            col += 1
    
    # Now handle k = 4 (the 5th column) with schoolbook
    for i in range(n):
        for j in range(m):
            if col >= total_rank:
                break
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    U_reshaped = U_np.reshape(n, m, -1)
    V_reshaped = V_np.reshape(m, p, -1)
    W_reshaped = W_np.reshape(p, n, -1)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_school = [[0] * rank_schoolbook for _ in range(n * m)]
        V_school = [[0] * rank_schoolbook for _ in range(m * p)]
        W_school = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
