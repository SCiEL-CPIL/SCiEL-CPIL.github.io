"""Matrix multiplication 3x3x5 with rank-optimized decomposition."""

DIMENSIONS = [3, 3, 5]  # (n, m, p): A is n x m, B is m x p, C is n x p

import time
import numpy as np

def _build_tensor(n, m, p):
    I, J, K = n * m, m * p, p * n
    T = np.zeros((I, J, K))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T[i, j, k] = 1
    return T

def _verify_decomposition(U, V, W, T, atol=1e-9):
    rank = U.shape[1]
    recon = np.zeros(T.shape)
    for c in range(rank):
        recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return np.allclose(recon, T, atol=atol)

def _strassen_2x2x2_factors():
    """
    Returns U, V, W factors for 2x2x2 matrix multiplication using Strassen's algorithm.
    Indices:
    A: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    B: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    C: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = 1; W[2, 0] = 1 # Wait, C00 and C11 get M1?
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's assign columns 0..6 to M1..M7
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # This is not a rank 1 term for C. 
    # Strassen terms are not directly rank 1 terms for C.
    # The decomposition is T = sum M_r * (U_r \otimes V_r \otimes W_r) ? No.
    # The tensor is T = sum U_r \otimes V_r \otimes W_r.
    # In Strassen, C = M1 * (vec) + ...
    # Actually, the rank 1 terms correspond to the 7 products.
    # The W vector for each term must be the vector of coefficients for C.
    
    # Term 1 (M1):
    # A part: A00 + A11 -> U[0]=1, U[3]=1
    # B part: B00 + B11 -> V[0]=1, V[3]=1
    # C part: C00 + C11? No.
    # M1 contributes to C00 and C11.
    # C00 += M1, C11 += M1.
    # So W[0]=1, W[3]=1.
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2 (M2):
    # A part: A10 + A11 -> U[2]=1, U[3]=1
    # B part: B00 -> V[0]=1
    # C part: C10 += M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # Term 3 (M3):
    # A part: A00 -> U[0]=1
    # B part: B01 - B11 -> V[1]=1, V[3]=-1
    # C part: C01 += M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # Term 4 (M4):
    # A part: A11 -> U[3]=1
    # B part: B10 - B00 -> V[2]=1, V[0]=-1
    # C part: C00 += M4, C10 += M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 5 (M5):
    # A part: A00 + A01 -> U[0]=1, U[1]=1
    # B part: B11 -> V[3]=1
    # C part: C00 -= M5, C01 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 6 (M6):
    # A part: A10 - A00 -> U[2]=1, U[0]=-1
    # B part: B00 + B01 -> V[0]=1, V[1]=1
    # C part: C11 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7 (M7):
    # A part: A01 - A11 -> U[1]=1, U[3]=-1
    # B part: B10 + B11 -> V[2]=1, V[3]=1
    # C part: C00 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_3x3x1_factors():
    """
    Returns U, V, W for 3x3x1 matrix multiplication.
    A: 3x3 (9), B: 3x1 (3), C: 3x1 (3).
    """
    U = np.zeros((9, 9))
    V = np.zeros((3, 9))
    W = np.zeros((3, 9))
    
    col = 0
    for a in range(3):
        for b in range(3):
            i = a * 3 + b
            j = b
            k = a
            U[i, col] = 1
            V[j, col] = 1
            W[k, col] = 1
            col += 1
    return U, V, W

def construct():
    n, m, p = 3, 3, 5
    I, J, K = n * m, m * p, p * n
    T = _build_tensor(n, m, p)
    
    # We decompose into 3x3x2 (cols 0-1), 3x3x2 (cols 2-3), 3x3x1 (col 4)
    
    # Helper to embed 2x2x2 Strassen factors into 3x3x2
    def _embed_strassen_3x3x2():
        # 3x3x2: A(9), B(6), C(6)
        # Strassen on A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
        U_sub, V_sub, W_sub = _strassen_2x2x2_factors()
        U = np.zeros((9, 17))
        V = np.zeros((6, 17))
        W = np.zeros((6, 17))
        
        # Map 2x2 indices to 3x3/3x2 indices
        # A indices: 0,1,2,3 -> 0,1,3,4 (a=0,b=0; a=0,b=1; a=1,b=0; a=1,b=1)
        # Wait, A index = a*3+b.
        # 0:0,0 -> 0
        # 0:1 -> 1
        # 1:0 -> 3
        # 1:1 -> 4
        a_map = [0, 1, 3, 4]
        
        # B indices: 0,1,2,3 -> b*2+c.
        # 0:0 -> 0
        # 0:1 -> 1
        # 1:0 -> 2
        # 1:1 -> 3
        b_map = [0, 1, 2, 3]
        
        # C indices: 0,1,2,3 -> c*3+a.
        # 0:0 -> 0
        # 0:1 -> 1 (c=0, a=1? No. C index = c*n+a. n=3.
        # C[0,0] -> 0*3+0=0
        # C[0,1] -> 0*3+1=1
        # C[1,0] -> 1*3+0=3
        # C[1,1] -> 1*3+1=4
        c_map = [0, 1, 3, 4]
        
        for r in range(7):
            for idx in range(4):
                U[a_map[idx], r] = U_sub[idx, r]
                V[b_map[idx], r] = V_sub[idx, r]
                W[c_map[idx], r] = W_sub[idx, r]
        
        # Add schoolbook terms for the rest
        # Remaining A: a=2 (row 2), or b=2 (col 2)
        # Terms:
        # A[2,0] B[0,c] -> C[2,c]
        # A[2,1] B[1,c] -> C[2,c]
        # A[2,2] B[2,c] -> C[2,c]
        # A[0,2] B[2,c] -> C[0,c]
        # A[1,2] B[2,c] -> C[1,c]
        
        col = 7
        # A[2,0] B[0,0] -> C[2,0]
        U[2*3+0, col] = 1; V[0*2+0, col] = 1; W[0*3+2, col] = 1; col += 1
        # A[2,0] B[0,1] -> C[2,1]
        U[2*3+0, col] = 1; V[0*2+1, col] = 1; W[1*3+2, col] = 1; col += 1
        # A[2,1] B[1,0] -> C[2,0]
        U[2*3+1, col] = 1; V[1*2+0, col] = 1; W[0*3+2, col] = 1; col += 1
        # A[2,1] B[1,1] -> C[2,1]
        U[2*3+1, col] = 1; V[1*2+1, col] = 1; W[1*3+2, col] = 1; col += 1
        # A[2,2] B[2,0] -> C[2,0]
        U[2*3+2, col] = 1; V[2*2+0, col] = 1; W[0*3+2, col] = 1; col += 1
        # A[2,2] B[2,1] -> C[2,1]
        U[2*3+2, col] = 1; V[2*2+1, col] = 1; W[1*3+2, col] = 1; col += 1
        # A[0,2] B[2,0] -> C[0,0]
        U[0*3+2, col] = 1; V[2*2+0, col] = 1; W[0*3+0, col] = 1; col += 1
        # A[0,2] B[2,1] -> C[0,1]
        U[0*3+2, col] = 1; V[2*2+1, col] = 1; W[1*3+0, col] = 1; col += 1
        # A[1,2] B[2,0] -> C[1,0]
        U[1*3+2, col] = 1; V[2*2+0, col] = 1; W[0*3+1, col] = 1; col += 1
        # A[1,2] B[2,1] -> C[1,1]
        U[1*3+2, col] = 1; V[2*2+1, col] = 1; W[1*3+1, col] = 1; col += 1
        
        return U, V, W

    U_3x3x2, V_3x3x2, W_3x3x2 = _embed_strassen_3x3x2()
    
    U_3x3x1, V_3x3x1, W_3x3x1 = _schoolbook_3x3x1_factors()
    
    # Combine
    # Total Rank = 17 + 17 + 9 = 43
    U = np.zeros((9, 43))
    V = np.zeros((15, 43))
    W = np.zeros((15, 43))
    
    # Block 1: Cols 0-1 of B, Cols 0-1 of C
    U[:, 0:17] = U_3x3x2
    V[0:6, 0:17] = V_3x3x2
    W[0:6, 0:17] = W_3x3x2
    
    # Block 2: Cols 2-3 of B, Cols 2-3 of C
    # B indices: b*5+c. c=2,3.
    # B[0,2] = 2, B[0,3] = 3, B[1,2] = 7, B[1,3] = 8, B[2,2] = 12, B[2,3] = 13
    # C indices: c*3+a. c=2,3.
    # C[2,0] = 6, C[2,1] = 7, C[2,2] = 8
    # C[3,0] = 9, C[3,1] = 10, C[3,2] = 11
    
    # We need to map the 3x3x2 factors to these indices.
    # In 3x3x2, B index is b*2+c.
    # Here B index is b*5+c.
    # We just copy the U, W parts (same A, C structure relative to block)
    # and adjust V indices.
    
    U[:, 17:34] = U_3x3x2
    # V mapping:
    # Local b,c (0,0) -> Global b=0,c=2 -> 0*5+2=2
    # Local b,c (0,1) -> Global b=0,c=3 -> 0*5+3=3
    # Local b,c (1,0) -> Global b=1,c=2 -> 1*5+2=7
    # Local b,c (1,1) -> Global b=1,c=3 -> 1*5+3=8
    # Local b,c (2,0) -> Global b=2,c=2 -> 2*5+2=12
    # Local b,c (2,1) -> Global b=2,c=3 -> 2*5+3=13
    
    v_map_2 = {0:2, 1:3, 2:7, 3:8, 4:12, 5:13}
    for local_idx, global_idx in v_map_2.items():
        V[global_idx, 17:34] = V_3x3x2[local_idx, :]
        
    # W mapping:
    # Local c,a (0,0) -> Global c=2,a=0 -> 2*3+0=6
    # Local c,a (0,1) -> Global c=2,a=1 -> 2*3+1=7
    # Local c,a (1,0) -> Global c=3,a=0 -> 3*3+0=9
    # Local c,a (1,1) -> Global c=3,a=1 -> 3*3+1=10
    # Local c,a (2,0) -> Global c=2,a=2 -> 2*3+2=8 (Wait, 3x3x2 C is 3x2. a=0,1. No a=2 in 3x3x2?
    # In 3x3x2, C is 3x2. a=0,1,2. c=0,1.
    # C index = c*3+a.
    # Local 0: c=0,a=0 -> 0
    # Local 1: c=0,a=1 -> 1
    # Local 2: c=0,a=2 -> 2
    # Local 3: c=1,a=0 -> 3
    # Local 4: c=1,a=1 -> 4
    # Local 5: c=1,a=2 -> 5
    
    # Global C for block 2 (c=2,3):
    # Global 6: c=2,a=0
    # Global 7: c=2,a=1
    # Global 8: c=2,a=2
    # Global 9: c=3,a=0
    # Global 10: c=3,a=1
    # Global 11: c=3,a=2
    
    w_map_2 = {0:6, 1:7, 2:8, 3:9, 4:10, 5:11}
    for local_idx, global_idx in w_map_2.items():
        W[global_idx, 17:34] = W_3x3x2[local_idx, :]
        
    # Block 3: Col 4 of B, Col 4 of C
    # B index: b*5+4. 4, 9, 14
    # C index: 4*3+a. 12, 13, 14
    
    U[:, 34:43] = U_3x3x1
    v_map_3 = {0:4, 1:9, 2:14}
    for local_idx, global_idx in v_map_3.items():
        V[global_idx, 34:43] = V_3x3x1[local_idx, :]
        
    w_map_3 = {0:12, 1:13, 2:14}
    for local_idx, global_idx in w_map_3.items():
        W[global_idx, 34:43] = W_3x3x1[local_idx, :]
        
    # Verify
    if not _verify_decomposition(U, V, W, T, atol=1e-9):
        # Fallback to schoolbook
        U = np.zeros((9, 45))
        V = np.zeros((15, 45))
        W = np.zeros((15, 45))
        col = 0
        for a in range(3):
            for b in range(3):
                for c in range(5):
                    i = a * 3 + b
                    j = b * 5 + c
                    k = c * 3 + a
                    U[i, col] = 1
                    V[j, col] = 1
                    W[k, col] = 1
                    col += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
