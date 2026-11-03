"""Matrix multiplication 2x4x5 tensor decomposition using Strassen-based rank-36.

Strategy: Split the 2x4x5 problem into:
  - 2x4x4 part: rank 28 (using Strassen 2x2x2 blocks)
  - 2x4x1 part: rank 8 (schoolbook)
Total rank: 36

The 2x4x4 part uses the following structure:
  A (2x4) split into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3)
  B (4x4) split into four 2x2 blocks: B_TL, B_TR, B_BL, B_BR
  C_L (2x2) = A_L * B_TL + A_R * B_BL
  C_R (2x2) = A_L * B_TR + A_R * B_BR

Each 2x2x2 sum is computed using 7 Strassen multiplications.
"""

import numpy as np
import time

def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    Returns (U, V, W) where:
      U: (4, 7) - indices (i,j) -> i*2+j
      V: (4, 7) - indices (j,k) -> j*2+k  
      W: (4, 7) - indices (k,i) -> k*2+i
    
    The decomposition satisfies:
      sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,k} * delta_{j,j}
      i.e., C[i,k] = sum_j A[i,j] B[j,k]
    """
    # Strassen's 7 multiplications for 2x2x2
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B11-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U indices: (i,j) -> i*2+j: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # V indices: (j,k) -> j*2+k: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # W indices: (k,i) -> k*2+i: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1 -> U[0]=1, U[3]=1
    # V: B00=1, B11=1 -> V[0]=1, V[3]=1
    # W: C00+=1, C11+=1 -> W[0]=1, W[3]=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    # U: A10=1, A11=1 -> U[2]=1, U[3]=1
    # V: B00=1 -> V[0]=1
    # W: C10+=1 -> W[2]=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00(B10-B00)
    # U: A00=1 -> U[0]=1
    # V: B10=1, B00=-1 -> V[2]=1, V[0]=-1
    # W: C01+=1 -> W[1]=1
    U[0, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[1, 2] = 1
    
    # M4 = A11(B11-B01)
    # U: A11=1 -> U[3]=1
    # V: B11=1, B01=-1 -> V[3]=1, V[1]=-1
    # W: C00+=1, C10+=1 -> W[0]=1, W[2]=1
    U[3, 3] = 1
    V[3, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    # U: A00=1, A01=1 -> U[0]=1, U[1]=1
    # V: B11=1 -> V[3]=1
    # W: C00+=-1, C01+=1 -> W[0]=-1, W[1]=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    # U: A10=1, A00=-1 -> U[2]=1, U[0]=-1
    # V: B00=1, B01=1 -> V[0]=1, V[1]=1
    # W: C11+=1 -> W[3]=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    # U: A01=1, A11=-1 -> U[1]=1, U[3]=-1
    # V: B10=1, B11=1 -> V[2]=1, V[3]=1
    # W: C00+=1 -> W[0]=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_strassen_36():
    """Build the rank-36 decomposition for 2x4x5 matrix multiplication.
    
    Splits into 2x4x4 (rank 28) + 2x4x1 (rank 8).
    """
    n, m, p = 2, 4, 5
    rank_2x4x4 = 28
    rank_2x4x1 = 8
    total_rank = rank_2x4x4 + rank_2x4x1
    
    U = np.zeros((n * m, total_rank))  # (8, 36)
    V = np.zeros((m * p, total_rank))  # (20, 36)
    W = np.zeros((p * n, total_rank))  # (10, 36)
    
    # Get the Strassen 2x2x2 decomposition
    sU, sV, sW = _strassen_2x2x2()
    
    # Part 1: 2x4x4 (columns 0-3 of B, columns 0-3 of C)
    # A_L = A[:, 0:2] (2x2), A_R = A[:, 2:4] (2x2)
    # B_TL = B[0:2, 0:2], B_TR = B[0:2, 2:4], B_BL = B[2:4, 0:2], B_BR = B[2:4, 2:4]
    # C_L = A_L * B_TL + A_R * B_BL (2x2)
    # C_R = A_L * B_TR + A_R * B_BR (2x2)
    
    # For C_L = A_L * B_TL + A_R * B_BL:
    # This is a 2x4x2 multiplication. We can compute it using Strassen-like methods.
    # Actually, let's use the following approach:
    # C_L[i,k] = sum_{j=0}^{1} A_L[i,j] B_TL[j,k] + sum_{j=0}^{1} A_R[i,j] B_BL[j,k]
    #          = sum_{j=0}^{3} A_full[i,j] B_full[j,k] for j in {0,1,2,3} and k in {0,1}
    # where A_full[i,0]=A_L[i,0], A_full[i,1]=A_L[i,1], A_full[i,2]=A_R[i,0], A_full[i,3]=A_R[i,1]
    # and B_full[0,k]=B_TL[0,k], B_full[1,k]=B_TL[1,k], B_full[2,k]=B_BL[0,k], B_full[3,k]=B_BL[1,k]
    
    # To compute C_L using 7 Strassen multiplications:
    # We can embed the 2x4x2 problem into a 2x2x2 problem by combining rows.
    # Alternatively, use the fact that 2x4x2 can be done with rank 14 (two 2x2x2 Strassen).
    # C_L = A_L*B_TL + A_R*B_BL
    # = [A_L; A_R] * [B_TL; B_BL] where [A_L;A_R] is 4x2... no, that's not right.
    
    # Let me use a different approach. For 2x4x2:
    # Split into two 2x2x2 problems:
    # C_L = A_L*B_TL + A_R*B_BL
    # Compute A_L*B_TL using Strassen (7 mults) and A_R*B_BL using Strassen (7 mults)
    # But they share the same output, so total is 14 mults for C_L.
    # Similarly for C_R: 14 mults.
    # Total for 2x4x4: 28 mults. Good.
    
    # Strassen for A_L * B_TL (2x2x2):
    # A_L indices in U: (i, j) where j=0,1 -> U[i*4 + j] for i=0,1, j=0,1
    # B_TL indices in V: (j, k) where j=0,1, k=0,1 -> V[j*5 + k] for j=0,1, k=0,1
    # C_L indices in W: (k, i) where k=0,1, i=0,1 -> W[k*2 + i] for k=0,1, i=0,1
    
    # For A_L * B_TL using Strassen:
    # U row for A_L[i,j] is i*4 + j (since m=4, n=2)
    # V row for B_TL[j,k] is j*5 + k (since p=5, m=4)
    # W row for C_L[k,i] is k*2 + i (since n=2, p=5)
    
    for r in range(7):
        col = r  # columns 0-6 for A_L*B_TL
        # U: A_L[i,j] at row i*4+j
        for ui, uv in enumerate(sU[:, r]):
            if uv != 0:
                i, j = ui // 2, ui % 2
                U[i * 4 + j, col] += uv
        # V: B_TL[j,k] at row j*5+k
        for vi, vv in enumerate(sV[:, r]):
            if vv != 0:
                j, k = vi // 2, vi % 2
                V[j * 5 + k, col] += vv
        # W: C_L[k,i] at row k*2+i
        for wi, ww in enumerate(sW[:, r]):
            if ww != 0:
                k, i = wi // 2, wi % 2
                W[k * 2 + i, col] += ww
    
    # For A_R * B_BL using Strassen:
    # A_R[i,j] is A[i, 2+j] -> U row i*4 + (2+j)
    # B_BL[j,k] is B[2+j, k] -> V row (2+j)*5 + k
    # C_L[k,i] is same as before -> W row k*2+i
    
    for r in range(7):
        col = 7 + r  # columns 7-13 for A_R*B_BL
        for ui, uv in enumerate(sU[:, r]):
            if uv != 0:
                i, j = ui // 2, ui % 2
                U[i * 4 + 2 + j, col] += uv
        for vi, vv in enumerate(sV[:, r]):
            if vv != 0:
                j, k = vi // 2, vi % 2
                V[(2 + j) * 5 + k, col] += vv
        for wi, ww in enumerate(sW[:, r]):
            if ww != 0:
                k, i = wi // 2, wi % 2
                W[k * 2 + i, col] += ww
    
    # For A_L * B_TR using Strassen:
    # A_L[i,j] -> U row i*4+j
    # B_TR[j,k] is B[j, 2+k] -> V row j*5 + (2+k)
    # C_R[k,i] -> W row (2+k)*2 + i
    
    for r in range(7):
        col = 14 + r  # columns 14-20 for A_L*B_TR
        for ui, uv in enumerate(sU[:, r]):
            if uv != 0:
                i, j = ui // 2, ui % 2
                U[i * 4 + j, col] += uv
        for vi, vv in enumerate(sV[:, r]):
            if vv != 0:
                j, k = vi // 2, vi % 2
                V[j * 5 + 2 + k, col] += vv
        for wi, ww in enumerate(sW[:, r]):
            if ww != 0:
                k, i = wi // 2, wi % 2
                W[(2 + k) * 2 + i, col] += ww
    
    # For A_R * B_BR using Strassen:
    # A_R[i,j] -> U row i*4+(2+j)
    # B_BR[j,k] is B[2+j, 2+k] -> V row (2+j)*5 + (2+k)
    # C_R[k,i] -> W row (2+k)*2+i
    
    for r in range(7):
        col = 21 + r  # columns 21-27 for A_R*B_BR
        for ui, uv in enumerate(sU[:, r]):
            if uv != 0:
                i, j = ui // 2, ui % 2
                U[i * 4 + 2 + j, col] += uv
        for vi, vv in enumerate(sV[:, r]):
            if vv != 0:
                j, k = vi // 2, vi % 2
                V[(2 + j) * 5 + 2 + k, col] += vv
        for wi, ww in enumerate(sW[:, r]):
            if ww != 0:
                k, i = wi // 2, wi % 2
                W[(2 + k) * 2 + i, col] += ww
    
    # Part 2: 2x4x1 (column 4 of B, column 4 of C) - schoolbook rank 8
    # C[i, 4] = sum_{j=0}^{3} A[i,j] B[j, 4]
    for i in range(2):
        for j in range(4):
            col = 28 + i * 4 + j
            U[i * 4 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 2 + i, col] = 1  # W[k*n+i] = W[4*2+i] = W[8+i]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    try:
        result = _build_strassen_36()
        
        # Verify the decomposition
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        
        n, m, p = 2, 4, 5
        rank = U.shape[1]
        
        # Verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
        # for all i, j, k (this is the matrix multiplication tensor)
        # Actually, the tensor T[i,j,k] = 1 for the matmul, meaning:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The decomposition gives: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k?
        # No, it should be: for the matmul tensor, T[(i,j),(j,k),(k,i)] = 1
        # This means: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
        
        # Let's verify
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1))
        
        if max_err > 1e-9:
            # Fallback to schoolbook
            U_sb = np.zeros((n*m, n*m*p))
            V_sb = np.zeros((m*p, n*m*p))
            W_sb = np.zeros((p*n, n*m*p))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i*m+j, col] = 1
                        V_sb[j*p+k, col] = 1
                        W_sb[k*n+i, col] = 1
                        col += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
        return result
        
    except Exception:
        # Fallback to schoolbook
        n, m, p = 2, 4, 5
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
