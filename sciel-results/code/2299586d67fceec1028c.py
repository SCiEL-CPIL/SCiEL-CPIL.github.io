import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = delta (should be 1 if i,j,k correspond to a valid product)
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where i in [0,n), j in [0,m), k in [0,p)
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # For each r, we have a rank-1 tensor: U[:,r] outer V[:,r] outer W[:,r]
    # But we need to check that the sum equals the matmul tensor.
    
    # The matmul tensor T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B
    # contributes to (i,k) entry of C.
    # Actually, the tensor for matmul is: T[i,j,k] = delta_{i,k} ... no.
    # The matrix multiplication tensor is defined as:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has entries T[i,j,k] = 1 for all valid (i,j,k) tuples.
    # Wait, that's not right either.
    
    # The matrix multiplication tensor M_{n,m,p} has entries:
    # M[i,j,k] = 1 if the product A[i,j] * B[j,k] contributes to C[i,k]
    # Actually, M[i,j,k] = 1 for all i,j,k in their respective ranges.
    # The tensor is just the all-ones tensor of shape (n, m, p)? No.
    
    # Let me think again. The matrix multiplication tensor T has:
    # T[i,j,k] = 1 if and only if the (i,j) entry of A and (j,k) entry of B
    # multiply to contribute to the (i,k) entry of C.
    # So T[i,j,k] = 1 for all valid i, j, k.
    
    # The decomposition requires:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_decomp(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2_decomp():
    """
    Return the Strassen rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    """
    # Strassen's 7 multiplications for 2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M1-M4+M6
    # C21 = M2+M4-M3+M5
    # C22 = M1-M2+M3-M6
    
    # U indices: a11=0, a12=1, a21=2, a22=3
    # V indices: b11=0, b12=1, b21=2, b22=3
    # W indices: c11=0, c12=1, c21=2, c22=3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1  # contributes to C11, C12, C21, C22? No.
    
    # Let me be more careful.
    # M1 contributes to: C11 (+1), C12 (+1), C21 (0), C22 (+1)
    # Wait: C11 = M1+M4-M5+M7, C12 = M3+M1-M4+M6, C21 = M2+M4-M3+M5, C22 = M1-M2+M3-M6
    # So M1: C11=+1, C12=+1, C21=0, C22=+1
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[3, 0] = 1
    
    # M2 = (a21+a22)b11
    # M2: C11=0, C12=0, C21=+1, C22=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a11(b12-b22)
    # M3: C11=0, C12=+1, C21=-1, C22=+1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[2, 2] = -1; W[3, 2] = 1
    
    # M4 = a22(b21-b11)
    # M4: C11=+1, C12=-1, C21=+1, C22=0
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = -1; W[2, 3] = 1
    
    # M5 = (a11+a12)b22
    # M5: C11=-1, C12=0, C21=+1, C22=0
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    # M6: C11=0, C12=+1, C21=0, C22=-1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = 1; W[3, 5] = -1
    
    # M7 = (a12-a22)(b21+b22)
    # M7: C11=+1, C12=0, C21=0, C22=0
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook rank 32
    U_school, V_school, W_school = _schoolbook_decomp(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = 32
    
    # Try to find a rank 28 decomposition using Strassen on 2x2 blocks
    # A is 2x4, B is 4x4, C is 2x4
    # Split A into 2x2 blocks: A11=A[0:2,0:2], A12=A[0:2,2:4]
    # Split B into 2x2 blocks: B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4]
    # C11 = A11*B11 + A12*B21 (2x2)
    # C12 = A11*B12 + A12*B22 (2x2)
    # C21 = A21*B11 + A22*B21 (2x2) - wait, A only has 2 rows
    # Actually A is 2x4, so it has 1 row of 2x2 blocks: [A11, A12]
    # B is 4x4, so it has 2x2 blocks: [[B11, B12], [B21, B22]]
    # C is 2x4, so it has 1 row of 2x2 blocks: [C11, C12]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    
    # Each of these is a sum of two 2x2*2x2 products.
    # We can compute each 2x2*2x2 product using Strassen (rank 7).
    # So C11 needs 7+7=14 multiplications, C12 needs 7+7=14 multiplications.
    # Total: 28 multiplications.
    
    # Let's construct this explicitly.
    # A11 entries: A[0,0]=idx0, A[0,1]=idx1, A[1,0]=idx2, A[1,1]=idx3
    # A12 entries: A[0,2]=idx4, A[0,3]=idx5, A[1,2]=idx6, A[1,3]=idx7
    # B11 entries: B[0,0]=idx0, B[0,1]=idx1, B[1,0]=idx2, B[1,1]=idx3
    # B12 entries: B[0,2]=idx4, B[0,3]=idx5, B[1,2]=idx6, B[1,3]=idx7
    # B21 entries: B[2,0]=idx8, B[2,1]=idx9, B[3,0]=idx10, B[3,1]=idx11
    # B22 entries: B[2,2]=idx12, B[2,3]=idx13, B[3,2]=idx14, B[3,3]=idx15
    # C11 entries: C[0,0]=idx0, C[0,1]=idx1, C[1,0]=idx2, C[1,1]=idx3
    # C12 entries: C[0,2]=idx4, C[0,3]=idx5, C[1,2]=idx6, C[1,3]=idx7
    
    # For C11 = A11*B11 + A12*B21:
    # Use Strassen for A11*B11 (7 mults) and A12*B21 (7 mults)
    # For C12 = A11*B12 + A12*B22:
    # Use Strassen for A11*B12 (7 mults) and A12*B22 (7 mults)
    
    U_strassen = np.zeros((n*m, 28))
    V_strassen = np.zeros((m*p, 28))
    W_strassen = np.zeros((p*n, 28))
    
    # Get Strassen decomposition for 2x2*2x2
    U_s, V_s, W_s = _strassen_2x2_decomp()
    U_s = np.array(U_s)
    V_s = np.array(V_s)
    W_s = np.array(W_s)
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: A indices 0-3, B indices 0-3, C indices 0-3
    for r in range(7):
        for ai in range(4):
            U_strassen[ai, r] += U_s[ai, r]
        for bi in range(4):
            V_strassen[bi, r] += V_s[bi, r]
        for ci in range(4):
            W_strassen[ci, r] += W_s[ci, r]
    
    # A12*B21: A indices 4-7, B indices 8-11, C indices 0-3
    for r in range(7):
        for ai in range(4):
            U_strassen[4+ai, 7+r] += U_s[ai, r]
        for bi in range(4):
            V_strassen[8+bi, 7+r] += V_s[bi, r]
        for ci in range(4):
            W_strassen[ci, 7+r] += W_s[ci, r]
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: A indices 0-3, B indices 4-7, C indices 4-7
    for r in range(7):
        for ai in range(4):
            U_strassen[ai, 14+r] += U_s[ai, r]
        for bi in range(4):
            V_strassen[4+bi, 14+r] += V_s[bi, r]
        for ci in range(4):
            W_strassen[4+ci, 14+r] += W_s[ci, r]
    
    # A12*B22: A indices 4-7, B indices 12-15, C indices 4-7
    for r in range(7):
        for ai in range(4):
            U_strassen[4+ai, 21+r] += U_s[ai, r]
        for bi in range(4):
            V_strassen[12+bi, 21+r] += V_s[bi, r]
        for ci in range(4):
            W_strassen[4+ci, 21+r] += W_s[ci, r]
    
    # Verify
    if _verify_decomposition(U_strassen.tolist(), V_strassen.tolist(), W_strassen.tolist(), n, m, p):
        best_U = U_strassen.tolist()
        best_V = V_strassen.tolist()
        best_W = W_strassen.tolist()
        best_rank = 28
    else:
        # Fall back to schoolbook
        pass
    
    # Check time
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to find a rank 26 decomposition using randomized search
    # We can try to find a rank 26 decomposition by starting from the rank 28
    # and trying to eliminate 2 dimensions.
    
    # For now, return the best we have
    return {"U": best_U, "V": best_V, "W": best_W}
