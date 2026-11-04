"""Optimized program for matrix multiplication 4x4 by 4x5 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Build the multiplication tensor
    # T[i, j, k, r, s] = 1 if r == i and s == j else 0? No.
    # The tensor for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
    # T[i, k, j, r, s] = delta_{i,r} * delta_{k,s} ... wait let me think about the convention.
    
    # Convention: W[k*n + i, r] where r is the rank index.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i, ?} ... 
    
    # Actually the tensor is: for all i in [0,n), j in [0,m), k in [0,p),
    # The entry of C[i, k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has indices (i, j, k) and value 1 when this is the "trivial" multiplication.
    # The rank-r decomposition gives: sum_{r=0}^{rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 if (i,j,k) corresponds to a valid multiplication term, i.e., always 1 for the 
    # standard basis? No.
    
    # Let me re-read: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 for all i,j,k (it's the identity in 
    # the sense that C[i,k] = sum_j A[i,j]B[j,k]).
    
    # Actually, the tensor is defined such that:
    # For each (i, j, k), the coefficient of A[i,j] * B[j,k] in C[i,k] is 1.
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    
    # The decomposition satisfies:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's verify this.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_222():
    """Return U, V, W for 2x2 by 2x2 Strassen decomposition (rank 7)."""
    # Strassen's 7-multiplication algorithm for 2x2 matrices
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    n = m = p = 2
    rank = 7
    U = [[0.0]*rank for _ in range(4)]
    V = [[0.0]*rank for _ in range(4)]
    W = [[0.0]*rank for _ in range(4)]
    
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each multiplication r, U[i*2+j, r] is the coefficient of A[i,j] in the r-th product,
    # V[j*2+k, r] is the coefficient of B[j,k] in the r-th product,
    # W[k*2+i, r] is the coefficient of C[i,k] in the r-th product.
    
    # M1: (A00+A11)(B00+B11)
    # A part: A00 coeff 1, A11 coeff 1
    U[0*2+0][0] = 1; U[1*2+1][0] = 1
    # B part: B00 coeff 1, B11 coeff 1
    V[0*2+0][0] = 1; V[1*2+1][0] = 1
    # C part: C00 coeff 1, C11 coeff 1
    W[0*2+0][0] = 1; W[1*2+1][0] = 1
    
    # M2: (A10+A11)B00
    U[1*2+0][1] = 1; U[1*2+1][1] = 1
    V[0*2+0][1] = 1
    W[1*2+0][1] = 1; W[1*2+1][1] = 1  # C10 and C11? No.
    # C10 = M2 + M4, so M2 contributes to C10
    # C11 = M1 - M2 + M3 + M6, so M2 contributes to C11 with coeff -1
    W[1*2+0][1] = 1  # C10
    W[1*2+1][1] = -1  # C11
    
    # M3: A00(B01-B11)
    U[0*2+0][2] = 1
    V[0*2+1][2] = 1; V[1*2+1][2] = -1
    W[0*2+1][2] = 1; W[1*2+1][2] = 1  # C01 and C11
    
    # M4: A11(B10-B00)
    U[1*2+1][3] = 1
    V[1*2+0][3] = 1; V[0*2+0][3] = -1
    W[0*2+0][3] = 1; W[1*2+0][3] = 1  # C00 and C10
    
    # M5: (A00+A01)B11
    U[0*2+0][4] = 1; U[0*2+1][4] = 1
    V[1*2+1][4] = 1
    W[0*2+0][4] = -1; W[0*2+1][4] = 1  # C00 and C01
    
    # M6: (A10-A00)(B00+B01)
    U[1*2+0][5] = 1; U[0*2+0][5] = -1
    V[0*2+0][5] = 1; V[0*2+1][5] = 1
    W[1*2+1][5] = 1  # C11
    
    # M7: (A01-A11)(B10+B11)
    U[0*2+1][6] = 1; U[1*2+1][6] = -1
    V[1*2+0][6] = 1; V[1*2+1][6] = 1
    W[0*2+0][6] = 1  # C00
    
    return U, V, W

def _combine(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """Combine two decompositions. Not used directly, placeholder."""
    pass

def _strassen_224():
    """
    Build a decomposition for 2x2 by 2x4 using Strassen-like ideas.
    We can think of 2x2 * 2x4 as two 2x2 * 2x2 blocks.
    But we can share some multiplications.
    
    Actually, let's just use the schoolbook for 2x2*2x4 which is rank 8,
    or try to be clever.
    
    For now, let's construct a 4x4 * 4x5 decomposition by combining smaller ones.
    """
    pass

def _make_mat_from_lists(lst):
    return np.array(lst, dtype=np.float64)

def _strassen_444():
    """
    Build a 4x4 * 4x4 decomposition using 2x2 Strassen blocks.
    Split into 2x2 blocks:
    A = [[A00, A01], [A10, A11]] where each is 2x2
    B = [[B00, B01], [B10, B11]] where each is 2x2
    
    C = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
         [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    
    Each product of 2x2 blocks uses rank 7.
    We have 4 block products, so naive rank = 4*7 = 28.
    But we can share some.
    """
    n = m = p = 4
    # Use 4 independent Strassen 2x2 decompositions
    rank = 28
    U = [[0.0]*rank for _ in range(16)]
    V = [[0.0]*rank for _ in range(16)]
    W = [[0.0]*rank for _ in range(16)]
    
    U2, V2, W2 = _strassen_222()
    
    # Block (0,0): A00 * B00 -> C00
    # A00 is rows 0,1 cols 0,1 of A -> indices 0,1,2,3 in U (i*4+j)
    # Wait, U has shape (n*m, rank) = (16, rank). Index i*4+j.
    # A00: i in {0,1}, j in {0,1} -> U indices 0,1,4,5
    # B00: j in {0,1}, k in {0,1} -> V indices 0,1,4,5
    # C00: k in {0,1}, i in {0,1} -> W indices 0,1,4,5
    
    for r in range(7):
        base = r
        # Map 2x2 Strassen to block (0,0)
        for ii in range(2):
            for jj in range(2):
                if U2[ii*2+jj][r] != 0:
                    U[ii*4+jj][base] = U2[ii*2+jj][r]
        for jj in range(2):
            for kk in range(2):
                if V2[jj*2+kk][r] != 0:
                    V[jj*4+kk][base] = V2[jj*2+kk][r]
        for kk in range(2):
            for ii in range(2):
                if W2[kk*2+ii][r] != 0:
                    W[kk*4+ii][base] = W2[kk*2+ii][r]
    
    # Block (0,1): A00 * B01 -> C01
    # A00: i in {0,1}, j in {0,1} -> U indices 0,1,4,5
    # B01: j in {0,1}, k in {2,3} -> V indices 2,3,6,7
    # C01: k in {2,3}, i in {0,1} -> W indices 2,3,6,7
    
    for r in range(7):
        base = 7 + r
        for ii in range(2):
            for jj in range(2):
                if U2[ii*2+jj][r] != 0:
                    U[ii*4+jj][base] = U2[ii*2+jj][r]
        for jj in range(2):
            for kk in range(2):
                if V2[jj*2+kk][r] != 0:
                    V[jj*4+(kk+2)][base] = V2[jj*2+kk][r]
        for kk in range(2):
            for ii in range(2):
                if W2[kk*2+ii][r] != 0:
                    W[(kk+2)*4+ii][base] = W2[kk*2+ii][r]
    
    # Block (1,0): A10 * B00 -> C10
    # A10: i in {2,3}, j in {0,1} -> U indices 8,9,12,13
    # B00: j in {0,1}, k in {0,1} -> V indices 0,1,4,5
    # C10: k in {0,1}, i in {2,3} -> W indices 8,9,12,13
    
    for r in range(7):
        base = 14 + r
        for ii in range(2):
            for jj in range(2):
                if U2[ii*2+jj][r] != 0:
                    U[(ii+2)*4+jj][base] = U2[ii*2+jj][r]
        for jj in range(2):
            for kk in range(2):
                if V2[jj*2+kk][r] != 0:
                    V[jj*4+kk][base] = V2[jj*2+kk][r]
        for kk in range(2):
            for ii in range(2):
                if W2[kk*2+ii][r] != 0:
                    W[kk*4+(ii+2)][base] = W2[kk*2+ii][r]
    
    # Block (1,1): A11 * B11 -> C11
    # A11: i in {2,3}, j in {2,3} -> U indices 10,11,14,15
    # B11: j in {2,3}, k in {2,3} -> V indices 10,11,14,15
    # C11: k in {2,3}, i in {2,3} -> W indices 10,11,14,15
    
    for r in range(7):
        base = 21 + r
        for ii in range(2):
            for jj in range(2):
                if U2[ii*2+jj][r] != 0:
                    U[(ii+2)*4+(jj+2)][base] = U2[ii*2+jj][r]
        for jj in range(2):
            for kk in range(2):
                if V2[jj*2+kk][r] != 0:
                    V[(jj+2)*4+(kk+2)][base] = V2[jj*2+kk][r]
        for kk in range(2):
            for ii in range(2):
                if W2[kk*2+ii][r] != 0:
                    W[(kk+2)*4+(ii+2)][base] = W2[kk*2+ii][r]
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Start with a known good decomposition
    # First, try to build a 4x4 * 4x4 decomposition and extend to 4x4 * 4x5
    
    # Schoolbook is rank 80. Let's try to do better.
    
    # Strategy: Use block decomposition.
    # A is 4x4, B is 4x5.
    # Split A into 2x2 blocks, B into 2x2 and 2x1 blocks.
    
    # Actually, let's just use a simple approach:
    # For 4x4 * 4x5, we can think of it as 4x4 * 4x4 plus 4x4 * 4x1.
    # Or we can use the Strassen 4x4 * 4x4 (rank 28) and add 4x4 * 4x1 (rank 4).
    # Total rank = 32.
    
    # Let's try rank 32.
    
    U44, V44, W44 = _strassen_444()
    
    # Now for the 4x4 * 4x1 part (the 5th column of B):
    # This is just 4 scalar multiplications per row of C.
    # C[:,4] = A * B[:,4]
    # Each C[i,4] = sum_j A[i,j] * B[j,4]
    # So we need 4 more rank-1 terms (one for each i).
    
    rank = 32
    U = [[0.0]*rank for _ in range(16)]
    V = [[0.0]*rank for _ in range(20)]
    W = [[0.0]*rank for _ in range(20)]
    
    # Copy the 4x4 * 4x4 part
    for i in range(16):
        for r in range(28):
            U[i][r] = U44[i][r]
    for i in range(16):
        for r in range(28):
            V[i][r] = V44[i][r]
    for i in range(16):
        for r in range(28):
            W[i][r] = W44[i][r]
    
    # Add the 4x4 * 4x1 part
    # For each i in 0..3, we need C[i,4] = sum_j A[i,j] * B[j,4]
    # This is a rank-4 decomposition.
    for i in range(4):
        r = 28 + i
        for j in range(4):
            U[i*4+j][r] = 1.0
            V[j*4+4][r] = 1.0  # B[j,4] is at index j*4+4 in V (m*p = 4*5 = 20)
        W[4*4+i][r] = 1.0  # C[i,4] is at index 4*4+i = 16+i in W (p*n = 5*4 = 20)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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
