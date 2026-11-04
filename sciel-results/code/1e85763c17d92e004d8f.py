import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_222():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # Strassen's 7 products for 2x2 matrix multiplication
    # C = A*B where A,B are 2x2
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))  # (4, 7)
    V = np.zeros((m*p, rank))  # (4, 7)
    W = np.zeros((p*n, rank))  # (4, 7)
    
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # A indices: i (row 0-1), j (col 0-1) -> A[i,j] at U[i*2+j, c]
    # B indices: j (row 0-1), k (col 0-1) -> B[j,k] at V[j*2+k, c]
    # C indices: k (col 0-1), i (row 0-1) -> C[i,k] at W[k*2+i, c]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 += M1, C11 += M1 (W[0,0]=C00, W[3,0]=C11... wait)
    
    # Let me be more careful with W indexing: W[k*n+i, c] where n=2
    # W[0*2+0, c] = W[0, c] -> C[0,0]
    # W[0*2+1, c] = W[1, c] -> C[1,0]  
    # W[1*2+0, c] = W[2, c] -> C[0,1]
    # W[1*2+1, c] = W[3, c] -> C[1,1]
    
    # M1 contributes to C00 and C11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10
    W[3, 1] = -1  # C11 -= M2
    
    # M3 = A00*(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C10 += M3... wait C10 = M2+M4, C01 = M3+M5
    # C01 = M3+M5, so W[2, 2] = 1
    W[2, 2] = 1
    
    # M4 = A11*(B01-B10)
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1  # C00 += M4
    W[2, 3] = 1  # C10 += M4
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1  # C00 -= M5
    W[2, 4] = 1   # C01 += M5
    
    # M6 = (A10-A00)*(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1   # C11 += M6
    
    # M7 = (A01-A11)*(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1   # C00 += M7
    
    return U, V, W

def _schoolbook_224():
    """Return U, V, W for 2x2x4 schoolbook decomposition (rank 8)."""
    n = 2; m = 2; p = 4
    rank = n * m * p  # 16? No, that's too much.
    # Actually for 2x2x4, schoolbook is 2*2*4 = 16? No.
    # Wait: multiplying 2x2 by 2x4 gives 2x4 output.
    # Each output element C[i,k] = sum_j A[i,j]*B[j,k]
    # There are 2*4 = 8 output elements, each needs 2 multiplications = 16 total?
    # No! The rank of the tensor is at most n*m*p = 2*2*4 = 16 for schoolbook.
    # But actually, we can share: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # So we need 2 (for j) * 2 (for i) * 4 (for k) = 16 terms? No.
    # Each term is A[i,j]*B[j,k] contributing to C[i,k].
    # There are 2*2*4 = 16 such terms.
    # But we can factor: for fixed j, we have A[i,j] (i=0,1) and B[j,k] (k=0,1,2,3)
    # So for each j, we have a rank-2 decomposition of the 2x4 sub-tensor.
    # Total rank = 2 * 2 = 4? No.
    
    # Let me just do it properly:
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    # Term (i,j,k): A[i,j] * B[j,k] -> contributes to C[i,k]
    # Number of terms = 2*2*4 = 16
    
    rank = 16
    U = np.zeros((n*m, rank))  # (4, 16)
    V = np.zeros((m*p, rank))  # (8, 16)
    W = np.zeros((p*n, rank))  # (8, 16)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U, V, W

def _kron_decompose(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two tensor decompositions."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    rank = r1 * r2
    
    n1, m1, p1 = U1.shape[0]//m1 if False else (None, None, None)
    # Need to infer dimensions
    # U1 shape: (n1*m1, r1), V1 shape: (m1*p1, r1), W1 shape: (p1*n1, r1)
    # U2 shape: (n2*m2, r2), V2 shape: (m2*p2, r2), W2 shape: (p2*n2, r2)
    
    n1m1 = U1.shape[0]
    m1p1 = V1.shape[0]
    p1n1 = W1.shape[0]
    n2m2 = U2.shape[0]
    m2p2 = V2.shape[0]
    p2n2 = W2.shape[0]
    
    # Infer n1, m1, p1
    # n1*m1 = n1m1, m1*p1 = m1p1, p1*n1 = p1n1
    # For 2x2x2: n1=m1=p1=2
    # For 2x2x4: n2=2, m2=2, p2=4
    
    # We know the target is 4x4x8, so:
    # n = n1*n2, m = m1*m2, p = p1*p2
    # 4 = n1*n2, 4 = m1*m2, 8 = p1*p2
    # With n1=m1=p1=2: n2=2, m2=2, p2=4. Yes.
    
    U = np.zeros((n1m1 * n2m2, rank))
    V = np.zeros((m1p1 * m2p2, rank))
    W = np.zeros((p1n1 * p2n2, rank))
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            # U: index (i1*m1+j1, i2*m2+j2) -> (i1*n2m2 + i2, j1) * ... 
            # Actually U index is [i*m+j, c]
            # For Kronecker: i = i1*n2 + i2, j = j1*m2 + j2
            # So U[i*m+j, c] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
            # But m = m1*m2, so i*m+j = (i1*n2+i2)*m1*m2 + j1*m2+j2
            # = i1*n2*m1*m2 + i2*m1*m2 + j1*m2 + j2
            
            # Let me just iterate over all combinations
            for i1 in range(2):  # n1=2
                for j1 in range(2):  # m1=2
                    for i2 in range(2):  # n2=2
                        for j2 in range(2):  # m2=2
                            idx1 = i1 * m1 + j1
                            idx2 = i2 * m2 + j2
                            idx = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                            U[idx, c] = U1[idx1, c1] * U2[idx2, c2]
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(4):
                            idx1 = j1 * p1 + k1
                            idx2 = j2 * p2 + k2
                            idx = (j1 * 2 + j2) * 4 + (k1 * 4 + k2)
                            V[idx, c] = V1[idx1, c1] * V2[idx2, c2]
            
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(4):
                        for i2 in range(2):
                            idx1 = k1 * n1 + i1
                            idx2 = k2 * n2 + i2
                            idx = (k1 * 2 + k2) * 2 + (i1 * 2 + i2)
                            W[idx, c] = W1[idx1, c1] * W2[idx2, c2]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Get Strassen 2x2x2
    U1, V1, W1 = _strassen_222()
    
    # Get schoolbook 2x2x4
    U2, V2, W2 = _schoolbook_224()
    
    # Kronecker product
    U, V, W = _kron_decompose(U1, V1, W1, U2, V2, W2)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
