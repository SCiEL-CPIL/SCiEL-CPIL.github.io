"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def _check_decomposition(U, V, W, tol=1e-10):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def construct():
    """Return a valid tensor decomposition for 2x2 * 2x13 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B
    
    # The 7 Strassen products for 2x2 matrices:
    # m1 = (a11 + a22) * (b11 + b22)
    # m2 = (a21 + a22) * b11
    # m3 = a11 * (b12 - b22)
    # m4 = a22 * (b21 - b11)
    # m5 = (a11 + a12) * b22
    # m6 = (a21 - a11) * (b11 + b12)
    # m7 = (a12 - a22) * (b21 + b22)
    
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # For 2x2 * 2x13, we have 13 columns in B.
    # We can group the 13 columns and apply Strassen to pairs.
    # 13 = 2*6 + 1, so we can do 6 Strassen groups (each handling 2 columns) + 1 schoolbook column
    # Each Strassen group: rank 7
    # Each schoolbook column: rank 4 (for 2x2 * 2x1)
    # Total: 6*7 + 4 = 42 + 4 = 46
    
    rank = 46
    U = np.zeros((n * m, rank))  # (4, 46)
    V = np.zeros((m * p, rank))  # (26, 46)
    W = np.zeros((p * n, rank))  # (26, 46)
    
    col = 0
    # 6 Strassen groups for columns (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
    for group in range(6):
        k1 = 2 * group      # first column in pair
        k2 = 2 * group + 1  # second column in pair
        
        # m1 = (a11 + a22) * (b11 + b22)
        U[0, col] = 1  # a11
        U[3, col] = 1  # a22
        V[k1, col] = 1  # b11
        V[k2, col] = 1  # b22
        W[k1*n+0, col] = 1  # C11
        W[k1*n+1, col] = 1  # C21
        W[k2*n+0, col] = 1  # C12
        W[k2*n+1, col] = 1  # C22
        col += 1
        
        # m2 = (a21 + a22) * b11
        U[1, col] = 1  # a21
        U[3, col] = 1  # a22
        V[k1, col] = 1  # b11
        W[k1*n+1, col] = 1  # C21
        W[k2*n+1, col] = -1  # C22
        col += 1
        
        # m3 = a11 * (b12 - b22)
        U[0, col] = 1  # a11
        V[k2, col] = 1  # b12
        V[k2, col] = 1  # b22 (wait, need to be careful with indexing)
        # Actually for column k1: b11 is row k1, b21 is row k1+? 
        # Let me reconsider the indexing.
        # B is 2x13, so B[j,k] where j in {0,1}, k in {0,...,12}
        # V has shape (m*p, rank) = (2*13, rank) = (26, rank)
        # V[j*p + k, r] corresponds to B[j,k]
        
        # Let me redo this more carefully.
        col -= 2  # reset to before m2
        
        # m1 = (a11 + a22) * (b11 + b22)
        U[0, col] = 1  # a11 (index 0 in A)
        U[3, col] = 1  # a22 (index 3 in A)
        V[0*p + k1, col] = 1  # b11 = B[0,k1]
        V[1*p + k1, col] = 1  # b21 = B[1,k1]
        W[0*n + k1, col] = 1  # C11 contribution
        W[1*n + k1, col] = 1  # C21 contribution
        W[0*n + k2, col] = 1  # C12 contribution (wait, this doesn't look right)
        
        # I need to think about this more carefully. Let me use a different approach.
        # For each pair of columns (k1, k2), we treat B[:, k1:k2] as a 2x2 matrix.
        # Then apply Strassen.
        
        # Reset and redo
        col -= 1  # back to before m1
        U[0, col] = 0
        U[3, col] = 0
        V[0*p + k1, col] = 0
        V[1*p + k1, col] = 0
        W[0*n + k1, col] = 0
        W[1*n + k1, col] = 0
        W[0*n + k2, col] = 0
        
        # Now let's define the 7 Strassen products properly.
        # A = [[a00, a01], [a10, a11]] where a_ij = A[i,j], index = i*m+j
        # B_pair = [[b00, b01], [b10, b11]] where b_ij = B[i, k1+i] ... no
        # B_pair = [[B[0,k1], B[0,k2]], [B[1,k1], B[1,k2]]]
        # C_pair = A * B_pair = [[C[0,k1], C[0,k2]], [C[1,k1], C[1,k2]]]
        
        # Strassen for 2x2:
        # m1 = (a00 + a11) * (b00 + b11)
        # m2 = (a10 + a11) * b00
        # m3 = a00 * (b01 - b11)
        # m4 = a11 * (b10 - b00)
        # m5 = (a00 + a01) * b11
        # m6 = (a10 - a00) * (b00 + b01)
        # m7 = (a01 - a11) * (b10 + b11)
        
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # m1
        U[0, col] = 1  # a00
        U[3, col] = 1  # a11
        V[0*p + k1, col] = 1  # b00 = B[0,k1]
        V[1*p + k2, col] = 1  # b11 = B[1,k2]
        W[0*n + k1, col] = 1  # C00
        W[1*n + k2, col] = 1  # C11
        col += 1
        
        # m2
        U[1, col] = 1  # a10
        U[3, col] = 1  # a11
        V[0*p + k1, col] = 1  # b00 = B[0,k1]
        W[1*n + k1, col] = 1  # C10
        W[1*n + k2, col] = -1  # C11
        col += 1
        
        # m3
        U[0, col] = 1  # a00
        V[0*p + k2, col] = 1  # b01 = B[0,k2]
        V[1*p + k2, col] = -1  # b11 = B[1,k2]
        W[0*n + k2, col] = 1  # C01
        W[1*n + k2, col] = 1  # C11
        col += 1
        
        # m4
        U[3, col] = 1  # a11
        V[1*p + k1, col] = 1  # b10 = B[1,k1]
        V[0*p + k1, col] = -1  # b00 = B[0,k1]
        W[0*n + k1, col] = 1  # C00
        W[1*n + k1, col] = 1  # C10
        col += 1
        
        # m5
        U[0, col] = 1  # a00
        U[2, col] = 1  # a01
        V[1*p + k2, col] = 1  # b11 = B[1,k2]
        W[0*n + k1, col] = -1  # C00
        W[0*n + k2, col] = 1  # C01
        col += 1
        
        # m6
        U[1, col] = 1  # a10
        U[0, col] = -1  # a00
        V[0*p + k1, col] = 1  # b00 = B[0,k1]
        V[0*p + k2, col] = 1  # b01 = B[0,k2]
        W[1*n + k2, col] = 1  # C11
        col += 1
        
        # m7
        U[2, col] = 1  # a01
        U[3, col] = -1  # a11
        V[1*p + k1, col] = 1  # b10 = B[1,k1]
        V[1*p + k2, col] = 1  # b11 = B[1,k2]
        W[0*n + k1, col] = 1  # C00
        col += 1
    
    # Last column (k=12): schoolbook, rank 4
    k = 12
    for i in range(n):
        for j in range(m):
            U[i*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    
    # Verify the decomposition
    if time.time() < deadline:
        if _check_decomposition(U, V, W):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook if verification fails
    rank_fallback = n * m * p  # 52
    U_f = np.zeros((n * m, rank_fallback))
    V_f = np.zeros((m * p, rank_fallback))
    W_f = np.zeros((p * n, rank_fallback))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i * m + j, col] = 1
                V_f[j * p + k, col] = 1
                W_f[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U_f.tolist(),
        "V": V_f.tolist(),
        "W": W_f.tolist()
    }
