"""
Rewritten program for 4x4x4 matrix multiplication tensor decomposition.
Uses the Kronecker product of the rank-7 Strassen decomposition for 2x2x2
to achieve a rank of 49.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as numpy arrays of shape (4, 7).
    
    Indices:
    U[i*2+j, r] = coeff of A[i,j]
    V[j*2+k, r] = coeff of B[j,k]
    W[k*2+i, r] = coeff of C[i,k] (Trace convention W[k*n+i, r])
    """
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Get base Strassen decomposition
    U2, V2, W2 = _strassen_2x2()
    
    # Construct 4x4 decomposition via Kronecker product
    # Rank = 7 * 7 = 49
    rank = 49
    
    # Reshape U2, V2, W2 to separate indices
    # U2: (i*2+j, r) -> (i, j, r) where i,j in {0,1}
    # We want to form U4[i*4+j, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    # Indices for 4x4: i = 2*i1 + i2, j = 2*j1 + j2
    # U2 indices: i1, j1 are the "block" indices, i2, j2 are the "inner" indices?
    # Actually, for Kronecker product of matrices A (4x4) = A1 (2x2) x A2 (2x2) is not quite right.
    # Matrix multiplication (AB)_{ik} = sum_j A_{ij} B_{jk}.
    # If we view A, B, C as 2x2 blocks of 2x2 matrices:
    # A = [[A11, A12], [A21, A22]] where Aij are 2x2 matrices.
    # C = A*B. C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor for 4x4 mult is the Kronecker product of the tensor for 2x2 mult.
    # T4 = T2 \otimes T2.
    # U4 = U2 \otimes U2, V4 = V2 \otimes V2, W4 = W2 \otimes W2.
    
    # We need to be careful with index ordering.
    # U2 shape (4, 7). Indices (i2*2+j2, r2).
    # We want U4 shape (16, 49). Indices (i4, r4).
    # i4 = i1*4 + i2? No, standard Kronecker index mapping:
    # If A = A1 \otimes A2, then A[i1*2+i2, j1*2+j2] = A1[i1,j1]A2[i2,j2].
    # Here, the "matrix" being decomposed is the multiplication tensor.
    # The factors U, V, W are matrices.
    # U4[i4, r4] = U2[i1, r1] U2[i2, r2] where i4 = i1*2 + i2? 
    # Let's check the index mapping for the 4x4 matrix indices.
    # A is 4x4. A[i,j]. If we split i = 2*i1 + i2, j = 2*j1 + j2.
    # Then A[i,j] is an element.
    # The Kronecker product of two 2x2 multiplications corresponds to:
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Using indices i=2*i1+i2, j=2*j1+j2, k=2*k1+k2.
    # C[2*i1+i2, 2*k1+k2] = sum_{j1,j2} A[2*i1+i2, 2*j1+j2] B[2*j1+j2, 2*k1+k2].
    # This is not simply a Kronecker product of the tensors unless we reorder indices.
    # The standard Kronecker product of matrix multiplication tensors T2 \otimes T2 
    # computes (A \otimes A') (B \otimes B') = (AB) \otimes (A'B').
    # This corresponds to block matrix multiplication.
    # The indices for the block matrix multiplication are:
    # Block I (2x2) contains 2x2 matrices.
    # A_block[i1, j1] is a 2x2 matrix with elements A[2*i1+i2, 2*j1+j2].
    # Then C_block[i1, k1] = sum_j1 A_block[i1, j1] B_block[j1, k1].
    # Element (i2, k2) of C_block[i1, k1] is:
    # C[2*i1+i2, 2*k1+k2] = sum_{j1} sum_{j2} A[2*i1+i2, 2*j1+j2] B[2*j1+j2, 2*k1+k2].
    # This matches the 4x4 multiplication exactly.
    # So the tensor is indeed T2 \otimes T2.
    
    # Therefore, U4 = U2 \otimes U2, etc.
    # We need to map the indices correctly.
    # U2 has shape (4, 7). Row index u2_idx = i2*2 + j2? No, U2 row index is i*2+j.
    # In the Kronecker product of tensors, the indices are multiplied.
    # U4[i4, r4] = U2[i1, r1] * U2[i2, r2]
    # where i4 = i1*2 + i2? 
    # Let's look at the row index of U4. It corresponds to the pair (i,j) of the 4x4 matrix A.
    # i = 2*i1 + i2, j = 2*j1 + j2.
    # The row index in U4 is i*4 + j = (2*i1+i2)*4 + (2*j1+j2) = 8*i1 + 4*i2 + 2*j1 + j2.
    # The row index in U2 is i*2 + j.
    # For the first factor (block), the indices are i1, j1. Row index u2_1 = i1*2 + j1.
    # For the second factor (inner), the indices are i2, j2. Row index u2_2 = i2*2 + j2.
    # So U4[8*i1 + 4*i2 + 2*j1 + j2, 7*r1 + r2] = U2[i1*2 + j1, r1] * U2[i2*2 + j2, r2].
    
    # We can construct this using broadcasting.
    # U2_reshaped1: (2, 2, 7) -> axes i1, j1, r1. Values U2[i1*2+j1, r1]
    # U2_reshaped2: (2, 2, 7) -> axes i2, j2, r2. Values U2[i2*2+j2, r2]
    
    U2_r1 = U2.reshape(2, 2, 7) # i1, j1, r1
    U2_r2 = U2.reshape(2, 2, 7) # i2, j2, r2
    
    # We want to compute the outer product and permute axes.
    # Tensor T_U shape (i1, j1, r1, i2, j2, r2)
    T_U = U2_r1[:, :, :, None, None, None] * U2_r2[None, None, None, :, :, :]
    # Axes: 0:i1, 1:j1, 2:r1, 3:i2, 4:j2, 5:r2
    # We want to reshape to (16, 49) with row index 8*i1 + 4*i2 + 2*j1 + j2.
    # This order is i1, i2, j1, j2.
    # So we need to transpose T_U to (i1, i2, j1, j2, r1, r2).
    T_U_perm = np.transpose(T_U, (0, 3, 1, 4, 2, 5))
    U = T_U_perm.reshape(16, 49)
    
    # Similarly for V
    # V2 row index j*2+k.
    # V4 row index j*4+k = (2*j1+j2)*4 + (2*k1+k2) = 8*j1 + 4*j2 + 2*k1 + k2.
    # V2_r1: (j1, k1, r1). V2_r2: (j2, k2, r2).
    V2_r1 = V2.reshape(2, 2, 7) # j1, k1, r1
    V2_r2 = V2.reshape(2, 2, 7) # j2, k2, r2
    T_V = V2_r1[:, :, :, None, None, None] * V2_r2[None, None, None, :, :, :]
    # Axes: j1, k1, r1, j2, k2, r2
    # Target order: j1, j2, k1, k2.
    T_V_perm = np.transpose(T_V, (0, 3, 1, 4, 2, 5))
    V = T_V_perm.reshape(16, 49)
    
    # Similarly for W
    # W2 row index k*2+i.
    # W4 row index k*4+i = (2*k1+k2)*4 + (2*i1+i2) = 8*k1 + 4*k2 + 2*i1 + i2.
    # W2_r1: (k1, i1, r1). W2_r2: (k2, i2, r2).
    W2_r1 = W2.reshape(2, 2, 7) # k1, i1, r1
    W2_r2 = W2.reshape(2, 2, 7) # k2, i2, r2
    T_W = W2_r1[:, :, :, None, None, None] * W2_r2[None, None, None, :, :, :]
    # Axes: k1, i1, r1, k2, i2, r2
    # Target order: k1, k2, i1, i2.
    T_W_perm = np.transpose(T_W, (0, 3, 1, 4, 2, 5))
    W = T_W_perm.reshape(16, 49)
    
    if time.time() > deadline:
        return _trivial_decomp(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _trivial_decomp(n, m, p):
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
# EVOLVE-BLOCK-END
