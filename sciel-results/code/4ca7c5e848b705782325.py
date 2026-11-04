import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    # Build target tensor
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    schoolbook_rank = n * m * p  # 112
    best_result = None
    best_rank = schoolbook_rank
    
    # Strategy 1: Strassen 4x4x4 (rank 49) + Structured 4x4x3 (rank 44)
    U_str, V_str, W_str = _strassen_4x4x4()
    
    # The Strassen part covers k=0,1,2,3
    # We need to add the part for k=4,5,6
    # This is equivalent to computing the last 3 columns of the product.
    
    U_res, V_res, W_res = _structured_4x4x3()
    
    U_combined = np.hstack([U_str, U_res])
    V_combined = np.hstack([V_str, V_res])
    W_combined = np.hstack([W_str, W_res])
    
    recon = np.einsum('ak,bk,ck->abc', U_combined, V_combined, W_combined)
    if np.allclose(recon, T, atol=1e-10):
        total_rank = U_combined.shape[1]
        if total_rank < best_rank:
            best_rank = total_rank
            best_result = {
                "U": U_combined.tolist(),
                "V": V_combined.tolist(),
                "W": W_combined.tolist()
            }
    
    # Fallback: schoolbook
    if best_result is None or best_rank >= schoolbook_rank:
        return _schoolbook(n, m, p)
    
    return best_result


def _strassen_4x4x4():
    """Return Strassen-based decomposition of 4x4x4 matrix multiplication.
    
    Uses Kronecker product of 2x2x2 Strassen decomposition.
    Returns U, V, W each of shape (16, 49) such that
    sum_k U[a,k]*V[b,k]*W[c,k] = T[a,b,c] for the 4x4x4 part.
    """
    U_2, V_2, W_2 = _strassen_2x2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            idx = r1 * 7 + r2
            U[:, idx] = np.kron(U_2[:, r1], U_2[:, r2])
            V[:, idx] = np.kron(V_2[:, r1], V_2[:, r2])
            W[:, idx] = np.kron(W_2[:, r1], W_2[:, r2])
    
    # Remap V and W to 4x4x7 layout (V is 28, W is 28)
    # V index is j*7+k. For 4x4x4, k is 0..3.
    # W index is k*4+i. For 4x4x4, k is 0..3.
    
    V_full = np.zeros((28, 49))
    for j in range(4):
        for k in range(4):
            V_full[j*7+k, :] = V[j*4+k, :]
    
    W_full = np.zeros((28, 49))
    for k in range(4):
        for i in range(4):
            W_full[k*4+i, :] = W[k*4+i, :]
    
    return U, V_full, W_full


def _strassen_2x2x2():
    """Return Strassen decomposition of 2x2x2 matrix multiplication.
    
    Returns U, V, W each of shape (4, 7) such that
    sum_k U[a,k]*V[b,k]*W[c,k] = T[a,b,c]
    where T[i*2+j, j*2+k, k*2+i] = 1
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _structured_4x4x3():
    """
    Construct a rank-44 decomposition for the 4x4x3 part of the tensor.
    This corresponds to the last 3 columns (k=4,5,6) of the 4x4x7 product.
    We use 4 blocks of 2x2x3 decomposition (rank 11 each).
    """
    U_223, V_223, W_223 = _strassen_2x2x3()
    
    # The 4x4x3 tensor is the sum of 4 terms:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01, B02], [B10, B11, B12]]
    # C = A B
    # Each entry C_ij is a sum of two 2x2x3 products.
    # However, we can just treat the 4x4x3 multiplication as 4 independent 2x2x3 multiplications
    # if we don't share the B factors.
    # Since we want a valid decomposition, we can just concatenate 4 copies of the 2x2x3 decomposition
    # and map the indices appropriately.
    
    # But wait, the B matrix is shared.
    # If we use independent decompositions, we are effectively using 4 different B's.
    # This will not sum to the correct T unless we account for the sharing.
    
    # Actually, the tensor for 4x4x3 is NOT a direct sum of 4 2x2x3 tensors.
    # It is a "block" tensor.
    # However, the rank of the block tensor is at most the sum of the ranks of the blocks.
    # So rank <= 4 * 11 = 44.
    
    # To construct it, we need to map the 2x2x3 factors to the 4x4x3 indices.
    
    U = np.zeros((16, 44))
    V = np.zeros((28, 44))
    W = np.zeros((28, 44))
    
    # The 4 blocks correspond to (i', j') in {0,1}x{0,1}
    # A_{i,j} = A_{2i'+i'', 2j'+j''}
    # B_{j,k} = B_{2j'+j'', k}  (k is 4,5,6 in global, 0,1,2 in local)
    # C_{k,i} = C_{k, 2i'+i''}
    
    # For each block (i', j'), we use a copy of the 2x2x3 decomposition.
    # The U factors for this block will be non-zero only for i=2i'+i'', j=2j'+j''
    # The V factors for this block will be non-zero only for j=2j'+j'', k=4..6
    # The W factors for this block will be non-zero only for k=4..6, i=2i'+i''
    
    for ip in range(2):
        for jp in range(2):
            # Local indices
            # i'' in 0,1 -> global i = 2*ip + i''
            # j'' in 0,1 -> global j = 2*jp + j''
            # k'' in 0,1,2 -> global k = 4 + k''
            
            offset = (ip * 2 + jp) * 11
            
            for local_k in range(11):
                global_k = offset + local_k
                
                # U: shape (16, 11) for this block
                # U_223 has shape (4, 11)
                # Local a = i''*2 + j''
                # Global a = i*4 + j = (2*ip+i'')*4 + (2*jp+j'')
                
                for i2 in range(2):
                    for j2 in range(2):
                        local_a = i2 * 2 + j2
                        global_a = (2*ip + i2) * 4 + (2*jp + j2)
                        U[global_a, global_k] = U_223[local_a, local_k]
                
                # V: shape (28, 11) for this block
                # V_223 has shape (6, 11) (2x2x3 -> V is 2*3=6)
                # Local b = j''*3 + k''
                # Global b = j*7 + k = (2*jp+j'')*7 + (4+k'')
                
                for j2 in range(2):
                    for k2 in range(3):
                        local_b = j2 * 3 + k2
                        global_b = (2*jp + j2) * 7 + (4 + k2)
                        V[global_b, global_k] = V_223[local_b, local_k]
                
                # W: shape (28, 11) for this block
                # W_223 has shape (6, 11) (2x2x3 -> W is 3*2=6)
                # Local c = k''*2 + i''
                # Global c = k*4 + i = (4+k'')*4 + (2*ip+i'')
                
                for k2 in range(3):
                    for i2 in range(2):
                        local_c = k2 * 2 + i2
                        global_c = (4 + k2) * 4 + (2*ip + i2)
                        W[global_c, global_k] = W_223[local_c, local_k]
    
    return U, V, W


def _strassen_2x2x3():
    """
    Return a rank-11 decomposition for 2x2x3 matrix multiplication.
    U: (4, 11), V: (6, 11), W: (6, 11)
    T[i*2+j, j*3+k, k*2+i] = 1
    """
    # We can use Strassen 2x2x2 (rank 7) for the first 2 columns
    # and Schoolbook 2x2x1 (rank 4) for the 3rd column.
    
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    # Part 1: 2x2x2 (k=0,1)
    U2, V2, W2 = _strassen_2x2x2()
    # V2 is (4, 7) for 2x2x2 (j*2+k). We need j*3+k for k=0,1.
    # V2[j*2+k, col]
    # We need V[j*3+k, col] for k=0,1.
    # So V[j*3+0, col] = V2[j*2+0, col]
    #    V[j*3+1, col] = V2[j*2+1, col]
    
    for j in range(2):
        for k in range(2):
            V[j*3+k, k] = V2[j*2+k, k]
    
    U[:, 0:7] = U2
    W[:, 0:7] = W2
    
    # Part 2: 2x2x1 (k=2)
    # Schoolbook: rank 4.
    # T[i*2+j, j*3+2, 2*2+i] = 1
    # U[i*2+j, col] = 1
    # V[j*3+2, col] = 1
    # W[2*2+i, col] = 1
    
    for i in range(2):
        for j in range(2):
            col = 7 + i*2 + j
            a = i*2 + j
            b = j*3 + 2
            c = 2*2 + i
            U[a, col] = 1
            V[b, col] = 1
            W[c, col] = 1
    
    return U, V, W


def _schoolbook(n, m, p):
    """Return schoolbook decomposition of n x m times m x p."""
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
