import numpy as np
import time

def _strassen_2x2():
    """Returns U, V, W for 2x2 Strassen (Rank 7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    # M3 = A00(B10 - B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    # M4 = A11(B01 - B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    # M5 = (A00 + A10)B11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    # M6 = (A01 - A00)(B00 + B10)
    U[1, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    # M7 = (A11 - A01)(B01 + B11)
    U[3, 6] = 1; U[1, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1
    return U, V, W

def _schoolbook_3x3():
    """Returns U, V, W for 3x3 Schoolbook (Rank 27)."""
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))
    r = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3 + j, r] = 1
                V[j*3 + k, r] = 1
                W[k*3 + i, r] = 1
                r += 1
    return U, V, W

def _kronecker_product(U1, V1, W1, U2, V2, W2):
    """
    Computes the Kronecker product of two tensor decompositions.
    U1, V1, W1 are for n1 x n1 matrix multiplication.
    U2, V2, W2 are for n2 x n2 matrix multiplication.
    Result is for (n1*n2) x (n1*n2) matrix multiplication.
    """
    n1 = int(np.sqrt(U1.shape[0]))
    n2 = int(np.sqrt(U2.shape[0]))
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    R = r1 * r2
    N = n1 * n2
    
    U = np.zeros((N*N, R))
    V = np.zeros((N*N, R))
    W = np.zeros((N*N, R))
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r = r1_idx * r2 + r2_idx
            for i1 in range(n1):
                for i2 in range(n2):
                    for j1 in range(n1):
                        for j2 in range(n2):
                            # Index in the large matrix
                            # Row index of A: (i1, i2) -> i1 * n2 + i2
                            # Col index of A: (j1, j2) -> j1 * n2 + j2
                            # A index: (i1*n2 + i2) * N + (j1*n2 + j2)
                            a_idx = (i1 * n2 + i2) * N + (j1 * n2 + j2)
                            
                            # B index: (j1*n2 + j2) * N + (k1*n2 + k2)
                            # We need to sum over k1, k2?
                            # No, the Kronecker product of tensors T1 and T2 is T1[i1, j1, k1] * T2[i2, j2, k2].
                            # This corresponds to the tensor for A1 x A2 and B1 x B2.
                            # The indices for the large tensor are:
                            # i = i1 * n2 + i2
                            # j = j1 * n2 + j2
                            # k = k1 * n2 + k2
                            # The term is U1[i1, j1, r1] * U2[i2, j2, r2] for the A part.
                            # Wait, U is indexed by (i, j).
                            # So U[a_idx, r] = U1[i1*n1+j1, r1_idx] * U2[i2*n2+j2, r2_idx]
                            
                            U[a_idx, r] = U1[i1 * n1 + j1, r1_idx] * U2[i2 * n2 + j2, r2_idx]
                            
                            # V index: (j, k) -> (j1*n2+j2) * N + (k1*n2+k2)
                            # But we don't have k1, k2 here.
                            # The Kronecker product structure means that for a fixed r,
                            # the tensor is the outer product of the rank-1 tensors of M1 and M2.
                            # So V[b_idx, r] = V1[j1*n1+k1, r1_idx] * V2[j2*n2+k2, r2_idx]
                            # We need to iterate over k1, k2 as well?
                            # No, the rank is R = r1 * r2.
                            # Each rank-1 term in the result is the product of a rank-1 term from M1 and M2.
                            # So we need to fill V and W for all indices.
                            # But V depends on k.
                            # So we must iterate over k1, k2 inside the loop?
                            # If we do, we will write to the same U, V, W entries multiple times?
                            # No, U depends only on i, j.
                            # V depends on j, k.
                            # W depends on k, i.
                            # So for a fixed r, we need to define U, V, W for all i, j, k.
                            # The Kronecker product of two rank-1 tensors u1 v1 w1 and u2 v2 w2
                            # is (u1 x u2) (v1 x v2) (w1 x w2).
                            # So U[i,j] = U1[i1,j1] * U2[i2,j2]
                            # V[j,k] = V1[j1,k1] * V2[j2,k2]
                            # W[k,i] = W1[k1,i1] * W2[k2,i2]
                            # This requires iterating over all i, j, k.
                            
                            # Let's restructure the loops.
                            pass
    # The above loop structure is inefficient or incorrect for filling V and W.
    # Let's use a simpler approach: iterate over all i, j, k of the large matrix.
    # But that would be O(N^6) which is too slow.
    # We should iterate over the rank-1 components.
    
    # Correct approach:
    # For each r1_idx, r2_idx:
    #   For each i1, i2, j1, j2, k1, k2:
    #     U[(i1*n2+i2)*N + (j1*n2+j2), r] = U1[i1*n1+j1, r1_idx] * U2[i2*n2+j2, r2_idx]
    #     V[(j1*n2+j2)*N + (k1*n2+k2), r] = V1[j1*n1+k1, r1_idx] * V2[j2*n2+k2, r2_idx]
    #     W[(k1*n2+k2)*N + (i1*n2+i2), r] = W1[k1*n1+i1, r1_idx] * W2[k2*n2+i2, r2_idx]
    
    # This is O(R * N^6) which is huge.
    # But we can vectorize or use the fact that U, V, W are sparse.
    # Actually, we can just construct the matrices by iterating over the non-zero entries of U1, V1, W1 and U2, V2, W2.
    
    # Let's use a dictionary to store the non-zero entries.
    # Or just use numpy broadcasting if we reshape.
    
    # Reshape U1 to (n1, n1, r1).
    U1_reshaped = U1.reshape(n1, n1, r1)
    U2_reshaped = U2.reshape(n2, n2, r2)
    
    # The Kronecker product of the tensors is not simply the Kronecker product of the matrices.
    # We need to compute the tensor product.
    
    # Let's just use the explicit loops with numpy arrays for speed.
    # N=6, so N^6 = 46656. R=189.
    # 46656 * 189 is about 8.8 million operations. This is fine.
    
    U = np.zeros((N*N, R))
    V = np.zeros((N*N, R))
    W = np.zeros((N*N, R))
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r = r1_idx * r2 + r2_idx
            for i1 in range(n1):
                for i2 in range(n2):
                    for j1 in range(n1):
                        for j2 in range(n2):
                            a_idx = (i1 * n2 + i2) * N + (j1 * n2 + j2)
                            u_val = U1[i1 * n1 + j1, r1_idx] * U2[i2 * n2 + j2, r2_idx]
                            if u_val != 0:
                                U[a_idx, r] = u_val
                            
                            for k1 in range(n1):
                                for k2 in range(n2):
                                    b_idx = (j1 * n2 + j2) * N + (k1 * n2 + k2)
                                    v_val = V1[j1 * n1 + k1, r1_idx] * V2[j2 * n2 + k2, r2_idx]
                                    if v_val != 0:
                                        V[b_idx, r] = v_val
                                    
                                    c_idx = (k1 * n2 + k2) * N + (i1 * n2 + i2)
                                    w_val = W1[k1 * n1 + i1, r1_idx] * W2[k2 * n2 + i2, r2_idx]
                                    if w_val != 0:
                                        W[c_idx, r] = w_val
    return U, V, W

def _verify(n, m, p, U, V, W, tol=1e-6):
    """Verify the decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    np.random.seed(42)
    for _ in range(150):
        a = np.random.randint(0, n*m)
        b = np.random.randint(0, m*p)
        c = np.random.randint(0, p*n)
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 110.0
    
    # Use Kronecker product of 2x2 Strassen and 3x3 Schoolbook
    U2, V2, W2 = _strassen_2x2()
    U3, V3, W3 = _schoolbook_3x3()
    
    U, V, W = _kronecker_product(U2, V2, W2, U3, V3, W3)
    
    if _verify(6, 6, 6, U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook 216
    U_sb = np.zeros((36, 216))
    V_sb = np.zeros((36, 216))
    W_sb = np.zeros((36, 216))
    col = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                U_sb[i*6+j, col] = 1
                V_sb[j*6+k, col] = 1
                W_sb[k*6+i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
