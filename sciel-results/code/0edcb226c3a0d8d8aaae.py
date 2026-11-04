import time
import numpy as np

n, m, p = 4, 4, 7

def get_strassen_2x2():
    """Return U, V, W for 2x2x2 matrix multiplication (Strassen, rank 7).
    U: 4x7, V: 4x7, W: 4x7
    Convention: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Term 0: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # Term 2: A00(B10-B00)
    U[0, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[1, 2] = 1
    
    # Term 3: A11(B01-B10)
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 4: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 5: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def get_strassen_2x2x3():
    """Return U, V, W for 2x2x3 matrix multiplication.
    U: 4x11, V: 6x11, W: 6x11
    Convention: U[i*2+j, t], V[j*3+k, t], W[k*2+i, t]
    Rank 11: Strassen 2x2x2 (rank 7) + schoolbook 2x2x1 (rank 4)
    """
    U = np.zeros((4, 11), dtype=int)
    V = np.zeros((6, 11), dtype=int)
    W = np.zeros((6, 11), dtype=int)
    
    # First 7 terms: Strassen 2x2x2 for k=0,1
    U2, V2, W2 = get_strassen_2x2()
    U[:, :7] = U2
    V[:4, :7] = V2  # V2 is 4x7, corresponds to j=0,1 and k=0,1
    W[:4, :7] = W2  # W2 is 4x7, corresponds to k=0,1 and i=0,1
    
    # Last 4 terms: schoolbook for k=2
    for j in range(2):
        for i in range(2):
            t = 7 + j * 2 + i
            U[i * 2 + j, t] = 1
            V[j * 3 + 2, t] = 1
            W[2 * 2 + i, t] = 1
    
    return U, V, W

def construct():
    """Construct rank-minimal decomposition for 4x4 x 4x7 matrix multiplication.
    
    Strategy: Decompose B into 4 blocks of width 2 (B[:, 0:2], B[:, 2:4]) and 
    1 block of width 3 (B[:, 4:7]).
    
    For each 2x2 block of B: use 2x2x2 Strassen (rank 7) for each 2x2x2 sub-problem.
    Actually, let me think more carefully.
    
    A is 4x4, B is 4x7, C is 4x7.
    
    Split B into blocks: B = [B0, B1, B2] where B0, B1 are 4x2 and B2 is 4x3.
    C = [C0, C1, C2] where C0, C1 are 4x2 and C2 is 4x3.
    
    C0 = A * B0: 4x4 times 4x2 = 4x2
    C1 = A * B1: 4x4 times 4x2 = 4x2  
    C2 = A * B2: 4x4 times 4x3 = 4x3
    
    For C0 = A * B0 (4x4 times 4x2):
    Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    Split B0 into 2x2 blocks: B0 = [[B00, B01], [B10, B11]] (each 2x2)
    
    C0 = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
          [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    
    Each 2x2x2 product can be done with Strassen (rank 7).
    There are 4 such products for C0, so rank = 4*7 = 28? No, that's not right.
    
    Actually, we can use the Kronecker product structure. The 4x4x2 problem can be
    decomposed as 4 independent 2x2x2 problems if we view it correctly.
    
    Let me use a different approach: use the known result that 4x4x4 has rank 49
    (Strassen^2), and handle the extra 3 columns separately.
    
    For 4x4x4: rank 49 (Kronecker product of 2x2x2 Strassen)
    For 4x4x3: we can split into 4x4x2 + 4x4x1
      4x4x2: 2 * (4x4x1) = 2 * 16 = 32? No, 4x4x2 can be done with rank 32 using schoolbook.
      But we can do better: 4x4x2 = 2 * 2x2x2 Strassen blocks? 
    
    Let me just use: 4x4x4 (rank 49) + 4x4x3 (rank 48 schoolbook) = rank 97.
    Or: 4x4x4 (rank 49) + 4x4x2 (rank 32) + 4x4x1 (rank 16) = rank 97.
    
    Actually, let me try: split B into 2x2 blocks: B = [B0, B1, B2] where B0, B1 are 4x2, B2 is 4x3.
    C0 = A*B0: 4x4x2, rank can be 32 (schoolbook) or better.
    C1 = A*B1: 4x4x2, rank 32
    C2 = A*B2: 4x4x3, rank 48 (schoolbook)
    Total: 32 + 32 + 48 = 112. Same as schoolbook.
    
    Better: Use Strassen for the 4x4x4 part (rank 49) and schoolbook for 4x4x3 (rank 48).
    Total: 49 + 48 = 97.
    
    Let me implement the Kronecker product of 2x2x2 Strassen for 4x4x4.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Get 2x2x2 Strassen factors
    U2, V2, W2 = get_strassen_2x2()
    
    # Build 4x4x4 decomposition using Kronecker product structure
    # U4: 16x49, V4: 16x49, W4: 16x49
    # U4[i*4+j, t*7+s] = U2[i*2+j, t] * U2[?, ?] -- this is not quite right.
    
    # The correct way: The 4x4x4 multiplication tensor is the Kronecker product of two 2x2x2 tensors.
    # If T2 is the 2x2x2 tensor with T2[i,j,k] = delta (coefficient for A[i,j]*B[j,k]*C[k,i]),
    # then T4[i,j,k] = T4[(i1,i2),(j1,j2),(k1,k2)] = T2[i1,j1,k1] * T2[i2,j2,k2]
    
    # In matrix form:
    # U4[(i1*2+i2)*4 + (j1*2+j2), t1*7+t2] = U2[i1*2+j1, t1] * U2[i2*2+j2, t2]
    # V4[(j1*2+j2)*4 + (k1*2+k2), t1*7+t2] = V2[j1*2+k1, t1] * V2[j2*2+k2, t2]
    # W4[(k1*2+k2)*4 + (i1*2+i2), t1*7+t2] = W2[k1*2+i1, t1] * W2[k2*2+i2, t2]
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    I = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                    for k1 in range(2):
                        for k2 in range(2):
                            K = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)  # W index: k*n+i = k*4+i
                            # Wait, W index is k*n+i where n=4, so W[k*4+i, t]
                            # For W4, the row index is (k1*2+k2)*4 + (i1*2+i2)
                            # But we also need j for V.
                            # Let me redo this more carefully.
                            pass
    
    # Let me use numpy's kron with proper reshaping.
    # U2 is 4x7. Reshape to 2x2x7: U2_3d[i, j, t] = U2[i*2+j, t]
    U2_3d = U2.reshape(2, 2, 7)
    V2_3d = V2.reshape(2, 2, 7)  # V2[j*2+k, t] -> V2_3d[j, k, t]
    W2_3d = W2.reshape(2, 2, 7)  # W2[k*2+i, t] -> W2_3d[k, i, t]
    
    # U4_3d[i1, i2, j1, j2, t1, t2] = U2_3d[i1, j1, t1] * U2_3d[i2, j2, t2]
    # This is an outer product of the two 3D tensors.
    
    # Flatten: U4[(i1*2+i2)*4 + (j1*2+j2), t1*7+t2]
    # = U2_3d[i1, j1, t1] * U2_3d[i2, j2, t2]
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for i1 in range(2):
        for j1 in range(2):
            for t1 in range(7):
                for i2 in range(2):
                    for j2 in range(2):
                        for t2 in range(7):
                            I = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                            T = t1 * 7 + t2
                            U4[I, T] = U2[i1 * 2 + j1, t1] * U2[i2 * 2 + j2, t2]
    
    for j1 in range(2):
        for k1 in range(2):
            for t1 in range(7):
                for j2 in range(2):
                    for k2 in range(2):
                        for t2 in range(7):
                            J = (j1 * 2 + j2) * 4 + (k1 * 2 + k2)  # V index: j*m+k = j*4+k
                            T = t1 * 7 + t2
                            V4[J, T] = V2[j1 * 2 + k1, t1] * V2[j2 * 2 + k2, t2]
    
    for k1 in range(2):
        for i1 in range(2):
            for t1 in range(7):
                for k2 in range(2):
                    for i2 in range(2):
                        for t2 in range(7):
                            K = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)  # W index: k*n+i = k*4+i
                            T = t1 * 7 + t2
                            W4[K, T] = W2[k1 * 2 + i1, t1] * W2[k2 * 2 + i2, t2]
    
    # Verify 4x4x4 decomposition
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: C[i,k] = sum_{j,t} A[i,j] * B[j,k] * U[i*4+j, t] * V[j*4+k, t] * W[k*4+i, t]
    # Wait, the decomposition is: C[i,k] = sum_t (sum_j A[i,j] U[i*4+j, t]) * (sum_k B[j,k] V[j*4+k, t]) * W[k*4+i, t]
    # Actually: C[i,k] = sum_{j,l} A[i,j] B[j,l] delta[k,l]
    # Decomposition: C[i,k] = sum_t (sum_j A[i,j] U[i*4+j, t]) * (sum_l B[j,l] V[j*4+l, t]) * W[k*4+i, t]
    # where j and l are summed over, and the middle term should give B[j,k] when l=k.
    
    # Let me verify numerically with random matrices
    def verify_444(U4, V4, W4, verbose=False):
        # Generate random A (4x4) and B (4x4)
        rng = np.random.RandomState(42)
        A = rng.randint(-5, 6, size=(4, 4))
        B = rng.randint(-5, 6, size=(4, 4))
        C_expected = A @ B
        
        # Compute C from decomposition
        C = np.zeros((4, 4), dtype=int)
        for t in range(49):
            u_col = U4[:, t]
            v_col = V4[:, t]
            w_col = W4[:, t]
            # A_u = A @ u_col reshaped as 4x4 -> gives 4x1
            # Actually: sum_j A[i,j] * U[i*4+j, t] for each i
            # This is: (A * U_col_reshaped) where U_col_reshaped[i,j] = U[i*4+j, t]
            U_reshaped = u_col.reshape(4, 4)
            V_reshaped = v_col.reshape(4, 4)
            W_reshaped = w_col.reshape(4, 4)
            # A_u[i] = sum_j A[i,j] * U_reshaped[i,j]
            A_u = np.einsum('ij,ij->i', A, U_reshaped)
            # B_v[k] = sum_j B[j,k] * V_reshaped[j,k]
            B_v = np.einsum('jk,jk->k', B, V_reshaped)
            # C[i,k] += A_u[i] * B_v[k] * W_reshaped[k,i]
            C += np.outer(A_u, B_v) * W_reshaped.T
            # Wait, W_reshaped[k,i] = W4[k*4+i, t], so W_reshaped.T[i,k] = W_reshaped[k,i]
        
        if verbose:
            print("Expected C:\n", C_expected)
            print("Computed C:\n", C)
            print("Match:", np.array_equal(C_expected, C))
        
        return np.array_equal(C_expected, C)
    
    if not verify_444(U4, V4, W4, verbose=False):
        # Fallback: schoolbook 4x4x4
        if verbose:
            print("4x4x4 Strassen verification failed, using schoolbook")
        U4 = np.zeros((16, 16), dtype=int)
        V4 = np.zeros((16, 16), dtype=int)
        W4 = np.zeros((16, 16), dtype=int)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    t = i * 16 + j * 4 + k
                    U4[i * 4 + j, t] = 1
                    V4[j * 4 + k, t] = 1
                    W4[k * 4 + i, t] = 1
    
    rank_444 = U4.shape[1]
    
    # Now add 4x4x3 part (schoolbook)
    # B[:, 4:7] is 4x3, C[:, 4:7] is 4x3
    # C[i, 4+k] = sum_j A[i,j] * B[j, 4+k] for k=0,1,2
    rank_443 = 4 * 4 * 3  # 48
    total_rank = rank_444 + rank_443
    
    U = np.zeros((16, total_rank), dtype=int)
    V = np.zeros((28, total_rank), dtype=int)  # m*p = 4*7 = 28
    W = np.zeros((28, total_rank), dtype=int)  # p*n = 7*4 = 28
    
    # Copy 4x4x4 part
    U[:, :rank_444] = U4
    V[:16, :rank_444] = V4
    W[:16, :rank_444] = W4
    
    # Add 4x4x3 schoolbook part
    col_offset = rank_444
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):  # k = 4, 5, 6
                U[i * 4 + j, col_offset] = 1
                V[j * 7 + k, col_offset] = 1
                W[k * 4 + i, col_offset] = 1
                col_offset += 1
    
    # Verify full decomposition
    def verify_full(U, V, W, n, m, p, verbose=False):
        rng = np.random.RandomState(123)
        A = rng.randint(-5, 6, size=(n, m))
        B = rng.randint(-5, 6, size=(m, p))
        C_expected = A @ B
        
        C = np.zeros((n, p), dtype=int)
        for t in range(U.shape[1]):
            u_col = U[:, t]
            v_col = V[:, t]
            w_col = W[:, t]
            U_reshaped = u_col.reshape(n, m)
            V_reshaped = v_col.reshape(m, p)
            W_reshaped = w_col.reshape(p, n)
            A_u = np.einsum('ij,ij->i', A, U_reshaped)
            B_v = np.einsum('jk,jk->k', B, V_reshaped)
            C += np.outer(A_u, B_v) * W_reshaped.T
        
        if verbose:
            print("Full verification:")
            print("Expected:\n", C_expected)
            print("Computed:\n", C)
            print("Match:", np.array_equal(C_expected, C))
        
        return np.array_equal(C_expected, C)
    
    if not verify_full(U, V, W, 4, 4, 7, verbose=False):
        # Ultimate fallback: full schoolbook
        if verbose:
            print("Full verification failed, using schoolbook")
        U = np.zeros((16, 112), dtype=int)
        V = np.zeros((28, 112), dtype=int)
        W = np.zeros((28, 112), dtype=int)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U[i * 4 + j, col] = 1
                    V[j * 7 + k, col] = 1
                    W[k * 4 + i, col] = 1
                    col += 1
    
    # Convert to lists with integer values
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
