"""Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 125
    
    # Schoolbook decomposition (fallback)
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_schoolbook
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify that U, V, W decompose the 5x5 matrix multiplication tensor."""
        r = U_mat.shape[1]
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k])
        # The tensor coefficient T[i,j,k] = 1 for all i,j,k
        # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        
        # We can verify by checking that for each (i,j,k), the sum equals 1
        # Using einsum for efficiency
        # T_reconstructed[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_3d = U_mat.reshape(n, m, r)
        V_3d = V_mat.reshape(m, p, r)
        W_3d = W_mat.reshape(p, n, r)
        
        # T_reconstructed[i,j,k] = sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
        # Note: W is indexed as [k*n+i, c], so W_3d[k,i,c]
        
        # Use einsum: sum over c
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        # Check if T_rec is all ones
        if not np.allclose(T_rec, 1.0, atol=tol, rtol=tol):
            return False
        return True
    
    # Strategy: Use block decomposition with Strassen for the 4x4x4 core
    # Split 5x5x5 into blocks of size 4x4x4 plus boundary terms
    
    # Approach: 
    # We decompose the 5x5 multiplication using the following block structure:
    # A = [[A11, A12], [A21, A22]] where A11 is 4x4, A12 is 4x1, A21 is 1x4, A22 is 1x1
    # B = [[B11, B12], [B21, B22]] where B11 is 4x4, B12 is 4x1, B21 is 1x4, B22 is 1x1
    # C = [[C11, C12], [C21, C22]] where C11 is 4x4, C12 is 4x1, C21 is 1x4, C22 is 1x1
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # We can use Strassen for the 4x4x4 multiplications and schoolbook for the rest.
    
    # First, let's implement Strassen decomposition for 2x2 matrices
    # M(2) = 7 using Strassen
    def strassen_2x2_decomp():
        """Return U, V, W for 2x2 matrix multiplication using Strassen's algorithm."""
        # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
        # U[i*2+j, c] = coefficient for A[i,j] in the c-th Strassen product
        # V[i*2+j, c] = coefficient for B[i,j] in the c-th Strassen product
        # W[i*2+j, c] = coefficient for C[i,j] in the c-th Strassen product
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen's algorithm for 2x2:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B11 - B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Index: A11=0, A12=1, A21=2, A22=3
        # B11=0, B12=1, B21=2, B22=3
        # C11=0, C12=1, C21=2, C22=3
        
        # M1: (A11 + A22)(B11 + B22)
        # A indices: 0, 3 -> U[0,0]=1, U[3,0]=1
        # B indices: 0, 3 -> V[0,0]=1, V[3,0]=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # C11 = M1..., C22 = M1...
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: (A21 + A22)B11
        # A indices: 2, 3 -> U[2,1]=1, U[3,1]=1
        # B indices: 0 -> V[0,1]=1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C21 = M2 + M4, C22 = -M2
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3: A11(B21 - B22)
        # A indices: 0 -> U[0,2]=1
        # B indices: 2, 3 -> V[2,2]=1, V[3,2]=-1
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        # C12 = M3 + M5, C22 = M3
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4: A22(B11 - B12)
        # A indices: 3 -> U[3,3]=1
        # B indices: 0, 1 -> V[0,3]=1, V[1,3]=-1
        U[3, 3] = 1
        V[0, 3] = 1; V[1, 3] = -1
        # C11 = -M4, C21 = M4
        W[0, 3] = -1; W[2, 3] = 1
        
        # M5: (A11 + A12)B22
        # A indices: 0, 1 -> U[0,4]=1, U[1,4]=1
        # B indices: 3 -> V[3,4]=1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C11 = M5, C12 = M5
        W[0, 4] = 1; W[1, 4] = 1
        
        # M6: (A21 - A11)(B11 + B12)
        # A indices: 2, 0 -> U[2,5]=1, U[0,5]=-1
        # B indices: 0, 1 -> V[0,6]=1, V[1,6]=1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 6] = 1; V[1, 6] = 1
        # C22 = M6
        W[3, 6] = 1
        
        # M7: (A12 - A22)(B21 + B22)
        # A indices: 1, 3 -> U[1,6]=1, U[3,6]=-1
        # B indices: 2, 3 -> V[2,6]=1, V[3,6]=1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # C11 = -M7
        W[0, 6] = -1
        
        return U, V, W
    
    # Now build the 5x5 decomposition using block structure
    # We'll use:
    # - Strassen for 4x4x4: rank 49 (7^2)
    # - Schoolbook for boundary blocks
    
    # Let's try a cleaner approach: use the fact that we can decompose
    # 5x5 = 4x4 + 1x1 boundary
    
    # Total rank estimate:
    # 4x4x4 core: 49 (Strassen)
    # A12*B21 -> C11: 4x1x4 = 16
    # A11*B12 -> C12: 4x4x1 = 16
    # A21*B11 -> C21: 1x4x4 = 16
    # A12*B22 -> C12: 4x1x1 = 4
    # A21*B12 -> C22: 1x4x1 = 4
    # A22*B21 -> C21: 1x1x4 = 4
    # A22*B22 -> C22: 1x1x1 = 1
    # Total: 49 + 16 + 16 + 16 + 4 + 4 + 4 + 1 = 110
    
    U = np.zeros((25, 110))
    V = np.zeros((25, 110))
    W = np.zeros((25, 110))
    
    col_offset = 0
    
    # 1. 4x4x4 core using Strassen (Kronecker product of 2x2 Strassen)
    # Strassen for 2x2 gives rank 7
    # For 4x4, we use Kronecker: rank 7*7 = 49
    S_U, S_V, S_W = strassen_2x2_decomp()
    
    # Kronecker product to get 4x4x4
    # U_4 = kron(S_U, S_U) but with proper index mapping
    # Actually, for 4x4 = (2x2) ⊗ (2x2), the indices are:
    # i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # U_4[i,j,c] where c = c1*7 + c2
    # U_4[2*i1+i2, 2*j1+j2, c1*7+c2] = S_U[i1, j1, c1] * S_U[i2, j2, c2]
    
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for c1 in range(7):
                    for i2 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                for c2 in range(7):
                                    i = i1 * 2 + i2
                                    j = j1 * 2 + j2
                                    k = k1 * 2 + k2
                                    c = c1 * 7 + c2
                                    
                                    # U_4[i*4+j, c] = S_U[i1*2+j1, c1] * S_U[i2*2+j2, c2]
                                    # Wait, S_U has shape (4, 7) for 2x2
                                    # S_U[idx, c] where idx = row*2 + col for 2x2
                                    # So S_U[i1*2+j1, c1] is the value for row i1, col j1
                
                                    u_val = S_U[i1 * 2 + j1, c1] * S_U[i2 * 2 + j2, c2]
                                    v_val = S_V[j1 * 2 + k1, c1] * S_V[j2 * 2 + k2, c2]
                                    w_val = S_W[k1 * 2 + i1, c1] * S_W[k2 * 2 + i2, c2]
                                    
                                    U_4[i * 4 + j, c] = u_val
                                    V_4[j * 4 + k, c] = v_val
                                    W_4[k * 4 + i, c] = w_val
    
    # Place 4x4x4 core into global U, V, W (rows 0-15)
    U[0:16, col_offset:col_offset+49] = U_4
    V[0:16, col_offset:col_offset+49] = V_4
    W[0:16, col_offset:col_offset+49] = W_4
    col_offset += 49
    
    # 2. A12*B21 -> C11: i in [0,4), j=4, k in [0,4)
    # A[i,4] * B[4,k] -> C[i,k]
    # U[i*5+4, c] = 1, V[4*5+k, c] = 1, W[k*5+i, c] = 1
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4, col_offset] = 1
            V[4 * 5 + k, col_offset] = 1
            W[k * 5 + i, col_offset] = 1
            col_offset += 1
    
    # 3. A11*B12 -> C12: i in [0,4), j in [0,4), k=4
    # A[i,j] * B[j,4] -> C[i,4]
    # U[i*5+j, c] = 1, V[j*5+4, c] = 1, W[4*5+i, c] = 1
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, col_offset] = 1
            V[j * 5 + 4, col_offset] = 1
            W[4 * 5 + i, col_offset] = 1
            col_offset += 1
    
    # 4. A21*B11 -> C21: i=4, j in [0,4), k in [0,4)
    # A[4,j] * B[j,k] -> C[4,k]
    # U[4*5+j, c] = 1, V[j*5+k, c] = 1, W[k*5+4, c] = 1
    for j in range(4):
        for k in range(4):
            U[4 * 5 + j, col_offset] = 1
            V[j * 5 + k, col_offset] = 1
            W[k * 5 + 4, col_offset] = 1
            col_offset += 1
    
    # 5. A12*B22 -> C12: i in [0,4), j=4, k=4
    # A[i,4] * B[4,4] -> C[i,4]
    # U[i*5+4, c] = 1, V[4*5+4, c] = 1, W[4*5+i, c] = 1
    for i in range(4):
        U[i * 5 + 4, col_offset] = 1
        V[4 * 5 + 4, col_offset] = 1
        W[4 * 5 + i, col_offset] = 1
        col_offset += 1
    
    # 6. A21*B12 -> C22: i=4, j in [0,4), k=4
    # A[4,j] * B[j,4] -> C[4,4]
    # U[4*5+j, c] = 1, V[j*5+4, c] = 1, W[4*5+4, c] = 1
    for j in range(4):
        U[4 * 5 + j, col_offset] = 1
        V[j * 5 + 4, col_offset] = 1
        W[4 * 5 + 4, col_offset] = 1
        col_offset += 1
    
    # 7. A22*B21 -> C21: i=4, j=4, k in [0,4)
    # A[4,4] * B[4,k] -> C[4,k]
    # U[4*5+4, c] = 1, V[4*5+k, c] = 1, W[k*5+4, c] = 1
    for k in range(4):
        U[4 * 5 + 4, col_offset] = 1
        V[4 * 5 + k, col_offset] = 1
        W[k * 5 + 4, col_offset] = 1
        col_offset += 1
    
    # 8. A22*B22 -> C22: i=4, j=4, k=4
    # A[4,4] * B[4,4] -> C[4,4]
    # U[4*5+4, c] = 1, V[4*5+4, c] = 1, W[4*5+4, c] = 1
    U[4 * 5 + 4, col_offset] = 1
    V[4 * 5 + 4, col_offset] = 1
    W[4 * 5 + 4, col_offset] = 1
    col_offset += 1
    
    # Verify the decomposition
    if verify_decomposition(U, V, W, tol=1e-9):
        best_U = U
        best_V = V
        best_W = W
        best_rank = 110
    
    # Check if we have time for further optimization
    if time.time() < DEADLINE - 10:
        # Try to verify and potentially find even lower rank
        # For now, we have a valid rank-110 decomposition
        
        pass
    
    # Return the best result
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
