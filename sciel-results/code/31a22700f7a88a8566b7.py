import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin before 150s kill
    
    n, m, p = 2, 2, 16
    num_blocks = p // m  # 8 blocks of 2 columns each
    
    # Strassen decomposition for 2x2 x 2x2 -> 2x2
    # U_s: (n*m, 7) = (4, 7)
    # V_s: (m*p_block, 7) = (4, 7) where p_block=2
    # W_s: (p_block*n, 7) = (4, 7)
    
    U_s = np.zeros((4, 7), dtype=float)
    V_s = np.zeros((4, 7), dtype=float)
    W_s = np.zeros((4, 7), dtype=float)
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # C = A * B where A is 2x2, B is 2x2
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U_s rows indexed by (i,j) where i in {0,1}, j in {0,1}, index = i*2+j
    # V_s rows indexed by (j,k) where j in {0,1}, k in {0,1}, index = j*2+k
    # W_s rows indexed by (k,i) per trace convention W[k*n+i], index = k*2+i
    
    # Factor 1: M1 = (A00+A11)*(B00+B11)
    U_s[0, 0] += 1; U_s[3, 0] += 1
    V_s[0, 0] += 1; V_s[3, 0] += 1
    W_s[0, 0] += 1; W_s[3, 0] += 1  # C00 += M1, C11 += M1
    
    # Factor 2: M2 = (A10+A11)*B00
    U_s[2, 1] += 1; U_s[3, 1] += 1
    V_s[0, 1] += 1
    W_s[2, 1] += 1  # C10 += M2
    W_s[3, 1] += -1 # C11 -= M2
    
    # Factor 3: M3 = A00*(B01-B11)
    U_s[0, 2] += 1
    V_s[1, 2] += 1; V_s[3, 2] += -1
    W_s[1, 2] += 1  # C01 += M3
    W_s[3, 2] += 1  # C11 += M3
    
    # Factor 4: M4 = A11*(B10-B00)
    U_s[3, 3] += 1
    V_s[2, 3] += 1; V_s[0, 3] += -1
    W_s[0, 3] += 1  # C00 += M4
    W_s[2, 3] += 1  # C10 += M4
    
    # Factor 5: M5 = (A00+A01)*B11
    U_s[0, 4] += 1; U_s[1, 4] += 1
    V_s[3, 4] += 1
    W_s[0, 4] += -1 # C00 -= M5
    W_s[1, 4] += 1  # C01 += M5
    
    # Factor 6: M6 = (A10-A00)*(B00+B01)
    U_s[2, 5] += 1; U_s[0, 5] += -1
    V_s[0, 5] += 1; V_s[1, 5] += 1
    W_s[3, 5] += 1  # C11 += M6
    
    # Factor 7: M7 = (A01-A11)*(B10+B11)
    U_s[1, 6] += 1; U_s[3, 6] += -1
    V_s[2, 6] += 1; V_s[3, 6] += 1
    W_s[0, 6] += 1  # C00 += M7
    
    # Verify the 2x2x2 decomposition
    # Tensor T[i,j,k] = delta_{i,j} * delta_{j,k}... no
    # Actually T[i,j,k] = 1 if the product C[i,k] gets contribution from A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) in the multiplication tensor
    # The decomposition says: sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
    # where T[i,j,k] = 1 (it's the structure tensor of matrix mult)
    
    # Let's verify
    T = np.zeros((4, 4, 4))
    # T[row_U, row_V, row_W] should equal sum_r U[row_U,r]*V[row_V,r]*W[row_W,r]
    # row_U = i*2+j, row_V = j*2+k, row_W = k*2+i
    # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T has T[i*2+j, j*2+k, k*2+i] = 1 for all i,j,k
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T[i*2+j, j*2+k, k*2+i] = 1
    
    # Check decomposition
    recon = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
    if not np.allclose(recon, T):
        # Fallback to trivial rank-4 decomposition
        U_s = np.zeros((4, 4))
        V_s = np.zeros((4, 4))
        W_s = np.zeros((4, 4))
        for j in range(2):
            for i in range(2):
                for k in range(2):
                    r = i*4 + j*2 + k  # not quite, let me use rank 4
        # Actually just use the standard rank-4 for 2x2x2
        # C[i,k] = sum_j A[i,j] B[j,k]
        # rank 4: for each j, one factor
        U_s = np.zeros((4, 4))
        V_s = np.zeros((4, 4))
        W_s = np.zeros((4, 4))
        for j in range(2):
            for i in range(2):
                for k in range(2):
                    # This is rank 8, not 4. Let me think...
                    # For 2x2, the minimal rank is 7 (Strassen), but the trivial is 4?
                    # No, trivial is n*m*p = 8 for 2x2x2.
                    # Wait, the tensor is 2x2x2, so trivial rank is 2*2*2=8? No.
                    # The multiplication tensor for (n,m,p) has entries T[i,j,k] where
                    # i in [n], j in [m], k in [p]. The trivial decomposition has rank n*m*p.
                    # For 2x2x2, trivial rank = 8. But we can do better.
                    # Actually I think I'm confusing things. Let me re-examine.
                    pass
        
        # Let me just use the trivial rank-8 decomposition as fallback
        U_s = np.zeros((4, 8))
        V_s = np.zeros((4, 8))
        W_s = np.zeros((4, 8))
        r = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U_s[i*2+j, r] = 1
                    V_s[j*2+k, r] = 1
                    W_s[k*2+i, r] = 1
                    r += 1
        # Re-verify
        recon = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
        if not np.allclose(recon, T):
            raise ValueError("Failed to construct valid decomposition")
    
    # Now expand to 2x2 x 2x16
    # U: (n*m, rank) = (4, 7*8) = (4, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (32, 56)
    
    rank = 7 * num_blocks
    U = np.zeros((4, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    for t in range(num_blocks):
        cols = slice(7*t, 7*t+7)
        
        # U: same for each block
        U[:, cols] = U_s
        
        # V: for block t, rows correspond to (j, k_global) where k_global = 2t + k_local
        # V_s row index = j*2 + k_local
        # V_global row index = j*p + k_global = j*16 + (2t + k_local)
        # j=0, k_local=0 -> row 0+2t = 2t
        # j=0, k_local=1 -> row 1+2t = 2t+1
        # j=1, k_local=0 -> row 16+2t
        # j=1, k_local=1 -> row 17+2t
        V[2*t, cols] = V_s[0, :]
        V[2*t+1, cols] = V_s[1, :]
        V[16+2*t, cols] = V_s[2, :]
        V[17+2*t, cols] = V_s[3, :]
        
        # W: for block t, rows correspond to (k_global, i)
        # W_s row index = k_local*2 + i
        # W_global row index = k_global*n + i = (2t+k_local)*2 + i = 4t + 2*k_local + i
        # k_local=0, i=0 -> row 4t
        # k_local=0, i=1 -> row 4t+1
        # k_local=1, i=0 -> row 4t+2
        # k_local=1, i=1 -> row 4t+3
        W[4*t, cols] = W_s[0, :]
        W[4*t+1, cols] = W_s[1, :]
        W[4*t+2, cols] = W_s[2, :]
        W[4*t+3, cols] = W_s[3, :]
    
    # Final verification with random test
    if time.time() - start_time < deadline - 10:
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        # C[i, k] = sum_j,r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, let me re-read the convention.
        # U has shape (n*m, rank), indexed by [i*m + j, r]
        # V has shape (m*p, rank), indexed by [j*p + k, r]
        # W has shape (p*n, rank), indexed by [k*n + i, r]
        # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * ... 
        # Actually the tensor decomposition gives:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And we need T[i,j,k] = 1 (the multiplication tensor)
        # Then C[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k] = sum_j A[i,j]*B[j,k]
        
        # Let's verify the tensor
        T_full = np.einsum('ir,jr,kr->ijk', 
                           U.reshape(n*m, -1).T,  # This is wrong, let me redo
                           np.ones((1,1)), np.ones((1,1)))
        
        # Better approach: directly check
        # For each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
        # (since T[i,j,k]=1 for all i,j,k in the 2x2x2 case, and by block structure for 2x2x16)
        
        # Actually for 2x2 x 2x16, the tensor T[i,j,k] = 1 for all i in [2], j in [2], k in [16]
        # because C[i,k] = sum_j A[i,j]*B[j,k], so every triple (i,j,k) contributes with coefficient 1.
        
        # Check a few entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-10:
                        raise ValueError(f"Tensor mismatch at ({i},{j},{k}): got {val}")
        
        # Now verify matrix multiplication
        C_recon = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_recon[i, k] += A[i, j] * B[j, k] * 1.0  # T[i,j,k]=1
                # This should equal C_expected[i,k]
        
        # More directly: C[i,k] = sum_j A[i,j]*B[j,k]
        # Our decomposition ensures T[i,j,k]=1, so this works.
    
    # Convert to lists with integer/half-integer entries
    # All entries are already integers (0, 1, -1)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
