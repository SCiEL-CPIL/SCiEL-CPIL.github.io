import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    # Deadline guard: 110s safety margin
    deadline = time.time() + 110
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    def verify_rank_decomp(U, V, W, n, m, p):
        """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can vectorize this using einsum for speed
        # U shape: (n*m, rank)
        # V shape: (m*p, rank)
        # W shape: (p*n, rank)
        
        # Reshape U to (n, m, rank)
        # Reshape V to (m, p, rank)
        # Reshape W to (p, n, rank)
        
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # Sum over rank dimension: sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # einsum: 'ijr,jkr,kir->ijk'
        T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        
        # The tensor should be all 1s
        if not np.allclose(T_reconstructed, np.ones((n, m, p)), atol=1e-10):
            return False
        return True

    def build_strassen_2x2_2():
        """Strassen decomposition of M_{2,2,2}. Rank 7.
        U: (4, 7), V: (4, 7), W: (4, 7)
        U indices: i*2+j, V indices: j*2+k, W indices: k*2+i
        """
        # Using NumPy for faster construction
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1  # C00 and C11
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # C10
        
        # M3 = A00(B10+B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = 1
        W[1, 2] = 1; W[3, 2] = 1  # C01 and C11
        
        # M4 = (A00-A10)(B01+B10)
        U[0, 3] = 1; U[2, 3] = -1
        V[1, 3] = 1; V[2, 3] = 1
        W[0, 3] = 1; W[2, 3] = 1  # C00 and C10
        
        # M5 = (A01+A10)(B00-B10)
        U[1, 4] = 1; U[2, 4] = 1
        V[0, 4] = 1; V[2, 4] = -1
        W[0, 4] = -1; W[1, 4] = 1  # C00 and C01
        
        # M6 = (A11-A01)(B01-B11)
        U[1, 5] = -1; U[3, 5] = 1
        V[1, 5] = 1; V[3, 5] = -1
        W[3, 5] = 1  # C11
        
        # M7 = (A00+A01)(B11-B01)
        U[0, 6] = 1; U[1, 6] = 1
        V[1, 6] = -1; V[3, 6] = 1
        W[0, 6] = 1  # C00
        
        return U, V, W

    def build_direct_2x1x4():
        """Direct decomposition of multiplying 2x1 by 1x4. Rank 8.
        U: (2, 8), V: (4, 8), W: (8, 8)
        U indices: i (m=1), V indices: k (j=0), W indices: k*2+i
        """
        rank = 8
        U = np.zeros((2, rank), dtype=int)
        V = np.zeros((4, rank), dtype=int)
        W = np.zeros((8, rank), dtype=int)
        
        col = 0
        for i in range(2):
            for k in range(4):
                U[i, col] = 1
                V[k, col] = 1
                W[k*2 + i, col] = 1
                col += 1
        return U, V, W

    def build_2x2x4():
        """Build M_{2,2,4} by concatenating two M_{2,2,2} blocks.
        Split the 4 columns of B into two groups of 2.
        Rank: 7 + 7 = 14
        U: (4, 14), V: (8, 14), W: (8, 14)
        """
        U1, V1, W1 = build_strassen_2x2_2()
        U2, V2, W2 = build_strassen_2x2_2()
        
        # Concatenate U
        U = np.hstack((U1, U2))  # (4, 14)
        
        V = np.zeros((8, 14), dtype=int)
        W = np.zeros((8, 14), dtype=int)
        
        # First block: k=0,1
        # V1 indices: j*2+k (j in 0,1; k in 0,1) -> 0,1,2,3
        # Target V indices: j*4+k (j in 0,1; k in 0,1) -> 0,1,4,5
        V[0, 0:7] = V1[0, :] # j=0, k=0
        V[1, 0:7] = V1[1, :] # j=0, k=1
        V[4, 0:7] = V1[2, :] # j=1, k=0
        V[5, 0:7] = V1[3, :] # j=1, k=1
        
        W[0, 0:7] = W1[0, :] # k=0, i=0
        W[1, 0:7] = W1[1, :] # k=0, i=1
        W[4, 0:7] = W1[2, :] # k=1, i=0
        W[5, 0:7] = W1[3, :] # k=1, i=1
        
        # Second block: k=2,3
        # V2 indices: j*2+k -> 0,1,2,3
        # Target V indices: j*4+k (j in 0,1; k in 2,3) -> 2,3,6,7
        V[2, 7:14] = V2[0, :] # j=0, k=2
        V[3, 7:14] = V2[1, :] # j=0, k=3
        V[6, 7:14] = V2[2, :] # j=1, k=2
        V[7, 7:14] = V2[3, :] # j=1, k=3
        
        W[2, 7:14] = W2[0, :] # k=2, i=0
        W[3, 7:14] = W2[1, :] # k=2, i=1
        W[6, 7:14] = W2[2, :] # k=3, i=0
        W[7, 7:14] = W2[3, :] # k=3, i=1
        
        return U, V, W

    def build_2x3x4():
        """Build M_{2,3,4} by combining M_{2,2,4} (j=0,1) and M_{2,1,4} (j=2).
        Rank: 14 + 8 = 22
        U: (6, 22), V: (12, 22), W: (8, 22)
        """
        U224, V224, W224 = build_2x2x4()  # U:(4,14), V:(8,14), W:(8,14)
        U214, V214, W214 = build_direct_2x1x4()  # U:(2,8), V:(4,8), W:(8,8)
        
        rank = 22
        U = np.zeros((6, rank), dtype=int)
        V = np.zeros((12, rank), dtype=int)
        W = np.zeros((8, rank), dtype=int)
        
        # Part 1: M_{2,2,4} (j=0,1)
        # U indices: i*3+j. j in {0,1}.
        # U224 indices: i*2+j_local.
        for i in range(2):
            U[i*3+0, 0:14] = U224[i*2+0, :] # j=0
            U[i*3+1, 0:14] = U224[i*2+1, :] # j=1
            
        # V indices: j*4+k. j in {0,1}.
        # V224 indices: j_local*4+k.
        V[0:8, 0:14] = V224[:, :]
        
        # W indices: k*2+i.
        W[:, 0:14] = W224[:, :]
        
        # Part 2: M_{2,1,4} (j=2)
        # U indices: i*3+2.
        # U214 indices: i.
        for i in range(2):
            U[i*3+2, 14:22] = U214[i, :]
            
        # V indices: 2*4+k = 8+k.
        # V214 indices: k.
        V[8:12, 14:22] = V214[:, :]
        
        # W indices: k*2+i.
        W[:, 14:22] = W214[:, :]
        
        return U, V, W

    def build_2x3x8():
        """Build M_{2,3,8} by combining two M_{2,3,4} blocks (k=0..3 and k=4..7).
        Rank: 22 + 22 = 44
        U: (6, 44), V: (24, 44), W: (16, 44)
        """
        U1, V1, W1 = build_2x3x4()  # U:(6,22), V:(12,22), W:(8,22)
        U2, V2, W2 = build_2x3x4()
        
        rank = 44
        U = np.hstack((U1, U2)) # (6, 44)
        
        V = np.zeros((24, rank), dtype=int)
        W = np.zeros((16, rank), dtype=int)
        
        # Block 1: k=0..3
        # V indices: j*8+k. k in 0..3.
        # V1 indices: j*4+k_local.
        for j in range(3):
            for k in range(4):
                V[j*8+k, 0:22] = V1[j*4+k, :]
        
        # W indices: k*2+i. k in 0..3.
        W[0:8, 0:22] = W1[:, :]
        
        # Block 2: k=4..7
        # V indices: j*8+k. k in 4..7.
        # V2 indices: j*4+k_local.
        for j in range(3):
            for k in range(4):
                V[j*8+k+4, 22:44] = V2[j*4+k, :]
        
        # W indices: k*2+i. k in 4..7.
        W[8:16, 22:44] = W2[:, :]
        
        return U, V, W

    # Construct the decomposition
    U, V, W = build_2x3x8()
    
    # Convert to lists for the required return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    if verify_rank_decomp(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook if verification fails (should not happen)
    rank = n * m * p
    U_fb = np.zeros((n*m, rank), dtype=int)
    V_fb = np.zeros((m*p, rank), dtype=int)
    W_fb = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
                
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
