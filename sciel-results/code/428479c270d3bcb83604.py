import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 Strassen.
        U shape (4, 7), V shape (4, 7), W shape (4, 7).
        Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1
        
        # M4 = A11(B00-B10)
        U[3, 3] = 1
        V[0, 3] = 1; V[2, 3] = -1
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

    def outer_2x1x2():
        """
        Returns U, V, W for 2x1x2 outer product.
        U shape (2, 1), V shape (2, 1), W shape (4, 1).
        Indices: U[i*1+j, r] -> U[i, r]
        V[j*2+k, r] -> V[k, r] (j=0)
        W[k*2+i, r] -> W[k*2+i, r]
        """
        U = np.zeros((2, 1))
        V = np.zeros((2, 1))
        W = np.zeros((4, 1))
        
        # U[i, 0] = A_i
        U[0, 0] = 1 # A_0
        U[1, 0] = 1 # A_1
        
        # V[k, 0] = B_k (j=0)
        V[0, 0] = 1 # B_0
        V[1, 0] = 1 # B_1
        
        # W[k*2+i, 0] = 1
        W[0, 0] = 1 # k=0, i=0
        W[1, 0] = 1 # k=0, i=1
        W[2, 0] = 1 # k=1, i=0
        W[3, 0] = 1 # k=1, i=1
        
        return U, V, W

    def block_2x3x2():
        """
        Returns U, V, W for 2x3x2.
        Split m=3 into 1+2.
        Part 1: 2x1x2 (Rank 1).
        Part 2: 2x2x2 (Rank 7).
        Total Rank 8.
        U shape (6, 8), V shape (6, 8), W shape (4, 8).
        U indices: i*3+j.
        V indices: j*2+k.
        W indices: k*2+i.
        """
        U_parts = []
        V_parts = []
        W_parts = []
        
        # Part 1: j=0. 2x1x2.
        U1, V1, W1 = outer_2x1x2()
        # Map to 2x3x2 indices.
        U1_full = np.zeros((6, 1))
        V1_full = np.zeros((6, 1))
        W1_full = np.zeros((4, 1))
        
        # U1[i, 0] -> U1_full[i*3+0, 0]
        U1_full[0, 0] = U1[0, 0]
        U1_full[3, 0] = U1[1, 0]
        
        # V1[k, 0] (j=0) -> V1_full[0*2+k, 0]
        V1_full[0, 0] = V1[0, 0]
        V1_full[1, 0] = V1[1, 0]
        
        # W1[k*2+i, 0] -> W1_full[k*2+i, 0]
        W1_full[:, 0] = W1[:, 0]
        
        U_parts.append(U1_full)
        V_parts.append(V1_full)
        W_parts.append(W1_full)
        
        # Part 2: j=1,2. 2x2x2.
        U2, V2, W2 = strassen_2x2x2()
        U2_full = np.zeros((6, 7))
        V2_full = np.zeros((6, 7))
        W2_full = np.zeros((4, 7))
        
        # U2[i*2+j, r] -> U2_full[i*3+(j+1), r]
        for i in range(2):
            for j in range(2):
                U2_full[i*3 + (j+1), :] = U2[i*2 + j, :]
        
        # V2[j*2+k, r] -> V2_full[(j+1)*2+k, r]
        for j in range(2):
            for k in range(2):
                V2_full[(j+1)*2 + k, :] = V2[j*2 + k, :]
        
        # W2[k*2+i, r] -> W2_full[k*2+i, r]
        W2_full[:, :] = W2[:, :]
        
        U_parts.append(U2_full)
        V_parts.append(V2_full)
        W_parts.append(W2_full)
        
        U = np.hstack(U_parts)
        V = np.hstack(V_parts)
        W = np.hstack(W_parts)
        
        return U, V, W

    # Global dimensions
    n, m, p = 4, 5, 6
    rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Blocks:
    # A rows: 0-1, 2-3
    # B cols: 0-1, 2-3, 4-5
    # Middle: 0-1, 2-4
    
    # C blocks:
    # (0,0) to (1,1)
    # (0,0) to (1,3)
    # (0,0) to (1,5)
    # (2,0) to (3,1)
    # (2,0) to (3,3)
    # (2,0) to (3,5)
    
    def add_block(i_start, j_start, k_start, n_b, m_b, p_b, U_loc, V_loc, W_loc):
        nonlocal rank
        U_glob = np.zeros((n*m, U_loc.shape[1]))
        V_glob = np.zeros((m*p, V_loc.shape[1]))
        W_glob = np.zeros((p*n, W_loc.shape[1]))
        
        for r in range(U_loc.shape[1]):
            for i in range(n_b):
                for j in range(m_b):
                    U_glob[(i_start+i)*m + (j_start+j), r] = U_loc[i*m_b+j, r]
            
            for j in range(m_b):
                for k in range(p_b):
                    V_glob[(j_start+j)*p + (k_start+k), r] = V_loc[j*p_b+k, r]
            
            for k in range(p_b):
                for i in range(n_b):
                    W_glob[(k_start+k)*n + (i_start+i), r] = W_loc[k*n_b+i, r]
        
        U_cols.append(U_glob)
        V_cols.append(V_glob)
        W_cols.append(W_glob)
        rank += U_loc.shape[1]

    # Precompute local blocks
    U_s, V_s, W_s = strassen_2x2x2()
    U_b, V_b, W_b = block_2x3x2()
    
    # 6 blocks
    # i_start: 0 or 2
    # k_start: 0, 2, 4
    
    for i_start in [0, 2]:
        for k_start in [0, 2, 4]:
            # Term 1: A[i_start:i_start+2, 0:2] * B[0:2, k_start:k_start+2]
            # 2x2x2
            add_block(i_start, 0, k_start, 2, 2, 2, U_s, V_s, W_s)
            
            # Term 2: A[i_start:i_start+2, 2:5] * B[2:5, k_start:k_start+2]
            # 2x3x2
            add_block(i_start, 2, k_start, 2, 3, 2, U_b, V_b, W_b)
            
    U = np.hstack(U_cols)
    V = np.hstack(V_cols)
    W = np.hstack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
