"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    Uses block decomposition with Strassen's algorithm on 2x2x2 sub-blocks.
    Target rank: 46.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # Total rank is sum of ranks of sub-blocks
    # A1 (2x2) * B1 (2x5): Rank 18 (7+7+4)
    # A2 (2x2) * B2 (2x5): Rank 18 (7+7+4)
    # A3 (2x1) * B3 (1x5): Rank 10
    # Total: 46
    
    R = 46
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    col = 0
    
    # Helper to add a sub-decomposition
    def add_subblock(j_offset_A, j_offset_B, k_offset_B, U_loc, V_loc, W_loc, i_offset=0):
        nonlocal col
        # U_loc shape: (n_loc * m_loc, R_loc)
        # V_loc shape: (m_loc * p_loc, R_loc)
        # W_loc shape: (p_loc * n_loc, R_loc)
        
        n_loc = n  # always 2
        m_loc = U_loc.shape[0] // n_loc
        p_loc = W_loc.shape[0] // n_loc
        
        for r in range(U_loc.shape[1]):
            # Process U
            for a_loc in range(U_loc.shape[0]):
                val = U_loc[a_loc, r]
                if val != 0:
                    i_loc = a_loc // m_loc
                    j_loc = a_loc % m_loc
                    i_glob = i_loc + i_offset
                    j_glob = j_loc + j_offset_A
                    a_glob = i_glob * m + j_glob
                    U[a_glob, col] += val
            
            # Process V
            for b_loc in range(V_loc.shape[0]):
                val = V_loc[b_loc, r]
                if val != 0:
                    j_loc = b_loc // p_loc
                    k_loc = b_loc % p_loc
                    j_glob = j_loc + j_offset_B
                    k_glob = k_loc + k_offset_B
                    b_glob = j_glob * p + k_glob
                    V[b_glob, col] += val
            
            # Process W
            for c_loc in range(W_loc.shape[0]):
                val = W_loc[c_loc, r]
                if val != 0:
                    k_loc = c_loc // n_loc
                    i_loc = c_loc % n_loc
                    k_glob = k_loc + k_offset_B
                    i_glob = i_loc + i_offset
                    c_glob = k_glob * n + i_glob
                    W[c_glob, col] += val
            
            col += 1
    
    # Define local decompositions
    
    def strassen_222():
        # Returns U, V, W for 2x2x2
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen's 7 products
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Indices:
        # A: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[3, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[0, 5] = -1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[3, 6] = 1; W[1, 6] = -1
        
        return U, V, W

    def schoolbook_221():
        # Returns U, V, W for 2x2x1
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # C00 = A00*B00 + A01*B10
        # C10 = A10*B00 + A11*B10
        
        # Term 1: A00 B00
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: A01 B10
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        # Term 3: A10 B00
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        # Term 4: A11 B10
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    def schoolbook_215():
        # Returns U, V, W for 2x1x5
        U = np.zeros((2, 10))
        V = np.zeros((5, 10))
        W = np.zeros((10, 10))
        
        # C_i,k = A_i0 * B_0k
        # Rank 10 decomposition:
        # For each i, k, we need A_i0 * B_0k.
        # This is a rank 1 tensor for each (i,k) pair?
        # No, we can share.
        # T_{i,k,c} = 1 if c = 5i+k? No, trace convention c = k*2+i.
        # Tensor T_{i, 0, k} = 1.
        # We need U_i,r V_0,r W_k,r = T_{i,k}.
        # Since T_{i,k} = 1 for all i,k, we can use rank 1?
        # No, the indices i and k are independent.
        # We need 10 terms.
        
        idx = 0
        for i in range(2):
            for k in range(5):
                U[i, idx] = 1
                V[k, idx] = 1
                c_loc = k * 2 + i
                W[c_loc, idx] = 1
                idx += 1
                
        return U, V, W

    # Construct Block 1: A1 (j=0,1) * B1 (j=0,1)
    # B1 columns: k=0,1 (2x2), k=2,3 (2x2), k=4 (2x1)
    
    # B1a: 2x2x2
    U1, V1, W1 = strassen_222()
    add_subblock(0, 0, 0, U1, V1, W1)
    
    # B1b: 2x2x2
    U1, V1, W1 = strassen_222()
    add_subblock(0, 0, 2, U1, V1, W1)
    
    # B1c: 2x2x1
    U1, V1, W1 = schoolbook_221()
    add_subblock(0, 0, 4, U1, V1, W1)
    
    # Construct Block 2: A2 (j=2,3) * B2 (j=2,3)
    # B2 columns: k=0,1 (2x2), k=2,3 (2x2), k=4 (2x1)
    
    # B2a: 2x2x2
    U1, V1, W1 = strassen_222()
    add_subblock(2, 2, 0, U1, V1, W1)
    
    # B2b: 2x2x2
    U1, V1, W1 = strassen_222()
    add_subblock(2, 2, 2, U1, V1, W1)
    
    # B2c: 2x2x1
    U1, V1, W1 = schoolbook_221()
    add_subblock(2, 2, 4, U1, V1, W1)
    
    # Construct Block 3: A3 (j=4) * B3 (j=4)
    # B3 is 1x5
    
    # B3: 2x1x5
    U1, V1, W1 = schoolbook_215()
    add_subblock(4, 4, 0, U1, V1, W1)
    
    # Verify
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    Rec = np.einsum('ak,bk,ck->abc', U, V, W)
    err = np.max(np.abs(T - Rec))
    
    if err > 1e-10:
        # Fallback to schoolbook
        U = np.zeros((n*m, n*m*p)); V = np.zeros((m*p, n*m*p)); W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col]=1.0; V[j*p+k, col]=1.0; W[k*n+i, col]=1.0; col+=1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
