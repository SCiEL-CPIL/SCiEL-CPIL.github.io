import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Strategy: Split n=3 into 1+2.
    - Term 1: 1x4x5 (rank 4, schoolbook)
    - Term 2: 2x4x5. Split m=4 into 2+2.
        - Subterm 2.1: 2x2x5. Split p=5 into 2+2+1.
            - 2x2x2 (Strassen, rank 7)
            - 2x2x2 (Strassen, rank 7)
            - 2x2x1 (Schoolbook, rank 4)
            Total: 18
        - Subterm 2.2: 2x2x5. Same as 2.1.
            Total: 18
    Total Rank = 4 + 18 + 18 = 40.
    """
    n, m, p = DIMENSIONS
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_indices, v_indices, w_indices, U_blk, V_blk, W_blk):
        """
        Adds a block decomposition.
        u_indices: list of global U indices corresponding to local rows 0..len-1 of U_blk.
        v_indices: list of global V indices corresponding to local rows 0..len-1 of V_blk.
        w_indices: list of global W indices corresponding to local rows 0..len-1 of W_blk.
        """
        rank = U_blk.shape[1]
        for r in range(rank):
            u_vec = np.zeros(n * m)
            for i, idx in enumerate(u_indices):
                u_vec[idx] = U_blk[i, r]
            U_parts.append(u_vec)
            
            v_vec = np.zeros(m * p)
            for i, idx in enumerate(v_indices):
                v_vec[idx] = V_blk[i, r]
            V_parts.append(v_vec)
            
            w_vec = np.zeros(p * n)
            for i, idx in enumerate(w_indices):
                w_vec[idx] = W_blk[i, r]
            W_parts.append(w_vec)

    # --- Helper Decompositions ---
    
    def strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7).
        Local indices:
        U: A00, A01, A10, A11
        V: B00, B01, B10, B11
        W: C00, C01, C10, C11
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
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def schoolbook_2x2x1():
        """
        Returns U, V, W for 2x2x1 matrix multiplication (rank 4).
        Local indices:
        U: A00, A01, A10, A11
        V: B0, B1
        W: C0, C1
        """
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # C0 = A00*B0 + A01*B1
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        # C1 = A10*B0 + A11*B1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    def schoolbook_1x4x5():
        """
        Returns U, V, W for 1x4x5 matrix multiplication (rank 4).
        Local indices:
        U: A0, A1, A2, A3
        V: B00, B01, B02, B03, B04
        W: C0, C1, C2, C3, C4
        """
        U = np.zeros((4, 4))
        V = np.zeros((5, 4))
        W = np.zeros((5, 4))
        
        for j in range(4):
            U[j, j] = 1
            for k in range(5):
                V[k, j] = 1
                W[k, j] = 1
        
        return U, V, W

    # --- Assembly ---
    
    # Global Index Mapping
    # A is 3x4. Flattened: A[i, j] -> i*4 + j
    # B is 4x5. Flattened: B[i, j] -> i*5 + j
    # C is 3x5. W[k*n+i] where C[k, i] is element at row i, col k.
    # Wait, standard matrix mult C = A B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The task says "output factor follows the official trace convention W[k*n+i,r]".
    # This usually means W is indexed by the output element C[i, k] at position k*n + i.
    # Let's verify: if W[k*n+i, r] corresponds to coefficient of C[i, k], then
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = C[i, k].
    # Yes, this is the standard tensor T_{i, j, k} = C_{i, k} depending on A_{i, j} and B_{j, k}.
    # So U index is i*m + j.
    # V index is j*p + k.
    # W index is k*n + i.

    # Term 1: Row 0 of A (1x4) * B (4x5) -> Row 0 of C (1x5)
    # U indices: [0, 1, 2, 3]
    # V indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # W indices: C[0, 0..4] -> k*3 + 0 for k=0..4 -> [0, 3, 6, 9, 12]
    U_blk, V_blk, W_blk = schoolbook_1x4x5()
    u_idx = [0, 1, 2, 3]
    v_idx = list(range(15))
    w_idx = [0, 3, 6, 9, 12]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Term 2: Rows 1,2 of A (2x4) * B (4x5) -> Rows 1,2 of C (2x5)
    # Split m=4 into 2+2.
    # Subterm 2.1: A[1:3, 0:2] (2x2) * B[0:2, :] (2x5)
    # Subterm 2.2: A[1:3, 2:4] (2x2) * B[2:4, :] (2x5)
    
    # Define index lists for A parts (Rows 1,2)
    # A_top (Rows 1,2; Cols 0,1): U indices [4, 5, 8, 9]
    # A_bot (Rows 1,2; Cols 2,3): U indices [6, 7, 10, 11]
    
    # Define index lists for B parts
    # B1 (Rows 0,1; Cols 0,1): V indices [0, 1, 5, 6]
    # B1 (Rows 0,1; Cols 2,3): V indices [2, 3, 7, 8]
    # B1 (Rows 0,1; Col 4):    V indices [4, 9]
    
    # B2 (Rows 2,3; Cols 0,1): V indices [10, 11, 15, 16]
    # B2 (Rows 2,3; Cols 2,3): V indices [12, 13, 17, 18]
    # B2 (Rows 2,3; Col 4):    V indices [14, 19]
    
    # Define index lists for C parts (W indices)
    # C rows 1,2; cols 0,1
    # C[1,0]->1, C[1,1]->4, C[2,0]->2, C[2,1]->5
    # C rows 1,2; cols 2,3
    # C[1,2]->7, C[1,3]->10, C[2,2]->8, C[2,3]->11
    # C rows 1,2; col 4
    # C[1,4]->13, C[2,4]->14

    # Subterm 2.1: A[1:3, 0:2] * B[0:2, :]
    # 2.1a: * B[0:2, 0:2] (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [4, 5, 8, 9]
    v_idx = [0, 1, 5, 6]
    w_idx = [1, 4, 2, 5] # C10, C11, C20, C21
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # 2.1b: * B[0:2, 2:3] (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [4, 5, 8, 9]
    v_idx = [2, 3, 7, 8]
    w_idx = [7, 10, 8, 11] # C12, C13, C22, C23
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # 2.1c: * B[0:2, 4] (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    u_idx = [4, 5, 8, 9]
    v_idx = [4, 9]
    w_idx = [13, 14] # C14, C24
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Subterm 2.2: A[1:3, 2:4] * B[2:4, :]
    # 2.2a: * B[2:4, 0:2] (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [6, 7, 10, 11]
    v_idx = [10, 11, 15, 16]
    w_idx = [1, 4, 2, 5]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # 2.2b: * B[2:4, 2:3] (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [6, 7, 10, 11]
    v_idx = [12, 13, 17, 18]
    w_idx = [7, 10, 8, 11]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # 2.2c: * B[2:4, 4] (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    u_idx = [6, 7, 10, 11]
    v_idx = [14, 19]
    w_idx = [13, 14]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)

    # Total Rank:
    # Term 1: 4
    # Term 2.1: 7 + 7 + 4 = 18
    # Term 2.2: 7 + 7 + 4 = 18
    # Total: 40
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
