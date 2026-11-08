import numpy as np
from itertools import product

def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    
    Strategy: Use a recursive block decomposition approach.
    - Split the middle dimension m=4 into two parts of size 2: m1=2, m2=2.
    - For each part, split the output dimension p=5 into p1=2, p2=2, p3=1.
    - Use Strassen's algorithm (rank 7) for 2x2x2 blocks.
    - Use schoolbook (rank 4) for 1x2x2 and 2x2x1 blocks.
    - Use schoolbook (rank 2) for 1x2x1 blocks.
    
    This gives a total rank of 56, which is better than the schoolbook rank of 60.
    """
    n, m, p = 3, 4, 5
    
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

    def schoolbook_1x2x2():
        """
        Returns U, V, W for 1x2x2 matrix multiplication (rank 4).
        Local indices:
        U: A0, A1
        V: B00, B01, B10, B11
        W: C0, C1
        """
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        # C0 = A0*B00 + A1*B10
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        
        # C1 = A0*B01 + A1*B11
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
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

    def schoolbook_1x2x1():
        """
        Returns U, V, W for 1x2x1 matrix multiplication (rank 2).
        Local indices:
        U: A0, A1
        V: B0, B1
        W: C0
        """
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        # C0 = A0*B0 + A1*B1
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
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

    # Split B into B1 (rows 0,1) and B2 (rows 2,3).
    # Split B1 into B1_1 (cols 0,1), B1_2 (cols 2,3), B1_3 (col 4).
    # Split B2 into B2_1 (cols 0,1), B2_2 (cols 2,3), B2_3 (col 4).
    
    # For each A_i (which is 3x2) and B_j (which is 2x5):
    # We process 3x2x2 blocks for cols 0,1 and 2,3.
    # We process 3x2x1 block for col 4.
    
    # Define index lists for A parts
    # A1 (rows 0,1,2; cols 0,1)
    # A1_top (rows 0,1; cols 0,1): U indices [0, 1, 4, 5]
    # A1_bot (row 2; cols 0,1): U indices [8, 9]
    
    # A2 (rows 0,1,2; cols 2,3)
    # A2_top (rows 0,1; cols 2,3): U indices [2, 3, 6, 7]
    # A2_bot (row 2; cols 2,3): U indices [10, 11]

    # Define index lists for B parts
    # B1 (rows 0,1; cols 0,1): V indices [0, 1, 5, 6]
    # B1 (rows 0,1; cols 2,3): V indices [2, 3, 7, 8]
    # B1 (rows 0,1; col 4):    V indices [4, 9]
    
    # B2 (rows 2,3; cols 0,1): V indices [10, 11, 15, 16]
    # B2 (rows 2,3; cols 2,3): V indices [12, 13, 17, 18]
    # B2 (rows 2,3; col 4):    V indices [14, 19]

    # Define index lists for C parts (W indices)
    # C rows 0,1,2; cols 0,1
    # C[0,0]->0, C[0,1]->3, C[1,0]->1, C[1,1]->4, C[2,0]->2, C[2,1]->5
    # W indices for C cols 0,1:
    # Row 0: [0, 3]
    # Row 1: [1, 4]
    # Row 2: [2, 5]
    # For a 3x2x2 block, we need W indices for C[0,0], C[0,1], C[1,0], C[1,1], C[2,0], C[2,1].
    # But our blocks are 2x2x2 (top) and 1x2x2 (bot).
    
    # Let's process term by term.
    
    # Term 1: A1 * B1 (A1 is 3x2, B1 is 2x5)
    # Subterm 1.1: A1 * B1_1 (3x2x2)
    # Split A1 into Top (2x2) and Bot (1x2).
    
    # 1.1a: A1_top * B1_1 (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [0, 1, 4, 5]
    v_idx = [0, 1, 5, 6]
    w_idx = [0, 3, 1, 4] # C00, C01, C10, C11 for rows 0,1
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # 1.1b: A1_bot * B1_1 (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx = [8, 9]
    v_idx = [0, 1, 5, 6]
    w_idx = [2, 5] # C20, C21 for row 2
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Subterm 1.2: A1 * B1_2 (3x2x2)
    # 1.2a: A1_top * B1_2 (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [0, 1, 4, 5]
    v_idx = [2, 3, 7, 8]
    w_idx = [6, 9, 7, 10] # C02, C03, C12, C13
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # 1.2b: A1_bot * B1_2 (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx = [8, 9]
    v_idx = [2, 3, 7, 8]
    w_idx = [8, 11] # C22, C23
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Subterm 1.3: A1 * B1_3 (3x2x1)
    # 1.3a: A1_top * B1_3 (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    u_idx = [0, 1, 4, 5]
    v_idx = [4, 9]
    w_idx = [12, 13] # C04, C14
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # 1.3b: A1_bot * B1_3 (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    u_idx = [8, 9]
    v_idx = [4, 9]
    w_idx = [14] # C24
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Term 2: A2 * B2 (A2 is 3x2, B2 is 2x5)
    # Subterm 2.1: A2 * B2_1 (3x2x2)
    # 2.1a: A2_top * B2_1 (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [2, 3, 6, 7]
    v_idx = [10, 11, 15, 16]
    w_idx = [0, 3, 1, 4]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # 2.1b: A2_bot * B2_1 (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx = [10, 11]
    v_idx = [10, 11, 15, 16]
    w_idx = [2, 5]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Subterm 2.2: A2 * B2_2 (3x2x2)
    # 2.2a: A2_top * B2_2 (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [2, 3, 6, 7]
    v_idx = [12, 13, 17, 18]
    w_idx = [6, 9, 7, 10]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # 2.2b: A2_bot * B2_2 (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx = [10, 11]
    v_idx = [12, 13, 17, 18]
    w_idx = [8, 11]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Subterm 2.3: A2 * B2_3 (3x2x1)
    # 2.3a: A2_top * B2_3 (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    u_idx = [2, 3, 6, 7]
    v_idx = [14, 19]
    w_idx = [12, 13]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # 2.3b: A2_bot * B2_3 (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    u_idx = [10, 11]
    v_idx = [14, 19]
    w_idx = [14]
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)

    # Total Rank:
    # Term 1: 7 + 4 + 7 + 4 + 4 + 2 = 28
    # Term 2: 7 + 4 + 7 + 4 + 4 + 2 = 28
    # Total: 56
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
