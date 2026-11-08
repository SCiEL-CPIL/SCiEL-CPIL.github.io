import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            U = np.zeros(n * m)
            V = np.zeros(m * p)
            W = np.zeros(p * n)
            
            for i, idx in enumerate(u_idx):
                U[idx] = U_blk[i, r]
            for i, idx in enumerate(v_idx):
                V[idx] = V_blk[i, r]
            for i, idx in enumerate(w_idx):
                W[idx] = W_blk[i, r]
                
            U_parts.append(U)
            V_parts.append(V)
            W_parts.append(W)

    # Strassen 2x2x2
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # Schoolbook helpers
    def schoolbook(n_u, n_v, n_w):
        # Returns U (n_u x rank), V (n_v x rank), W (n_w x rank)
        # Rank is n_u * n_v * n_w / (n_u*n_v*n_w) ? No, rank is number of terms.
        # For 1x1x1, rank 1.
        # For 2x2x1, rank 4.
        # General: sum of products.
        rank = n_u * n_v * n_w # This is wrong for Strassen, but for schoolbook it is the number of scalar multiplications?
        # No, schoolbook rank for n x m x p is n*m*p?
        # Wait, 2x2x1 schoolbook:
        # C00 = A00B00 + A01B10
        # C01 = A00B01 + A01B11
        # C10 = A10B00 + A11B10
        # C11 = A10B01 + A11B11
        # This is 4 scalar multiplications? No, it's 8?
        # No, the tensor rank of 2x2x1 is 4?
        # Let's check.
        # T = sum_{i,j,k} A_ij B_jk C_ki.
        # For 2x2x1: i in 0..1, j in 0..1, k in 0..0.
        # Terms:
        # A00 B00 C00
        # A01 B10 C00
        # A10 B00 C10
        # A11 B10 C10
        # Wait, C is 1x2? No, 2x2x1 means A 2x2, B 2x1, C 1x2?
        # Convention: A n x m, B m x p, C p x n.
        # 2x2x1: A 2x2, B 2x1, C 1x2.
        # C00 = A00 B00 + A01 B10
        # C01 = A00 B01? No, B is 2x1. B has only col 0.
        # So B01 doesn't exist.
        # C is 1x2. C00, C01.
        # C00 = A00 B00 + A01 B10.
        # C01 = A10 B00 + A11 B10.
        # Terms:
        # 1: A00 B00 C00
        # 2: A01 B10 C00
        # 3: A10 B00 C01
        # 4: A11 B10 C01
        # Rank 4.
        
        # So for n x m x p, rank is n*m*p?
        # No, for 1x2x2:
        # A 1x2, B 2x2, C 2x1.
        # C00 = A00 B00 + A01 B10
        # C10 = A00 B01 + A01 B11
        # Terms: 4.
        # So yes, schoolbook rank is n*m*p.
        
        rank = n_u * n_v * n_w
        U = np.zeros((n_u, rank))
        V = np.zeros((n_v, rank))
        W = np.zeros((n_w, rank))
        
        r = 0
        for i in range(n_u):
            for j in range(n_v):
                for k in range(n_w):
                    U[i, r] = 1
                    V[j, r] = 1
                    W[k, r] = 1
                    r += 1
        return U, V, W

    # Block 1: 3x3x3 using Strassen tiling (Rank 26)
    # A[0:2, 0:2] x B[0:2, 0:2]
    add_block([0, 1, 4, 5], [0, 1, 5, 6], [0, 3, 1, 4], *strassen_2x2x2())
    
    # A[0:2, 0:2] x B[0:2, 2:3]
    # A 2x2, B 2x1, C 1x2
    U, V, W = schoolbook(2, 2, 1)
    add_block([0, 1, 4, 5], [2, 7], [6, 9], U, V, W)
    
    # A[0:2, 2:3] x B[2:3, 0:2]
    # A 2x1, B 1x2, C 2x2
    U, V, W = schoolbook(2, 1, 2)
    add_block([2, 6], [10, 15], [0, 3, 1, 4], U, V, W)
    
    # A[0:2, 2:3] x B[2:3, 2:3]
    # A 2x1, B 1x1, C 1x2
    U, V, W = schoolbook(2, 1, 1)
    add_block([2, 6], [12], [6, 9], U, V, W)
    
    # A[2:3, 0:2] x B[0:2, 0:2]
    # A 1x2, B 2x2, C 2x1
    U, V, W = schoolbook(1, 2, 2)
    add_block([8], [0, 1, 5, 6], [2, 5], U, V, W)
    
    # A[2:3, 0:2] x B[0:2, 2:3]
    # A 1x2, B 2x1, C 1x1
    U, V, W = schoolbook(1, 2, 1)
    add_block([8], [2, 7], [8], U, V, W)
    
    # A[2:3, 2:3] x B[2:3, 0:2]
    # A 1x1, B 1x2, C 2x1
    U, V, W = schoolbook(1, 1, 2)
    add_block([10], [10, 15], [2, 5], U, V, W)
    
    # A[2:3, 2:3] x B[2:3, 2:3]
    # A 1x1, B 1x1, C 1x1
    U, V, W = schoolbook(1, 1, 1)
    add_block([10], [12], [8], U, V, W)
    
    # Block 2: 3x1x5 (Rank 15)
    # A[0:3, 3] x B[3, 0:5]
    # A 3x1, B 1x5, C 5x3
    U, V, W = schoolbook(3, 1, 5)
    # u_idx: A col 3, rows 0,1,2 -> 3, 7, 11
    # v_idx: B row 3, cols 0..4 -> 15, 16, 17, 18, 19
    # w_idx: C cols 0..4, rows 0..2 -> 
    # C is 5x3. Index k*n + i.
    # k=0, i=0,1,2 -> 0, 1, 2
    # k=1, i=0,1,2 -> 3, 4, 5
    # k=2, i=0,1,2 -> 6, 7, 8
    # k=3, i=0,1,2 -> 9, 10, 11
    # k=4, i=0,1,2 -> 12, 13, 14
    w_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    add_block([3, 7, 11], [15, 16, 17, 18, 19], w_idx, U, V, W)
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
