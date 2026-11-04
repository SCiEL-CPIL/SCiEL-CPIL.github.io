import time
import numpy as np

def construct():
    n, m, p = 3, 4, 8
    deadline = time.time() + 130
    
    # We aim for rank 88 using block decomposition.
    # n=3, m=4, p=8
    # Split n into 2+1.
    # C_top (2x8) = A_top (2x4) * B (4x8)
    # C_bot (1x8) = A_bot (1x4) * B (4x8)
    
    # Rank of 2x4x8:
    # Split m into 2+2.
    # C_top = A_top_L (2x2) * B_L (2x8) + A_top_R (2x2) * B_R (2x8)
    # Rank(2x2x8) = 4 * Rank(2x2x2) = 4 * 7 = 28.
    # Rank(2x4x8) = 28 + 28 = 56.
    
    # Rank of 1x4x8:
    # Split m into 2+2.
    # C_bot = A_bot_L (1x2) * B_L (2x8) + A_bot_R (1x2) * B_R (2x8)
    # Rank(1x2x8) = 4 * Rank(1x2x2) = 4 * 4 = 16.
    # Rank(1x4x8) = 16 + 16 = 32.
    
    # Total rank = 56 + 32 = 88.
    
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a block decomposition
    # block_U: indices for U in the block
    # block_V: indices for V in the block
    # block_W: indices for W in the block
    # base_rank: offset in the rank dimension
    
    def add_strassen_222(base_r, u_offset, v_offset, w_offset, scale=1.0):
        # u_offset: (i, j) in local 2x2 -> global index in U
        # v_offset: (j, k) in local 2x2 -> global index in V
        # w_offset: (k, i) in local 2x2 -> global index in W
        # Strassen terms
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U entries for A
        # A11 is at (0,0), A12 at (0,1), A21 at (1,0), A22 at (1,1)
        # Global U index: u_offset(i, j)
        
        # Term 1
        U[u_offset(0,0), base_r] = 1.0
        U[u_offset(1,1), base_r] = 1.0
        V[v_offset(0,0), base_r] = 1.0
        V[v_offset(0,1), base_r] = 1.0
        W[w_offset(0,0), base_r] = 1.0
        W[w_offset(0,1), base_r] = 1.0
        W[w_offset(1,0), base_r] = 1.0
        W[w_offset(1,1), base_r] = 1.0
        
        # Term 2
        U[u_offset(1,0), base_r+1] = 1.0
        U[u_offset(1,1), base_r+1] = 1.0
        V[v_offset(0,0), base_r+1] = 1.0
        W[w_offset(0,0), base_r+1] = 1.0
        W[w_offset(0,1), base_r+1] = 1.0
        
        # Term 3
        U[u_offset(0,0), base_r+2] = 1.0
        V[v_offset(0,1), base_r+2] = 1.0
        V[v_offset(1,1), base_r+2] = -1.0
        W[w_offset(0,1), base_r+2] = 1.0
        W[w_offset(1,1), base_r+2] = 1.0
        
        # Term 4
        U[u_offset(1,1), base_r+3] = 1.0
        V[v_offset(1,0), base_r+3] = 1.0
        V[v_offset(0,0), base_r+3] = -1.0
        W[w_offset(1,0), base_r+3] = 1.0
        W[w_offset(1,1), base_r+3] = 1.0
        
        # Term 5
        U[u_offset(0,0), base_r+4] = 1.0
        U[u_offset(0,1), base_r+4] = 1.0
        V[v_offset(1,1), base_r+4] = 1.0
        W[w_offset(1,1), base_r+4] = 1.0
        W[w_offset(1,0), base_r+4] = 1.0
        
        # Term 6
        U[u_offset(1,0), base_r+5] = 1.0
        U[u_offset(0,0), base_r+5] = -1.0
        V[v_offset(0,0), base_r+5] = 1.0
        V[v_offset(0,1), base_r+5] = 1.0
        W[w_offset(1,0), base_r+5] = 1.0
        W[w_offset(1,1), base_r+5] = 1.0
        
        # Term 7
        U[u_offset(0,1), base_r+6] = 1.0
        U[u_offset(1,1), base_r+6] = -1.0
        V[v_offset(1,0), base_r+6] = 1.0
        V[v_offset(1,1), base_r+6] = 1.0
        W[w_offset(1,1), base_r+6] = 1.0
        W[w_offset(1,0), base_r+6] = 1.0

    def add_rank4_122(base_r, u_offset, v_offset, w_offset):
        # 1x2x2 matmul
        # C1 = A1 B11 + A2 B21
        # C2 = A1 B12 + A2 B22
        # A1 at (0,0), A2 at (0,1) ? No, A is 1x2.
        # U index: (i=0, j).
        # V index: (j, k).
        # W index: (k, i=0).
        
        # Term 1: A1 B11 -> C1
        U[u_offset(0,0), base_r] = 1.0
        V[v_offset(0,0), base_r] = 1.0
        W[w_offset(0,0), base_r] = 1.0
        
        # Term 2: A2 B21 -> C1
        U[u_offset(0,1), base_r+1] = 1.0
        V[v_offset(1,0), base_r+1] = 1.0
        W[w_offset(0,0), base_r+1] = 1.0
        
        # Term 3: A1 B12 -> C2
        U[u_offset(0,0), base_r+2] = 1.0
        V[v_offset(0,1), base_r+2] = 1.0
        W[w_offset(1,0), base_r+2] = 1.0
        
        # Term 4: A2 B22 -> C2
        U[u_offset(0,1), base_r+3] = 1.0
        V[v_offset(1,1), base_r+3] = 1.0
        W[w_offset(1,0), base_r+3] = 1.0

    # Global indices
    # U: i in 0..2, j in 0..3. Index = i*4 + j
    # V: j in 0..3, k in 0..7. Index = j*8 + k
    # W: k in 0..7, i in 0..2. Index = k*3 + i
    
    r = 0
    
    # Top part: n=2 (i=0,1), m=4, p=8
    # Split m into 2+2 (j=0,1 and j=2,3)
    # Split p into 4x2 (k=0..1, 2..3, 4..5, 6..7)
    
    # Block 1: A_top_L (i=0,1; j=0,1), B_L (j=0,1; k=0..7)
    # This is 2x2x8.
    # Split k into 4 blocks of 2.
    for kb in range(4):
        k_start = kb * 2
        # u_offset(i, j) -> global U index
        def u_off(i, j):
            return i * 4 + j
        # v_offset(j, k) -> global V index
        def v_off(j, k):
            return j * 8 + (k_start + k)
        # w_offset(k, i) -> global W index
        def w_off(k, i):
            return (k_start + k) * 3 + i
            
        add_strassen_222(r, u_off, v_off, w_off)
        r += 7
        
    # Block 2: A_top_R (i=0,1; j=2,3), B_R (j=2,3; k=0..7)
    for kb in range(4):
        k_start = kb * 2
        def u_off(i, j):
            return i * 4 + (j + 2)
        def v_off(j, k):
            return (j + 2) * 8 + (k_start + k)
        def w_off(k, i):
            return (k_start + k) * 3 + i
            
        add_strassen_222(r, u_off, v_off, w_off)
        r += 7
        
    # Bottom part: n=1 (i=2), m=4, p=8
    # Split m into 2+2
    # Split p into 4x2
    
    # Block 3: A_bot_L (i=2; j=0,1), B_L (j=0,1; k=0..7)
    # This is 1x2x8.
    for kb in range(4):
        k_start = kb * 2
        def u_off(i, j):
            return 2 * 4 + j
        def v_off(j, k):
            return j * 8 + (k_start + k)
        def w_off(k, i):
            return (k_start + k) * 3 + 2
            
        add_rank4_122(r, u_off, v_off, w_off)
        r += 4
        
    # Block 4: A_bot_R (i=2; j=2,3), B_R (j=2,3; k=0..7)
    for kb in range(4):
        k_start = kb * 2
        def u_off(i, j):
            return 2 * 4 + (j + 2)
        def v_off(j, k):
            return (j + 2) * 8 + (k_start + k)
        def w_off(k, i):
            return (k_start + k) * 3 + 2
            
        add_rank4_122(r, u_off, v_off, w_off)
        r += 4
        
    # Verify
    # T[i,j,k] = 1
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r_idx in range(rank):
                    s += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                if abs(s - 1.0) > 1e-6:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
