import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by summing block decompositions.
    # Split p=5 into p1=2 and p2=3.
    # Block 1: M_{3,3,2}
    # Block 2: M_{3,3,3}
    
    # Helper to get schoolbook decomposition for n x m x p
    def schoolbook(n_b, m_b, p_b):
        rank = n_b * m_b * p_b
        U = [[0]*rank for _ in range(n_b*m_b)]
        V = [[0]*rank for _ in range(m_b*p_b)]
        W = [[0]*rank for _ in range(p_b*n_b)]
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[i*m_b+j][col] = 1
                    V[j*p_b+k][col] = 1
                    W[k*n_b+i][col] = 1
                    col += 1
        return U, V, W

    # Helper to get Strassen decomposition for 2x2x2
    def strassen_222():
        # Returns U, V, W for 2x2x2 with rank 7
        # Indices: i in 0,1; j in 0,1; k in 0,1
        # U is 4x7, V is 4x7, W is 4x7
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Strassen formulas for C = AB
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map these to U, V, W entries.
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # n=2, m=2, p=2
        
        # M1
        # U: (0,0)=1, (1,1)=1 -> indices 0, 3
        # V: (0,0)=1, (0,1)=1? No.
        # (A11 + A22) means U[0,0]=1, U[1,1]=1.
        # (B11 + B22) means V[0,0]=1, V[1,1]=1.
        # W: C11 and C22 get M1.
        # C11 is i=0, k=0 -> W[0*2+0]=W[0].
        # C22 is i=1, k=1 -> W[1*2+1]=W[3].
        
        # Let's define the 7 terms
        terms = [
            # (U_mask, V_mask, W_mask)
            # U_mask: dict (i,j) -> val
            # V_mask: dict (j,k) -> val
            # W_mask: dict (k,i) -> val
        ]
        
        # It's easier to just fill the arrays directly.
        
        # Term 0: M1 = (A11+A22)(B11+B22)
        # U[0]=1, U[3]=1
        # V[0]=1, V[3]=1
        # W[0]=1 (C11), W[3]=1 (C22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # Term 1: M2 = (A21+A22)B11
        # U[2]=1 (A21), U[3]=1 (A22)
        # V[0]=1 (B11)
        # W[2]=1 (C21)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # Term 2: M3 = A11(B21-B22)
        # U[0]=1 (A11)
        # V[2]=1 (B21), V[3]=-1 (B22)
        # W[1]=1 (C12)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # Term 3: M4 = A22(B12-B11)
        # U[3]=1 (A22)
        # V[1]=1 (B12), V[0]=-1 (B11)
        # W[0]=1 (C11), W[2]=1 (C21)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # Term 4: M5 = (A11+A12)B22
        # U[0]=1 (A11), U[1]=1 (A12)
        # V[3]=1 (B22)
        # W[0]=-1 (C11), W[1]=1 (C12)
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # Term 5: M6 = (A21-A11)(B11+B12)
        # U[2]=1 (A21), U[0]=-1 (A11)
        # V[0]=1 (B11), V[1]=1 (B12)
        # W[3]=1 (C22)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # Term 6: M7 = (A12-A22)(B21+B22)
        # U[1]=1 (A12), U[3]=-1 (A22)
        # V[2]=1 (B21), V[3]=1 (B22)
        # W[0]=1 (C11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    def get_block_decomp(n_b, m_b, p_b):
        if n_b == 2 and m_b == 2 and p_b == 2:
            return strassen_222()
        else:
            return schoolbook(n_b, m_b, p_b)

    def merge_blocks(blocks, n, m, p):
        # blocks: list of (n_b, m_b, p_b, i_off, j_off, k_off, U, V, W)
        total_rank = sum(len(b[6][0]) for b in blocks)
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        
        rank_offset = 0
        for (n_b, m_b, p_b, i_off, j_off, k_off, U_b, V_b, W_b) in blocks:
            r_b = len(U_b[0])
            for i in range(n_b):
                for j in range(m_b):
                    for r in range(r_b):
                        U[(i+i_off)*m + (j+j_off)][rank_offset + r] = U_b[i*m_b+j][r]
            for j in range(m_b):
                for k in range(p_b):
                    for r in range(r_b):
                        V[(j+j_off)*p + (k+k_off)][rank_offset + r] = V_b[j*p_b+k][r]
            for k in range(p_b):
                for i in range(n_b):
                    for r in range(r_b):
                        W[(k+k_off)*n + (i+i_off)][rank_offset + r] = W_b[k*n_b+i][r]
            rank_offset += r_b
            
        return U, V, W

    # Define blocks for M_{3,3,5}
    # Split p=5 into 2 and 3.
    # Block 1: n=3, m=3, p=2. i_off=0, j_off=0, k_off=0.
    # Block 2: n=3, m=3, p=3. i_off=0, j_off=0, k_off=2.
    
    # We need to decompose M_{3,3,2} and M_{3,3,3} further into sub-blocks.
    
    def decompose_332():
        # n=3, m=3, p=2
        # Split i into 2,1. Split j into 2,1.
        # Blocks:
        # 1. n_b=2, m_b=2, p_b=2. i_off=0, j_off=0.
        # 2. n_b=2, m_b=1, p_b=2. i_off=0, j_off=2.
        # 3. n_b=1, m_b=2, p_b=2. i_off=2, j_off=0.
        # 4. n_b=1, m_b=1, p_b=2. i_off=2, j_off=2.
        
        blocks = []
        
        # Block 1
        U1, V1, W1 = get_block_decomp(2, 2, 2)
        blocks.append((2, 2, 2, 0, 0, 0, U1, V1, W1))
        
        # Block 2
        U2, V2, W2 = get_block_decomp(2, 1, 2)
        blocks.append((2, 1, 2, 0, 2, 0, U2, V2, W2))
        
        # Block 3
        U3, V3, W3 = get_block_decomp(1, 2, 2)
        blocks.append((1, 2, 2, 2, 0, 0, U3, V3, W3))
        
        # Block 4
        U4, V4, W4 = get_block_decomp(1, 1, 2)
        blocks.append((1, 1, 2, 2, 2, 0, U4, V4, W4))
        
        return merge_blocks(blocks, 3, 3, 2)

    def decompose_333():
        # n=3, m=3, p=3
        # Split i into 2,1. Split j into 2,1. Split k into 2,1.
        # 8 blocks.
        blocks = []
        
        # (n_b, m_b, p_b, i_off, j_off, k_off)
        specs = [
            (2, 2, 2, 0, 0, 0),
            (2, 2, 1, 0, 0, 2),
            (2, 1, 2, 0, 2, 0),
            (2, 1, 1, 0, 2, 2),
            (1, 2, 2, 2, 0, 0),
            (1, 2, 1, 2, 0, 2),
            (1, 1, 2, 2, 2, 0),
            (1, 1, 1, 2, 2, 2),
        ]
        
        for (n_b, m_b, p_b, i_off, j_off, k_off) in specs:
            U_b, V_b, W_b = get_block_decomp(n_b, m_b, p_b)
            blocks.append((n_b, m_b, p_b, i_off, j_off, k_off, U_b, V_b, W_b))
            
        return merge_blocks(blocks, 3, 3, 3)

    # Now combine M_{3,3,2} and M_{3,3,3}
    U_332, V_332, W_332 = decompose_332()
    U_333, V_333, W_333 = decompose_333()
    
    # Merge them into final U, V, W for 3x3x5
    # U_332 is 9x17, U_333 is 9x26
    # Final U is 9x(17+26)
    
    rank_332 = len(U_332[0])
    rank_333 = len(U_333[0])
    total_rank = rank_332 + rank_333
    
    U = [[0]*total_rank for _ in range(9)]
    V = [[0]*total_rank for _ in range(15)]
    W = [[0]*total_rank for _ in range(15)]
    
    for i in range(9):
        for r in range(rank_332):
            U[i][r] = U_332[i][r]
        for r in range(rank_333):
            U[i][rank_332 + r] = U_333[i][r]
            
    for i in range(15):
        for r in range(rank_332):
            V[i][r] = V_332[i][r]
        for r in range(rank_333):
            V[i][rank_332 + r] = V_333[i][r]
            
    for i in range(15):
        for r in range(rank_332):
            W[i][r] = W_332[i][r]
        for r in range(rank_333):
            W[i][rank_332 + r] = W_333[i][r]
            
    return {"U": U, "V": V, "W": W}

