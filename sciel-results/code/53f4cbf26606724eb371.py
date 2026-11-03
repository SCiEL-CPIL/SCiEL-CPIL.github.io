"""
Rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication.
Uses block decomposition with Strassen for 2x2*2x2 and trivial for 2x2*2x1.
M_{2,2,5} = M_{2,2,2} (Strassen, rank 7) + 3 * M_{2,2,1} (trivial, rank 4 each) = 7 + 12 = 19.
M_{4,4,5} = 4 * M_{2,2,5} = 4 * 19 = 76.
"""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We decompose M_{4,4,5} as a block sum.
    # A is 4x4, B is 4x5, C is 4x5.
    # Split A into 2x2 blocks: A_{11}, A_{12}, A_{21}, A_{22} (each 2x2)
    # Split B into top 2 rows (B_top, 2x5) and bottom 2 rows (B_bot, 2x5)
    # C = [A_{11} B_top + A_{12} B_bot; A_{21} B_top + A_{22} B_bot]
    # Each term is M_{2,2,5}.
    # M_{2,2,5} can be decomposed as M_{2,2,2} (Strassen, rank 7) + 3 * M_{2,2,1} (trivial, rank 4 each) = 19.
    # Total rank = 4 * 19 = 76.
    
    def add_trivial_2x2x1(U, V, W, rank_offset, i_offset, j_offset, k_offset):
        """Add trivial decomposition for M_{2,2,1} at given offsets."""
        col = rank_offset
        for i in range(2):
            for j in range(2):
                u_idx = (i_offset + i) * 4 + (j_offset + j)
                v_idx = (j_offset + j) * 5 + (k_offset)
                w_idx = (k_offset) * 4 + (i_offset + i)
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
        return col
    
    def add_strassen_2x2(U, V, W, rank_offset, i_offset, j_offset, k_offset):
        """Add Strassen decomposition for M_{2,2,2} at given offsets."""
        col = rank_offset
        
        # A indices: (i_offset+i, j_offset+j) for i,j in 0..1
        # B indices: (j_offset+j, k_offset+k) for j,k in 0..1
        # C indices: (k_offset+k, i_offset+i) for k,i in 0..1
        
        # Strassen's 7 multiplications:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b21-b22)
        # M4 = a22(b12-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Term 1: M1
        # U: a11 + a22 -> indices (i_offset, j_offset), (i_offset+1, j_offset+1)
        # V: b11 + b22 -> indices (j_offset, k_offset), (j_offset+1, k_offset+1)
        # W: c11 + c22 -> indices (k_offset, i_offset), (k_offset+1, i_offset+1)
        
        u1 = [(i_offset, j_offset), (i_offset+1, j_offset+1)]
        v1 = [(j_offset, k_offset), (j_offset+1, k_offset+1)]
        w1 = [(k_offset, i_offset), (k_offset+1, i_offset+1)]
        
        for (i, j) in u1:
            U[i*4+j][col] = 1
        for (j, k) in v1:
            V[j*5+k][col] = 1
        for (k, i) in w1:
            W[k*4+i][col] = 1
        col += 1
        
        # Term 2: M2
        # U: a21 + a22 -> indices (i_offset+1, j_offset), (i_offset+1, j_offset+1)
        # V: b11 -> index (j_offset, k_offset)
        # W: c21 -> index (k_offset+1, i_offset)
        
        u2 = [(i_offset+1, j_offset), (i_offset+1, j_offset+1)]
        v2 = [(j_offset, k_offset)]
        w2 = [(k_offset+1, i_offset)]
        
        for (i, j) in u2:
            U[i*4+j][col] = 1
        for (j, k) in v2:
            V[j*5+k][col] = 1
        for (k, i) in w2:
            W[k*4+i][col] = 1
        col += 1
        
        # Term 3: M3
        # U: a11 -> index (i_offset, j_offset)
        # V: b21 - b22 -> indices (j_offset+1, k_offset), (j_offset+1, k_offset+1)
        # W: c12 -> index (k_offset+1, i_offset+1)
        
        u3 = [(i_offset, j_offset)]
        v3 = [(j_offset+1, k_offset), (j_offset+1, k_offset+1)]
        w3 = [(k_offset+1, i_offset+1)]
        
        for (i, j) in u3:
            U[i*4+j][col] = 1
        V[(j_offset+1)*5+k_offset][col] = 1
        V[(j_offset+1)*5+k_offset+1][col] = -1
        for (k, i) in w3:
            W[k*4+i][col] = 1
        col += 1
        
        # Term 4: M4
        # U: a22 -> index (i_offset+1, j_offset+1)
        # V: b12 - b11 -> indices (j_offset, k_offset+1), (j_offset, k_offset)
        # W: c11 + c21 -> indices (k_offset, i_offset), (k_offset+1, i_offset)
        
        u4 = [(i_offset+1, j_offset+1)]
        v4 = [(j_offset, k_offset+1), (j_offset, k_offset)]
        w4 = [(k_offset, i_offset), (k_offset+1, i_offset)]
        
        for (i, j) in u4:
            U[i*4+j][col] = 1
        V[j_offset*5+k_offset+1][col] = 1
        V[j_offset*5+k_offset][col] = -1
        for (k, i) in w4:
            W[k*4+i][col] = 1
        col += 1
        
        # Term 5: M5
        # U: a11 + a12 -> indices (i_offset, j_offset), (i_offset, j_offset+1)
        # V: b22 -> index (j_offset+1, k_offset+1)
        # W: -c11 + c12 -> indices (k_offset, i_offset), (k_offset+1, i_offset+1)
        
        u5 = [(i_offset, j_offset), (i_offset, j_offset+1)]
        v5 = [(j_offset+1, k_offset+1)]
        w5 = [(k_offset, i_offset), (k_offset+1, i_offset+1)]
        
        for (i, j) in u5:
            U[i*4+j][col] = 1
        for (j, k) in v5:
            V[j*5+k][col] = 1
        W[k_offset*4+i_offset][col] = -1
        W[(k_offset+1)*4+(i_offset+1)][col] = 1
        col += 1
        
        # Term 6: M6
        # U: a21 - a11 -> indices (i_offset+1, j_offset), (i_offset, j_offset)
        # V: b11 + b12 -> indices (j_offset, k_offset), (j_offset, k_offset+1)
        # W: c22 -> index (k_offset+1, i_offset+1)
        
        u6 = [(i_offset+1, j_offset), (i_offset, j_offset)]
        v6 = [(j_offset, k_offset), (j_offset, k_offset+1)]
        w6 = [(k_offset+1, i_offset+1)]
        
        U[(i_offset+1)*4+j_offset][col] = 1
        U[i_offset*4+j_offset][col] = -1
        for (j, k) in v6:
            V[j*5+k][col] = 1
        for (k, i) in w6:
            W[k*4+i][col] = 1
        col += 1
        
        # Term 7: M7
        # U: a12 - a22 -> indices (i_offset, j_offset+1), (i_offset+1, j_offset+1)
        # V: b21 + b22 -> indices (j_offset+1, k_offset), (j_offset+1, k_offset+1)
        # W: c11 -> index (k_offset, i_offset)
        
        u7 = [(i_offset, j_offset+1), (i_offset+1, j_offset+1)]
        v7 = [(j_offset+1, k_offset), (j_offset+1, k_offset+1)]
        w7 = [(k_offset, i_offset)]
        
        U[i_offset*4+j_offset+1][col] = 1
        U[(i_offset+1)*4+j_offset+1][col] = -1
        for (j, k) in v7:
            V[j*5+k][col] = 1
        for (k, i) in w7:
            W[k*4+i][col] = 1
        col += 1
        
        return col
    
    # Total rank = 4 * 19 = 76
    rank = 76
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block (1,1): A_{11} * B_top
    # A_{11} is rows 0-1, cols 0-1 of A
    # B_top is rows 0-1, cols 0-4 of B
    # Use Strassen for cols 0-1, trivial for cols 2-4
    
    # Strassen for M_{2,2,2} with i_offset=0, j_offset=0, k_offset=0
    col = add_strassen_2x2(U, V, W, col, 0, 0, 0)
    
    # Trivial for M_{2,2,1} with i_offset=0, j_offset=0, k_offset=2
    col = add_trivial_2x2x1(U, V, W, col, 0, 0, 2)
    # Trivial for M_{2,2,1} with i_offset=0, j_offset=0, k_offset=3
    col = add_trivial_2x2x1(U, V, W, col, 0, 0, 3)
    # Trivial for M_{2,2,1} with i_offset=0, j_offset=0, k_offset=4
    col = add_trivial_2x2x1(U, V, W, col, 0, 0, 4)
    
    # Block (1,2): A_{12} * B_bot
    # A_{12} is rows 0-1, cols 2-3 of A
    # B_bot is rows 2-3, cols 0-4 of B
    
    # Strassen for M_{2,2,2} with i_offset=0, j_offset=2, k_offset=0
    col = add_strassen_2x2(U, V, W, col, 0, 2, 0)
    
    # Trivial for M_{2,2,1} with i_offset=0, j_offset=2, k_offset=2
    col = add_trivial_2x2x1(U, V, W, col, 0, 2, 2)
    # Trivial for M_{2,2,1} with i_offset=0, j_offset=2, k_offset=3
    col = add_trivial_2x2x1(U, V, W, col, 0, 2, 3)
    # Trivial for M_{2,2,1} with i_offset=0, j_offset=2, k_offset=4
    col = add_trivial_2x2x1(U, V, W, col, 0, 2, 4)
    
    # Block (2,1): A_{21} * B_top
    # A_{21} is rows 2-3, cols 0-1 of A
    # B_top is rows 0-1, cols 0-4 of B
    
    # Strassen for M_{2,2,2} with i_offset=2, j_offset=0, k_offset=0
    col = add_strassen_2x2(U, V, W, col, 2, 0, 0)
    
    # Trivial for M_{2,2,1} with i_offset=2, j_offset=0, k_offset=2
    col = add_trivial_2x2x1(U, V, W, col, 2, 0, 2)
    # Trivial for M_{2,2,1} with i_offset=2, j_offset=0, k_offset=3
    col = add_trivial_2x2x1(U, V, W, col, 2, 0, 3)
    # Trivial for M_{2,2,1} with i_offset=2, j_offset=0, k_offset=4
    col = add_trivial_2x2x1(U, V, W, col, 2, 0, 4)
    
    # Block (2,2): A_{22} * B_bot
    # A_{22} is rows 2-3, cols 2-3 of A
    # B_bot is rows 2-3, cols 0-4 of B
    
    # Strassen for M_{2,2,2} with i_offset=2, j_offset=2, k_offset=0
    col = add_strassen_2x2(U, V, W, col, 2, 2, 0)
    
    # Trivial for M_{2,2,1} with i_offset=2, j_offset=2, k_offset=2
    col = add_trivial_2x2x1(U, V, W, col, 2, 2, 2)
    # Trivial for M_{2,2,1} with i_offset=2, j_offset=2, k_offset=3
    col = add_trivial_2x2x1(U, V, W, col, 2, 2, 3)
    # Trivial for M_{2,2,1} with i_offset=2, j_offset=2, k_offset=4
    col = add_trivial_2x2x1(U, V, W, col, 2, 2, 4)
    
    if col != rank:
        raise ValueError(f"Column count mismatch: {col} != {rank}")
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
