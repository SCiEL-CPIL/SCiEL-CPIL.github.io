import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # We will construct a rank 38 solution.
    # Split m=4 into two blocks of m=2.
    # For each block, split p=5 into p=2 and p=3.
    # Use Strassen (rank 7) for 2x2x2 part.
    # Use naive (rank 12) for 2x2x3 part.
    
    # Total rank = 2 * (7 + 12) = 38.
    
    rank_total = 38
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank 1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Strassen for 2x2x2
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = A B
    # P1 = A11(B11+B22)
    # P2 = (A11+A21)B22
    # P3 = (A12+A22)B11
    # P4 = A22(B22-B11)
    # P5 = (A11+A12)B21
    # P6 = (A21+A22)(B12-B11)
    # P7 = (A11+A21)(B12+B22)
    # C11 = P1 - P2 + P4 + P7
    # C12 = P3 + P5 - P4 - P6
    # C21 = P2 + P4 - P5 - P7
    # C22 = P1 - P3 + P6 + P7
    
    def strassen_block(a_base, b_base, c_base):
        nonlocal col
        # a_base, b_base, c_base are offsets in U, V, W
        # A indices: a_base, a_base+1, a_base+2, a_base+3
        # B indices: b_base, b_base+1, b_base+2, b_base+3
        # C indices: c_base, c_base+1, c_base+2, c_base+3
        
        # P1 = A11(B11+B22)
        U[a_base][col] = 1
        V[b_base][col] = 1
        V[b_base+3][col] = 1
        W[c_base][col] = 1
        W[c_base+3][col] = 1
        col += 1
        
        # P2 = (A11+A21)B22
        U[a_base][col] = 1
        U[a_base+2][col] = 1
        V[b_base+3][col] = 1
        W[c_base+2][col] = 1
        W[c_base+3][col] = 1
        col += 1
        
        # P3 = (A12+A22)B11
        U[a_base+1][col] = 1
        U[a_base+3][col] = 1
        V[b_base][col] = 1
        W[c_base+1][col] = 1
        W[c_base+2][col] = 1
        col += 1
        
        # P4 = A22(B22-B11)
        U[a_base+3][col] = 1
        V[b_base+3][col] = 1
        V[b_base][col] = -1
        W[c_base][col] = 1
        W[c_base+1][col] = -1
        W[c_base+2][col] = 1
        col += 1
        
        # P5 = (A11+A12)B21
        U[a_base][col] = 1
        U[a_base+1][col] = 1
        V[b_base+2][col] = 1
        W[c_base+1][col] = 1
        W[c_base+2][col] = -1
        col += 1
        
        # P6 = (A21+A22)(B12-B11)
        U[a_base+2][col] = 1
        U[a_base+3][col] = 1
        V[b_base+1][col] = 1
        V[b_base][col] = -1
        W[c_base+1][col] = -1
        W[c_base+3][col] = 1
        col += 1
        
        # P7 = (A11+A21)(B12+B22)
        U[a_base][col] = 1
        U[a_base+2][col] = 1
        V[b_base+1][col] = 1
        V[b_base+3][col] = 1
        W[c_base][col] = 1
        W[c_base+1][col] = -1
        W[c_base+2][col] = -1
        W[c_base+3][col] = 1
        col += 1

    def naive_block(a_base, b_base, c_base, p_block):
        nonlocal col
        # p_block is 3
        for i in range(2):
            for j in range(2):
                for k in range(p_block):
                    U[a_base + i*2 + j][col] = 1
                    V[b_base + j*p_block + k][col] = 1
                    W[c_base + k*n + i][col] = 1
                    col += 1

    # Block 1: A1 (2x2) * B11 (2x2)
    # A1 is rows 0,1 of A (cols 0,1) -> U indices 0,1,2,3
    # B11 is rows 0,1 of B (cols 0,1) -> V indices 0,1,2,3
    # C11 is cols 0,1 of C (rows 0,1) -> W indices 0,1,2,3 (k*n+i)
    # k=0,1; i=0,1.
    # W indices:
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    strassen_block(0, 0, 0)
    
    # Block 2: A1 (2x2) * B12 (2x3)
    # B12 is rows 0,1 of B (cols 2,3,4) -> V indices 4,5,6, 7,8,9
    # C12 is cols 2,3,4 of C (rows 0,1) -> W indices
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    # k=3, i=0 -> 6
    # k=3, i=1 -> 7
    # k=4, i=0 -> 8
    # k=4, i=1 -> 9
    naive_block(0, 4, 4, 3)
    
    # Block 3: A2 (2x2) * B21 (2x2)
    # A2 is rows 0,1 of A (cols 2,3) -> U indices 4,5,6,7
    # B21 is rows 2,3 of B (cols 0,1) -> V indices 10,11,12,13
    # C21 is cols 0,1 of C (rows 0,1) -> W indices 0,1,2,3
    strassen_block(4, 10, 0)
    
    # Block 4: A2 (2x2) * B22 (2x3)
    # B22 is rows 2,3 of B (cols 2,3,4) -> V indices 14,15,16, 17,18,19
    # C22 is cols 2,3,4 of C (rows 0,1) -> W indices 4,5,6,7,8,9
    naive_block(4, 14, 4, 3)
    
    # Check if we exceeded rank
    if col != rank_total:
        # Fallback to seed if something went wrong
        rank_seed = n * m * p
        U_seed = [[0] * rank_seed for _ in range(n * m)]
        V_seed = [[0] * rank_seed for _ in range(m * p)]
        W_seed = [[0] * rank_seed for _ in range(p * n)]
        col_seed = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_seed[i * m + j][col_seed] = 1
                    V_seed[j * p + k][col_seed] = 1
                    W_seed[k * n + i][col_seed] = 1
                    col_seed += 1
        return {"U": U_seed, "V": V_seed, "W": W_seed}
    
    return {"U": U, "V": V, "W": W}
