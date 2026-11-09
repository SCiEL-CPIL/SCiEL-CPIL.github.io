import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # n=3 (rows of A), m=4 (cols of A / rows of B), p=4 (cols of B)
    
    # Strassen 2x2x2 decomposition (rank 7)
    # A is 2x2, B is 2x2, C = A*B is 2x2
    # U indices: (i,j) for i,j in {0,1}
    # V indices: (j,k) for j,k in {0,1}  
    # W indices: (k,i) for k,i in {0,1}
    strassen_222 = [
        # M1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)*B00 -> contributes to C01, C11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # M3 = A00*(B01-B11) -> contributes to C10, C11
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        # M4 = A11*(B10-B00) -> contributes to C00, C01
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # M5 = (A00+A01)*B11 -> contributes to C00, C10
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
        # M6 = (A10-A00)*(B00+B01) -> contributes to C11
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01-A11)*(B10+B11) -> contributes to C00
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Schoolbook 1x2x2 decomposition (rank 4)
    # A is 1x2, B is 2x2, C = A*B is 1x2
    # U indices: (i,j) for i=0, j in {0,1}
    # V indices: (j,k) for j in {0,1}, k in {0,1}
    # W indices: (k,i) for k in {0,1}, i=0
    schoolbook_122 = [
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),  # A00*B00 -> C00
        ({(0,1): 1}, {(1,0): 1}, {(0,0): 1}),  # A01*B10 -> C00
        ({(0,0): 1}, {(0,1): 1}, {(1,0): 1}),  # A00*B01 -> C01
        ({(0,1): 1}, {(1,1): 1}, {(1,0): 1}),  # A01*B11 -> C01
    ]
    
    total_rank = 44
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    rank_col = 0
    
    def add_block(i_off, j_off, k_off, terms):
        nonlocal rank_col
        for term in terms:
            u_dict, v_dict, w_dict = term
            
            for (li, lj), coeff in u_dict.items():
                idx_u = (i_off + li) * m + (j_off + lj)
                U[idx_u, rank_col] += coeff
                
            for (lj, lk), coeff in v_dict.items():
                idx_v = (j_off + lj) * p + (k_off + lk)
                V[idx_v, rank_col] += coeff
                
            for (lk, li), coeff in w_dict.items():
                idx_w = (k_off + lk) * n + (i_off + li)
                W[idx_w, rank_col] += coeff
                
            rank_col += 1
            
            if time.time() > deadline:
                break
    
    # Top 2x4 block of A times 4x4 B
    # Split into 2x2 sub-blocks
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_block(0, 0, 0, strassen_222)
    # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_block(0, 2, 0, strassen_222)
    # A[0:2, 0:2] * B[0:2, 2:4] -> C[2:4, 0:2]
    add_block(0, 0, 2, strassen_222)
    # A[0:2, 2:4] * B[2:4, 2:4] -> C[2:4, 0:2]
    add_block(0, 2, 2, strassen_222)
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Bottom 1x4 block of A times 4x4 B
    # A[2, 0:2] * B[0:2, 0:1] -> C[0, 2] (but this is 1x1, not 1x2)
    # Actually, for 1x2 * 2x2 -> 1x2, we need to handle each output column
    # A[2, 0:2] * B[0:2, 0:2] -> C[0:2, 2]
    add_block(2, 0, 0, schoolbook_122)
    # A[2, 2:4] * B[2:4, 0:2] -> C[0:2, 2]
    add_block(2, 2, 0, schoolbook_122)
    # A[2, 0:2] * B[0:2, 2:4] -> C[2:4, 2]
    add_block(2, 0, 2, schoolbook_122)
    # A[2, 2:4] * B[2:4, 2:4] -> C[2:4, 2]
    add_block(2, 2, 2, schoolbook_122)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
