import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Strassen coefficients for P1..P7
    # P1: (A11+A22)(B11+B22)
    # P2: (A21+A22)B11
    # P3: A11(B12-B22)
    # P4: A22(B21-B11)
    # P5: (A11+A12)B22
    # P6: (A21-A11)(B11+B12)
    # P7: (A12-A22)(B21+B22)
    
    # Output contributions:
    # C11: P1, P4, -P5, P7
    # C12: P3, P5
    # C21: P2, P4
    # C22: P1, -P2, P3, P6
    
    # We define the local indices for A, B, C in the 2x2x2 block
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    # Mapping for Group 1 (A1, B1)
    # A1 is A[:, 0:2]
    # B1 is B[0:2, :]
    
    # Mapping for Group 2 (A2, B2)
    # A2 is A[:, 2:4]
    # B2 is B[2:4, :]

    def get_a_idx(group, local_i, local_j):
        # group 1: cols 0,1. group 2: cols 2,3
        global_col = local_j + 2 * (group - 1)
        global_row = local_i
        return global_row * m + global_col

    def get_b_idx(group, local_i, local_j, block_col):
        # group 1: rows 0,1. group 2: rows 2,3
        # block_col: 0 for cols 0,1; 1 for cols 2,3; 2 for cols 4,5
        global_row = local_i + 2 * (group - 1)
        global_col = local_j + 2 * block_col
        return global_row * p + global_col

    def get_c_idx(local_i, local_j, block_col):
        # local_i is row of C (0,1)
        # local_j is col of C within block (0,1)
        # block_col is which 2-col block of C (0,1,2)
        global_row = local_i
        global_col = local_j + 2 * block_col
        # W index: k*n + i -> global_col * n + global_row
        return global_col * n + global_row

    col = 0
    for group in [1, 2]:
        for block_col in [0, 1, 2]:
            # 7 Strassen terms
            # Term 1: P1
            # A: A11+A22
            # B: B11+B22
            # C: C11 +1, C22 +1
            U[get_a_idx(group, 0, 0)][col] += 1
            U[get_a_idx(group, 1, 1)][col] += 1
            V[get_b_idx(group, 0, 0, block_col)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] += 1
            W[get_c_idx(0, 0, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] += 1
            col += 1
            
            # Term 2: P2
            # A: A21+A22
            # B: B11
            # C: C21 +1, C22 -1
            U[get_a_idx(group, 1, 0)][col] += 1
            U[get_a_idx(group, 1, 1)][col] += 1
            V[get_b_idx(group, 0, 0, block_col)][col] += 1
            W[get_c_idx(1, 0, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] -= 1
            col += 1
            
            # Term 3: P3
            # A: A11
            # B: B12-B22
            # C: C12 +1, C22 +1
            U[get_a_idx(group, 0, 0)][col] += 1
            V[get_b_idx(group, 0, 1, block_col)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] -= 1
            W[get_c_idx(0, 1, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] += 1
            col += 1
            
            # Term 4: P4
            # A: A22
            # B: B21-B11
            # C: C11 +1, C21 +1
            U[get_a_idx(group, 1, 1)][col] += 1
            V[get_b_idx(group, 1, 0, block_col)][col] += 1
            V[get_b_idx(group, 0, 0, block_col)][col] -= 1
            W[get_c_idx(0, 0, block_col)][col] += 1
            W[get_c_idx(1, 0, block_col)][col] += 1
            col += 1
            
            # Term 5: P5
            # A: A11+A12
            # B: B22
            # C: C11 -1, C12 +1
            U[get_a_idx(group, 0, 0)][col] += 1
            U[get_a_idx(group, 0, 1)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] += 1
            W[get_c_idx(0, 0, block_col)][col] -= 1
            W[get_c_idx(0, 1, block_col)][col] += 1
            col += 1
            
            # Term 6: P6
            # A: A21-A11
            # B: B11+B12
            # C: C22 +1
            U[get_a_idx(group, 1, 0)][col] += 1
            U[get_a_idx(group, 0, 0)][col] -= 1
            V[get_b_idx(group, 0, 0, block_col)][col] += 1
            V[get_b_idx(group, 0, 1, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] += 1
            col += 1
            
            # Term 7: P7
            # A: A12-A22
            # B: B21+B22
            # C: C11 +1
            U[get_a_idx(group, 0, 1)][col] += 1
            U[get_a_idx(group, 1, 1)][col] -= 1
            V[get_b_idx(group, 1, 0, block_col)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] += 1
            W[get_c_idx(0, 0, block_col)][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
