"""Rewritten program: Strassen-based rank-49 decomposition for 2x2 * 2x14."""
import time

DIMENSIONS = [2, 2, 14]

def construct():
    """
    Decompose the tensor for multiplying a 2x2 matrix (A) by a 2x14 matrix (B)
    into a rank-49 exact decomposition using Strassen's algorithm on the 2x2 block.
    
    We split B into 7 blocks of size 2x2: B = [B_0, B_1, ..., B_6] where B_j is 2x2.
    Then C = A * B has 7 columns of 2x2 blocks: C_j = A * B_j.
    
    For each block j, we compute A * B_j using Strassen (rank 7 for 2x2*2x2).
    Total rank = 7 blocks * 7 Strassen terms = 49.
    
    The output convention is W[k*n+i, r] where n=2, p=14.
    So W has shape (p*n, rank) = (28, rank).
    Index in W: k*n + i where k is the row index in the 2x14 output (0..13) and i is the column index (0..1).
    
    Wait, let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p.
    The tensor T[i,j,k] = C[i,k] = sum_j A[i,j] * B[j,k].
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    W[k*n + i, r] means the index in W is k*n + i, where k is the column of C (0..p-1) and i is the row of C (0..n-1).
    
    So for our case: n=2, m=2, p=14.
    U: (4, rank), V: (28, rank), W: (28, rank).
    
    U[i*m+j, c] corresponds to A[i,j]
    V[j*p+k, c] corresponds to B[j,k]
    W[k*n+i, c] corresponds to C[i,k]
    
    For each block j_block (0..6), B's columns 2*j_block and 2*j_block+1 form a 2x2 block.
    Strassen for 2x2 * 2x2:
    Let A = [[a00, a01], [a10, a11]], B_block = [[b00, b01], [b10, b11]].
    C_block = A * B_block = [[c00, c01], [c10, c11]].
    
    Strassen terms:
    M1 = (a00 + a11) * (b00 + b11)
    M2 = (a10 + a11) * b00
    M3 = a00 * (b01 - b11)
    M4 = a11 * (b10 - b00)
    M5 = (a00 + a01) * b11
    M6 = (a10 - a00) * (b00 + b01)
    M7 = (a01 - a11) * (b10 + b11)
    
    c00 = M1 + M4 - M5 + M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M2 + M3 + M6
    
    For each block j_block and each Strassen term t (0..6), we have one rank-1 term.
    Total rank = 7 * 7 = 49.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // m  # 14 // 2 = 7
    num_strassen = 7
    rank = num_blocks * num_strassen  # 49
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 49)
    V = [[0] * rank for _ in range(m * p)]  # (28, 49)
    W = [[0] * rank for _ in range(p * n)]  # (28, 49)
    
    # Strassen coefficients:
    # For each Strassen term, we need:
    # - U coefficients for A elements (a00, a01, a10, a11)
    # - V coefficients for B_block elements (b00, b01, b10, b11)
    # - W coefficients for C_block elements (c00, c01, c10, c11)
    
    # U coefficients for each Strassen term (index: [a00, a01, a10, a11])
    # U index: i*m + j, so a00=0, a01=1, a10=2, a11=3
    U_coeffs = [
        [1, 0, 0, 1],  # M1: (a00 + a11)
        [0, 0, 1, 1],  # M2: (a10 + a11)
        [1, 0, 0, 0],  # M3: a00
        [0, 0, 0, 1],  # M4: a11
        [1, 1, 0, 0],  # M5: (a00 + a01)
        [0, 0, 1, -1], # M6: (a10 - a00)
        [0, 1, 0, -1], # M7: (a01 - a11)
    ]
    
    # V coefficients for each Strassen term (index: [b00, b01, b10, b11])
    V_coeffs = [
        [1, 0, 0, 1],  # M1: (b00 + b11)
        [1, 0, 0, 0],  # M2: b00
        [0, 1, 0, -1], # M3: (b01 - b11)
        [0, 0, 1, -1], # M4: (b10 - b00)
        [0, 0, 0, 1],  # M5: b11
        [1, 1, 0, 0],  # M6: (b00 + b01)
        [0, 0, 1, 1],  # M7: (b10 + b11)
    ]
    
    # W coefficients for each Strassen term (index: [c00, c01, c10, c11])
    W_coeffs = [
        [1, 0, 0, -1], # M1: c00 += M1, c11 -= M1
        [0, 0, 0, 1],  # M2: c11 += M2... wait let me re-check
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        # So:
        # M1: c00 += 1, c11 += 1
        # M2: c10 += 1, c11 -= 1
        # M3: c01 += 1, c11 += 1
        # M4: c00 += 1, c10 += 1
        # M5: c00 -= 1, c01 += 1
        # M6: c11 += 1
        # M7: c00 += 1
        [1, 0, 0, 1],  # M1: c00 += 1, c11 += 1
        [0, 0, 1, -1], # M2: c10 += 1, c11 -= 1
        [0, 1, 0, 1],  # M3: c01 += 1, c11 += 1
        [1, 0, 1, 0],  # M4: c00 += 1, c10 += 1
        [-1, 1, 0, 0], # M5: c00 -= 1, c01 += 1
        [0, 0, 0, 1],  # M6: c11 += 1
        [1, 0, 0, 0],  # M7: c00 += 1
    ]
    
    # Now fill in U, V, W
    for block in range(num_blocks):
        for t in range(num_strassen):
            col = block * num_strassen + t
            
            # U: A elements
            for a_idx in range(4):
                U[a_idx][col] = U_coeffs[t][a_idx]
            
            # V: B elements for this block
            # B[j, k] where j is 0 or 1, k is 2*block, 2*block+1
            # b00 = B[0, 2*block], b01 = B[0, 2*block+1]
            # b10 = B[1, 2*block], b11 = B[1, 2*block+1]
            # V index: j*p + k
            for b_idx in range(4):
                if b_idx == 0:  # b00
                    j, k = 0, 2 * block
                elif b_idx == 1:  # b01
                    j, k = 0, 2 * block + 1
                elif b_idx == 2:  # b10
                    j, k = 1, 2 * block
                else:  # b11
                    j, k = 1, 2 * block + 1
                V[j * p + k][col] = V_coeffs[t][b_idx]
            
            # W: C elements for this block
            # C[i, k] where i is 0 or 1, k is 2*block, 2*block+1
            # c00 = C[0, 2*block], c01 = C[0, 2*block+1]
            # c10 = C[1, 2*block], c11 = C[1, 2*block+1]
            # W index: k*n + i
            for c_idx in range(4):
                if c_idx == 0:  # c00
                    i, k = 0, 2 * block
                elif c_idx == 1:  # c01
                    i, k = 0, 2 * block + 1
                elif c_idx == 2:  # c10
                    i, k = 1, 2 * block
                else:  # c11
                    i, k = 1, 2 * block + 1
                W[k * n + i][col] = W_coeffs[t][c_idx]
    
    return {"U": U, "V": V, "W": W}
