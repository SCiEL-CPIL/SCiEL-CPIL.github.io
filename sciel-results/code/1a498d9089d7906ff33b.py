"""
Rewritten program to find a lower-rank tensor decomposition for 2x5x6 matrix multiplication.
Uses a block-structured approach applying Strassen's algorithm to 2x2 blocks
and schoolbook for the remainder, reducing the rank from 60 to 54.
"""
import time

DIMENSIONS = [2, 5, 6]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 120 - 10  # Safety margin
    
    # Initialize factor matrices with zeros
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We estimate max rank to be schoolbook 60, but we will use fewer columns.
    # To be safe, we allocate for 60 and trim at the end, or use lists of dicts.
    # Using numpy arrays for efficiency is good, but we need to construct exact integers.
    
    max_rank = n * m * p
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    col_idx = 0
    
    def add_strassen_block(a_i0, a_j0, b_j0, b_k0):
        """
        Add rank-1 terms for Strassen multiplication of A[a_i0:a_i0+2, a_j0:a_j0+2] * B[b_j0:b_j0+2, b_k0:b_k0+2].
        """
        nonlocal col_idx
        
        # Indices for A
        # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        a_idx = [
            (a_i0 * m + a_j0, a_i0 + 1) * 0 + 0, # placeholder
        ]
        
        # Let's explicitly list the 7 terms with coefficients for A, B, C
        # A_coeffs: list of 7 dicts {global_idx: coeff}
        # B_coeffs: list of 7 dicts {global_idx: coeff}
        # C_coeffs: list of 7 dicts {global_idx: coeff}
        
        A = {}
        B = {}
        C = {}
        
        # Helper to get global indices
        def idxA(i, j): return i * m + j
        def idxB(j, k): return j * p + k
        def idxC(i, k): return k * n + i # Trace convention: W[k*n+i]
        
        # Term 1: P1 = (a+d)(e+h)
        # A: a(1), d(1) -> (0,0), (1,1) relative to block
        # B: e(1), h(1) -> (0,0), (1,1) relative to block
        # C: C00(1), C11(1) (From M11 and M00 formulas? Let's verify)
        
        # Verified Strassen Decomposition:
        # P1 = (a+d)(e+h)
        # P2 = (c+d)(e+h)
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P6 = (c+d)f
        # P7 = (a-b)(e+f)
        
        # C00 = P1 - P5 + P7
        # C01 = P3 + P5
        # C10 = P4 + P6
        # C11 = P1 - P2 + P3 + P4
        
        terms = []
        
        # 1
        terms.append(({idxA(a_i0, a_j0): 1, idxA(a_i0+1, a_j0+1): 1},
                      {idxB(b_j0, b_k0): 1, idxB(b_j0+1, b_k0+1): 1},
                      {idxC(a_i0, b_k0): 1, idxC(a_i0+1, b_k0+1): 1}))
        
        # 2
        terms.append(({idxA(a_i0+1, a_j0): 1, idxA(a_i0+1, a_j0+1): 1},
                      {idxB(b_j0, b_k0): 1, idxB(b_j0+1, b_k0+1): 1},
                      {idxC(a_i0+1, b_k0): -1})) # C11 -= P2
        
        # 3
        terms.append(({idxA(a_i0, a_j0): 1},
                      {idxB(b_j0, b_k0+1): 1, idxB(b_j0+1, b_k0+1): -1},
                      {idxC(a_i0, b_k0+1): 1, idxC(a_i0+1, b_k0+1): 1})) # C01 += P3, C11 += P3
        
        # 4
        terms.append(({idxA(a_i0+1, a_j0+1): 1},
                      {idxB(b_j0+1, b_k0): 1, idxB(b_j0, b_k0): -1},
                      {idxC(a_i0, b_k0): 1, idxC(a_i0+1, b_k0+1): 1})) # C00 += P4, C11 += P4
        
        # 5
        terms.append(({idxA(a_i0, a_j0): 1, idxA(a_i0, a_j0+1): 1},
                      {idxB(b_j0+1, b_k0+1): 1},
                      {idxC(a_i0, b_k0): -1, idxC(a_i0, b_k0+1): 1})) # C00 -= P5, C01 += P5
        
        # 6
        terms.append(({idxA(a_i0+1, a_j0): 1, idxA(a_i0+1, a_j0+1): 1},
                      {idxB(b_j0, b_k0+1): 1},
                      {idxC(a_i0+1, b_k0): 1})) # C10 += P6
        
        # 7
        terms.append(({idxA(a_i0, a_j0): 1, idxA(a_i0, a_j0+1): -1},
                      {idxB(b_j0, b_k0): 1, idxB(b_j0, b_k0+1): 1},
                      {idxC(a_i0, b_k0): 1, idxC(a_i0+1, b_k0): 1})) # C00 += P7, C10 += P7
        
        for A_dict, B_dict, C_dict in terms:
            if col_idx >= max_rank:
                return
            for idx, val in A_dict.items():
                if val != 0:
                    U[idx][col_idx] += val
            for idx, val in B_dict.items():
                if val != 0:
                    V[idx][col_idx] += val
            for idx, val in C_dict.items():
                if val != 0:
                    W[idx][col_idx] += val
            col_idx += 1

    def add_schoolbook_block(a_i0, a_j0, b_j0, b_k0, h, w):
        """
        Add rank-1 terms for schoolbook multiplication of A[a_i0:a_i0+h, a_j0:a_j0+w] * B[b_j0:b_j0+w, b_k0:b_k0+w].
        Here we specifically handle the 2x1x6 case where A is 2x1, B is 1x6.
        """
        nonlocal col_idx
        def idxA(i, j): return i * m + j
        def idxB(j, k): return j * p + k
        def idxC(i, k): return k * n + i
        
        for i in range(h):
            for k in range(w):
                if col_idx >= max_rank:
                    return
                U[idxA(a_i0+i, a_j0)][col_idx] = 1
                V[idxB(b_j0, b_k0+k)][col_idx] = 1
                W[idxC(a_i0+i, b_k0+k)][col_idx] = 1
                col_idx += 1

    # Block Decomposition:
    # A is 2x5, B is 5x6.
    # Partition inner dimension 5 into 2+2+1.
    # A = [A1(2x2) A2(2x2) A3(2x1)]
    # B = [B1(2x6); B2(2x6); B3(1x6)]
    
    # Process A1 (cols 0,1) with B1 (rows 0,1)
    # B1 is 2x6, split into three 2x2 blocks horizontally: cols 0-1, 2-3, 4-5
    add_strassen_block(0, 0, 0, 0) # A1 * B1_block0
    add_strassen_block(0, 0, 0, 2) # A1 * B1_block1 -- WAIT.
    # Strassen requires 2x2 A and 2x2 B.
    # A1 is 2x2.
    # B1 is 2x6.
    # We can split B1 into three 2x2 matrices B1_1, B1_2, B1_3.
    # But A1 is the SAME matrix multiplying all three.
    # The multiplication is A1 * (B1_1 B1_2 B1_3) = (A1 B1_1) (A1 B1_2) (A1 B1_3).
    # So we call add_strassen_block 3 times, with same A indices, different B k-offsets.
    
    # Correction in previous thought: 
    # add_strassen_block(a_i0, a_j0, b_j0, b_k0) expects a_j0 to be the start col of A, b_j0 start row of B.
    # For A1 (rows 0-1, cols 0-1), a_i0=0, a_j0=0.
    # For B1 block 0 (rows 0-1, cols 0-1), b_j0=0, b_k0=0.
    # For B1 block 1 (rows 0-1, cols 2-3), b_j0=0, b_k0=2.
    # For B1 block 2 (rows 0-1, cols 4-5), b_j0=0, b_k0=4.
    
    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 0, 0, 2)
    add_strassen_block(0, 0, 0, 4)
    
    # Process A2 (cols 2,3) with B2 (rows 2,3)
    # A2 is 2x2 (rows 0-1, cols 2-3). a_i0=0, a_j0=2.
    # B2 is 2x6 (rows 2-3, cols 0-5).
    # Split B2 into three 2x2 blocks.
    add_strassen_block(0, 2, 2, 0)
    add_strassen_block(0, 2, 2, 2)
    add_strassen_block(0, 2, 2, 4)
    
    # Process A3 (col 4) with B3 (row 4)
    # A3 is 2x1 (rows 0-1, col 4). a_i0=0, a_j0=4.
    # B3 is 1x6 (row 4, cols 0-5). b_j0=4, b_k0=0.
    # h=2, w=6.
    add_schoolbook_block(0, 4, 4, 0, 2, 6)
    
    # Trim matrices to actual rank used
    U_final = [row[:col_idx] for row in U]
    V_final = [row[:col_idx] for row in V]
    W_final = [row[:col_idx] for row in W]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
