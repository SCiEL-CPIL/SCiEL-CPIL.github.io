import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    block_size = 3
    n_blocks = n // block_size  # 2

    # Strassen rank for 2x2 blocks of 3x3 matrices
    # Each 3x3 multiplication has rank 27 (schoolbook)
    # Strassen uses 7 multiplications instead of 8
    rank = 7 * 27  # 189

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0

    # Index helpers for 3x3 blocks
    # Block (ib, jb) where ib, jb in {0, 1}
    # Local indices r, c in {0, 1, 2}
    
    def u_idx(ib, kb, r, c):
        """U index for A_block(ib, kb) at local position (r, c)"""
        return (ib * block_size + r) * m + (kb * block_size + c)

    def v_idx(kb, jb, r, c):
        """V index for B_block(kb, jb) at local position (r, c)"""
        return (kb * block_size + r) * p + (jb * block_size + c)

    def w_idx(ib, jb, r, c):
        """W index for C_block(ib, jb) at local position (r, c)"""
        return (jb * block_size + c) * n + (ib * block_size + r)

    # Strassen's 7 multiplications for 2x2 blocks
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each Strassen multiplication, we need to multiply two 3x3 matrices
    # Each 3x3 multiplication has rank 27 (schoolbook)
    
    # Strassen steps: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: list of (block, coeff) where block is (ib, kb)
    # B_coeffs: list of (block, coeff) where block is (kb, jb)
    # C_coeffs: list of (block, coeff) where block is (ib, jb)
    
    strassen_steps = [
        # M1 = (A00 + A11)(B00 + B11)
        {
            'A': [((0, 0), 1), ((1, 1), 1)],
            'B': [((0, 0), 1), ((1, 1), 1)],
            'C': [((0, 0), 1), ((1, 1), 1)]
        },
        # M2 = (A10 + A11)B00
        {
            'A': [((1, 0), 1), ((1, 1), 1)],
            'B': [((0, 0), 1)],
            'C': [((1, 0), 1), ((1, 1), -1)]
        },
        # M3 = A00(B01 - B11)
        {
            'A': [((0, 0), 1)],
            'B': [((0, 1), 1), ((1, 1), -1)],
            'C': [((0, 1), 1), ((1, 1), 1)]
        },
        # M4 = A11(B10 - B00)
        {
            'A': [((1, 1), 1)],
            'B': [((1, 0), 1), ((0, 0), -1)],
            'C': [((0, 0), 1), ((1, 0), 1)]
        },
        # M5 = (A00 + A01)B11
        {
            'A': [((0, 0), 1), ((0, 1), 1)],
            'B': [((1, 1), 1)],
            'C': [((0, 0), -1), ((0, 1), 1)]
        },
        # M6 = (A10 - A00)(B00 + B01)
        {
            'A': [((1, 0), 1), ((0, 0), -1)],
            'B': [((0, 0), 1), ((0, 1), 1)],
            'C': [((1, 1), 1)]
        },
        # M7 = (A01 - A11)(B10 + B11)
        {
            'A': [((0, 1), 1), ((1, 1), -1)],
            'B': [((1, 0), 1), ((1, 1), 1)],
            'C': [((0, 0), 1)]
        },
    ]

    # For each Strassen step, we need to perform 3x3 matrix multiplication
    # The 3x3 multiplication is done via schoolbook (rank 27)
    # For each (r, c) in 3x3 local indices, we have one term
    
    for step in strassen_steps:
        A_coeffs = step['A']
        B_coeffs = step['B']
        C_coeffs = step['C']
        
        # For each local position (r, c) in the 3x3 blocks
        for r in range(block_size):
            for c in range(block_size):
                # U part: A_s[r, c] = sum coeff * A_block[r, c]
                for (ib, kb), coeff_a in A_coeffs:
                    U[u_idx(ib, kb, r, c), col] += coeff_a
                
                # V part: B_s[r, c] = sum coeff * B_block[r, c]
                # Note: B_block(kb, jb) has row index kb, col index jb
                # For B_s[r, c], we need B_block[kb, jb][r, c]
                # But wait - in the Strassen formula, M = A_s * B_s
                # where A_s is 3x3, B_s is 3x3
                # M[r, c] = sum_k A_s[r, k] * B_s[k, c]
                # So we need to iterate over k as well
                
                # Actually, let me reconsider. The schoolbook rank-27 decomposition
                # for 3x3 matrix multiplication C = A * B is:
                # For each (i, j, k) in {0,1,2}^3:
                #   U[i*3+k, rank] = A[i, k]
                #   V[k*3+j, rank] = B[k, j]
                #   W[j*3+i, rank] = C[i, j]
                
                # So for the Strassen step, we need to do this for the combined matrices
                # Let me restructure: iterate over (i, j, k) for the 3x3 multiplication
                
                pass  # This approach is getting complicated, let me use a cleaner method
    
    # Let me use a cleaner approach: for each Strassen step, do full 3x3 schoolbook
    
    col = 0
    for step in strassen_steps:
        A_coeffs = step['A']
        B_coeffs = step['B']
        C_coeffs = step['C']
        
        # For each (i, j, k) in 3x3 local indices
        for i in range(block_size):
            for j in range(block_size):
                for k in range(block_size):
                    # U part: A_s[i, k] = sum coeff * A_block[i, k]
                    for (ib, kb), coeff_a in A_coeffs:
                        U[u_idx(ib, kb, i, k), col] += coeff_a
                    
                    # V part: B_s[k, j] = sum coeff * B_block[k, j]
                    # B_block(kb, jb) where kb is row block, jb is col block
                    for (kb, jb), coeff_b in B_coeffs:
                        V[v_idx(kb, jb, k, j), col] += coeff_b
                    
                    # W part: C_s[i, j] = sum coeff * C_block[i, j]
                    for (ib, jb), coeff_c in C_coeffs:
                        W[w_idx(ib, jb, i, j), col] += coeff_c
                    
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
