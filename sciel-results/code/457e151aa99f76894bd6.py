"""Rank-84 exact tensor decomposition for 4x4 * 4x6 matrix multiplication using Kronecker product of Strassen and schoolbook."""

import time
import numpy as np

def construct():
    """
    Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Kronecker product of Strassen (rank 7 for 2x2x2) and schoolbook (rank 12 for 2x2x3).
    This gives total rank 7 * 12 = 84, beating pure schoolbook rank 96.
    
    The 4x4 * 4x6 multiplication is decomposed as:
    - Split A into 2x2 blocks (4 blocks of 2x2)
    - Split B into 2x2 blocks (4 blocks of 2x2, only first 2 cols of each 2x3 block used)
    - Split C into 2x2 blocks (4 blocks of 2x2)
    
    Wait - the correct decomposition:
    A is 4x4, B is 4x6, C is 4x6.
    Split into 2x2 blocks:
    A: 2x2 blocks of size 2x2 (4 blocks)
    B: 2x3 blocks - but Strassen works on 2x2. 
    
    Actually, let me reconsider. The proper Kronecker approach:
    4x4 * 4x6 = (2x2 blocks) * (2x3 blocks) -> (2x3 blocks)
    
    For Strassen on the 2x2 block structure:
    A has 4 blocks: A00, A01, A10, A11 each 2x2
    B has 4 blocks: B00, B01, B10, B11 each 2x3
    C has 4 blocks: C00, C01, C10, C11 each 2x3
    
    Strassen computes 7 products of (linear combo of A blocks) * (linear combo of B blocks)
    But B blocks are 2x3, so each product is 2x2 * 2x3 = 2x3.
    
    Each 2x2 * 2x3 multiplication has schoolbook rank 12.
    Total: 7 * 12 = 84.
    
    For each Strassen step i, and each schoolbook term (r,s,k):
    U[A_entry] = coeff_A, V[B_entry] = coeff_B, W[C_entry] = coeff_C
    """
    start_time = time.time()
    DEADLINE = 140.0  # seconds
    
    n, m, p = 4, 4, 6
    
    rank = 84
    U = np.zeros((n * m, rank), dtype=object)   # 16 x 84
    V = np.zeros((m * p, rank), dtype=object)   # 24 x 84
    W = np.zeros((p * n, rank), dtype=object)   # 24 x 84
    
    # Strassen's 7 intermediate products for 2x2 block structure
    # A blocks: A00=(rows0-1,cols0-1), A01=(rows0-1,cols2-3), A10=(rows2-3,cols0-1), A11=(rows2-3,cols2-3)
    # B blocks: B00=(rows0-1,cols0-2), B01=(rows0-1,cols3-5), B10=(rows2-3,cols0-2), B11=(rows2-3,cols3-5)
    # C blocks: C00=(rows0-1,cols0-2), C01=(rows0-1,cols3-5), C10=(rows2-3,cols0-2), C11=(rows2-3,cols3-5)
    
    # For each Strassen step, we have:
    # U contributions: list of (a_block, a_row, a_col, coeff)
    # V contributions: list of (b_block, b_row, b_col, coeff)  
    # W contributions: list of (c_block, c_row, c_col, coeff)
    # where a_row/a_col are within the 2x2 A block
    # b_row/b_col are within the 2x3 B block
    # c_row/c_col are within the 2x3 C block
    
    # Strassen steps: (A_combination, B_combination, C_combination)
    # Each combination is a list of (block_id, row_in_block, col_in_block, coeff)
    
    strassen_steps = [
        # M1 = (A00 + A11)(B00 + B11) = C00 + C11
        {
            'A': [(0, 0, 0, 1), (0, 1, 1, 1), (3, 0, 0, 1), (3, 1, 1, 1)],
            'B': [(0, 0, 0, 1), (0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 1),
                   (3, 0, 0, 1), (3, 0, 1, 1), (3, 1, 0, 1), (3, 1, 1, 1)],
            'C': [(0, 0, 0, 1), (0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 1),
                   (3, 0, 0, 1), (3, 0, 1, 1), (3, 1, 0, 1), (3, 1, 1, 1)]
        },
        # M2 = (A10 + A11)B00 = C10 - C00... wait, let me re-derive.
        # Standard Strassen for 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
    ]
    
    # Let me redo this properly. For each Strassen step, the A and B parts are
    # specific entries within specific blocks. The schoolbook expansion handles
    # the actual multiplication.
    
    # For a 2x2 * 2x3 schoolbook: C[r,k] = sum_s A[r,s]*B[s,k]
    # So for each (r,s,k) triple, we need A[r,s], B[s,k], C[r,k]
    
    # Strassen step definitions - for each step, which (block, row, col) entries of A, B, C
    # participate in each schoolbook term
    
    # M1 = (A00+A11)(B00+B11)
    # A part: A00[r,s] + A11[r,s] for r,s in {0,1}
    # B part: B00[s,k] + B11[s,k] for s in {0,1}, k in {0,1,2}
    # C part: C00[r,k] + C11[r,k] for r in {0,1}, k in {0,1,2}
    
    strassen_defs = [
        # M1 = (A00+A11)(B00+B11) -> C00 + C11
        {
            'A_entries': [(0, 1), (3, 1)],  # (block_id, coeff)
            'B_entries': [(0, 1), (3, 1)],
            'C_entries': [(0, 1), (3, 1)]
        },
        # M2 = (A10+A11)B00 -> C10 + C11... 
        # Actually: C10 = M2 + M4, C11 = M1 - M2 + M3 + M6
        # So M2 contributes to C10 with +1 and C11 with -1
        {
            'A_entries': [(2, 1), (3, 1)],
            'B_entries': [(0, 1)],
            'C_entries': [(2, 1), (3, -1)]
        },
        # M3 = A00(B01-B11) -> C01 + C11
        {
            'A_entries': [(0, 1)],
            'B_entries': [(1, 1), (3, -1)],
            'C_entries': [(1, 1), (3, 1)]
        },
        # M4 = A11(B10-B00) -> C00 + C10
        {
            'A_entries': [(3, 1)],
            'B_entries': [(2, 1), (0, -1)],
            'C_entries': [(0, 1), (2, 1)]
        },
        # M5 = (A00+A01)B11 -> -C00 + C01
        {
            'A_entries': [(0, 1), (1, 1)],
            'B_entries': [(3, 1)],
            'C_entries': [(0, -1), (1, 1)]
        },
        # M6 = (A10-A00)(B00+B01) -> C11
        {
            'A_entries': [(2, 1), (0, -1)],
            'B_entries': [(0, 1), (1, 1)],
            'C_entries': [(3, 1)]
        },
        # M7 = (A01-A11)(B10+B11) -> C00
        {
            'A_entries': [(1, 1), (3, -1)],
            'B_entries': [(2, 1), (3, 1)],
            'C_entries': [(0, 1)]
        },
    ]
    
    # Block to global index mapping:
    # A blocks (2x2 each): 
    #   Block 0 (A00): rows 0-1, cols 0-1 -> U indices 0,1,4,5
    #   Block 1 (A01): rows 0-1, cols 2-3 -> U indices 2,3,6,7
    #   Block 2 (A10): rows 2-3, cols 0-1 -> U indices 8,9,12,13
    #   Block 3 (A11): rows 2-3, cols 2-3 -> U indices 10,11,14,15
    # U index = row*4 + col
    
    # B blocks (2x3 each):
    #   Block 0 (B00): rows 0-1, cols 0-2 -> V indices 0,1,2,6,7,8
    #   Block 1 (B01): rows 0-1, cols 3-5 -> V indices 3,4,5,9,10,11
    #   Block 2 (B10): rows 2-3, cols 0-2 -> V indices 12,13,14,18,19,20
    #   Block 3 (B11): rows 2-3, cols 3-5 -> V indices 15,16,17,21,22,23
    # V index = row*6 + col
    
    # C blocks (2x3 each):
    #   Block 0 (C00): rows 0-1, cols 0-2
    #   Block 1 (C01): rows 0-1, cols 3-5
    #   Block 2 (C10): rows 2-3, cols 0-2
    #   Block 3 (C11): rows 2-3, cols 3-5
    # W index = col*4 + row (W[k*n+i] convention)
    
    def a_block_to_global(block_id, r, s):
        """A block (block_id, local_row, local_col) -> global U index"""
        if block_id == 0:    # A00
            return r * 4 + s
        elif block_id == 1:  # A01
            return r * 4 + (s + 2)
        elif block_id == 2:  # A10
            return (r + 2) * 4 + s
        elif block_id == 3:  # A11
            return (r + 2) * 4 + (s + 2)
    
    def b_block_to_global(block_id, s, k):
        """B block (block_id, local_row=s, local_col=k) -> global V index"""
        if block_id == 0:    # B00
            return s * 6 + k
        elif block_id == 1:  # B01
            return s * 6 + (k + 3)
        elif block_id == 2:  # B10
            return (s + 2) * 6 + k
        elif block_id == 3:  # B11
            return (s + 2) * 6 + (k + 3)
    
    def c_block_to_global(block_id, r, k):
        """C block (block_id, local_row=r, local_col=k) -> global W index"""
        if block_id == 0:    # C00
            return k * 4 + r
        elif block_id == 1:  # C01
            return (k + 3) * 4 + r
        elif block_id == 2:  # C10
            return k * 4 + (r + 2)
        elif block_id == 3:  # C11
            return (k + 3) * 4 + (r + 2)
    
    # For each Strassen step and each schoolbook term (r,s,k):
    # The A part is a linear combination of A block entries at position (r,s)
    # The B part is a linear combination of B block entries at position (s,k)
    # The C part is a linear combination of C block entries at position (r,k)
    
    for i, step in enumerate(strassen_defs):
        for r in range(2):       # output row within block
            for s in range(2):   # summation index
                for k in range(3):  # output column within block
                    c_global = i * 12 + (r * 6 + s * 3 + k)
                    
                    # Fill U: linear combination of A block entries at (r,s)
                    for block_id, coeff in step['A_entries']:
                        u_idx = a_block_to_global(block_id, r, s)
                        U[u_idx, c_global] += coeff
                    
                    # Fill V: linear combination of B block entries at (s,k)
                    for block_id, coeff in step['B_entries']:
                        v_idx = b_block_to_global(block_id, s, k)
                        V[v_idx, c_global] += coeff
                    
                    # Fill W: linear combination of C block entries at (r,k)
                    for block_id, coeff in step['C_entries']:
                        w_idx = c_block_to_global(block_id, r, k)
                        W[w_idx, c_global] += coeff
                    
                    # Time guard
                    if time.time() - start_time > DEADLINE - 5:
                        break
                if time.time() - start_time > DEADLINE - 5:
                    break
            if time.time() - start_time > DEADLINE - 5:
                break
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # Convert to list-of-lists format
    U_list = [[int(x) if x == int(x) else x for x in row] for row in U.tolist()]
    V_list = [[int(x) if x == int(x) else x for x in row] for row in V.tolist()]
    W_list = [[int(x) if x == int(x) else x for x in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
