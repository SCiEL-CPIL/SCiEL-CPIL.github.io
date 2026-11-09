import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    # Rank 85
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. Outer Product Part: A_R (col 4 of A) * B_RL (row 4 of B, cols 0..3)
    # A[i, 4] * B[4, k] -> C[i, k]
    for i in range(n):
        for k in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # 2. Dot Product Part: A (all) * B_R (col 4 of B)
    # A[i, j] * B[j, 4] -> C[i, 4]
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    # 3. Strassen Part: A_L (cols 0..3) * B_LL (rows 0..3, cols 0..3)
    # This computes C[0..3, 0..3]
    
    # We define the 7 Strassen products for the 4x4 block.
    # A_L is 4x4, B_LL is 4x4.
    # Split into 2x2 blocks of 2x2 matrices.
    
    # Define linear forms for the 7 products.
    # Each form is a dict: {(i, j): coeff} for A, {(j, k): coeff} for B.
    
    # p1 = (A11 + A22) * (B11 + B22)
    # A11: (0,0),(0,1),(1,0),(1,1)
    # A22: (2,2),(2,3),(3,2),(3,3)
    # B11: (0,0),(0,1),(1,0),(1,1)
    # B22: (2,2),(2,3),(3,2),(3,3)
    
    def get_block_indices(block_type, size=2):
        # block_type: '11', '12', '21', '22'
        # Returns list of (i, j)
        if block_type == '11':
            return [(0,0), (0,1), (1,0), (1,1)]
        elif block_type == '12':
            return [(0,2), (0,3), (1,2), (1,3)]
        elif block_type == '21':
            return [(2,0), (2,1), (3,0), (3,1)]
        elif block_type == '22':
            return [(2,2), (2,3), (3,2), (3,3)]

    # We need to construct the 7 terms.
    # Term structure: (A_coeff_dict, B_coeff_dict, W_contrib_dict)
    # W_contrib_dict maps (r, c) in 4x4 block to weight.
    
    terms = []
    
    # p1
    A1 = {idx: 1 for idx in get_block_indices('11')}
    A2 = {idx: 1 for idx in get_block_indices('22')}
    A_p1 = {}
    for idx in A1: A_p1[idx] = A_p1.get(idx, 0) + 1
    for idx in A2: A_p1[idx] = A_p1.get(idx, 0) + 1
    
    B1 = {idx: 1 for idx in get_block_indices('11')}
    B2 = {idx: 1 for idx in get_block_indices('22')}
    B_p1 = {}
    for idx in B1: B_p1[idx] = B_p1.get(idx, 0) + 1
    for idx in B2: B_p1[idx] = B_p1.get(idx, 0) + 1
    
    # p1 contributes to M00, M11
    # M00 = p1 + p4 - p5 + p7
    # M11 = p1 - p2 + p3 + p6
    W_p1 = {(0,0): 1, (3,3): 1} # M00 is top-left of 4x4, M11 is bottom-right.
    # Wait, M00 corresponds to C[0:2, 0:2]. M11 to C[2:4, 2:4].
    # So indices in 4x4 block:
    # (0,0) is C[0,0]
    # (3,3) is C[3,3]
    # But M00 is a 2x2 matrix.
    # The Strassen formula gives the 2x2 result M.
    # M[0,0] corresponds to C[0,0]
    # M[0,1] corresponds to C[0,1]
    # M[1,0] corresponds to C[1,0]
    # M[1,1] corresponds to C[1,1]
    #
    # So for p1, it contributes to M[0,0] and M[1,1] of the 2x2 result?
    # No, the 7 products p1..p7 are scalars (in the context of 2x2 Strassen, they are 2x2 matrix products).
    # In our 4x4 case, p1..p7 are 2x2 matrix products.
    # So p1 is a 2x2 matrix.
    # The formula C11 = M1 + M4 - M5 + M7 refers to the elements of the 2x2 result.
    # So p1 contributes to ALL elements of the 2x2 block it belongs to?
    # No.
    # In 2x2 Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Here M1..M7 are 2x2 matrices.
    # So M1 contributes to C11, C22.
    # M4 contributes to C11, C21.
    # etc.
    #
    # So for the 4x4 Strassen, we are computing 4 2x2 blocks of C.
    # Let's call them C_TL, C_TR, C_BL, C_BR.
    # C_TL = M1 + M4 - M5 + M7 (where M are 2x2 matrices)
    # C_TR = M3 + M5
    # C_BL = M2 + M4
    # C_BR = M1 - M2 + M3 + M6
    #
    # So each Mk (which is a 2x2 matrix product of linear forms) contributes to specific 2x2 blocks of C.
    # And within each 2x2 block, it contributes to all 4 elements with the same coefficient?
    # No, Mk is a 2x2 matrix.
    # C_TL = M1 + ...
    # This means the entire 2x2 matrix M1 is added to C_TL.
    # So yes, each element of M1 is added to the corresponding element of C_TL.
    
    # So, for each of the 7 products:
    # 1. Determine A_form, B_form.
    # 2. Determine which C_blocks it contributes to, and with what weight.
    
    # p1: A=A11+A22, B=B11+B22.
    # Contributes to C_TL (weight 1), C_BR (weight 1).
    
    # p2: A=A21+A22, B=B11.
    # Contributes to C_BL (weight 1), C_BR (weight -1).
    
    # p3: A=A11, B=B12-B22.
    # Contributes to C_TR (weight 1), C_BR (weight 1).
    
    # p4: A=A22, B=B21-B11.
    # Contributes to C_TL (weight 1), C_BL (weight 1).
    
    # p5: A=A11+A12, B=B22.
    # Contributes to C_TL (weight -1), C_TR (weight 1).
    
    # p6: A=A21-A11, B=B11+B12.
    # Contributes to C_BR (weight 1).
    
    # p7: A=A12-A22, B=B21+B22.
    # Contributes to C_TL (weight 1).
    
    # Let's define these properly.
    
    def add_strassen_term(A_indices, B_indices, W_blocks):
        nonlocal col
        # A_indices: list of (i, j, coeff)
        # B_indices: list of (j, k, coeff)
        # W_blocks: dict block_name -> weight
        # block_name: 'TL', 'TR', 'BL', 'BR'
        
        # Expand product
        for (i, j, ca) in A_indices:
            for (jj, k, cb) in B_indices:
                if j == jj:
                    # Term A[i, j] * B[j, k]
                    U[i * m + j][col] = ca
                    V[j * p + k][col] = cb
                    for block, weight in W_blocks.items():
                        if weight == 0: continue
                        if block == 'TL':
                            # C[0:2, 0:2]
                            for r in range(2):
                                for c in range(2):
                                    W[(r) * n + (c)][col] += weight
                        elif block == 'TR':
                            # C[0:2, 2:4]
                            for r in range(2):
                                for c in range(2):
                                    W[(r) * n + (2 + c)][col] += weight
                        elif block == 'BL':
                            # C[2:4, 0:2]
                            for r in range(2):
                                for c in range(2):
                                    W[(2 + r) * n + (c)][col] += weight
                        elif block == 'BR':
                            # C[2:4, 2:4]
                            for r in range(2):
                                for c in range(2):
                                    W[(2 + r) * n + (2 + c)][col] += weight
                    col += 1

    # Define blocks
    A11 = [(0,0,1), (0,1,1), (1,0,1), (1,1,1)]
    A12 = [(0,2,1), (0,3,1), (1,2,1), (1,3,1)]
    A21 = [(2,0,1), (2,1,1), (3,0,1), (3,1,1)]
    A22 = [(2,2,1), (2,3,1), (3,2,1), (3,3,1)]
    
    B11 = [(0,0,1), (0,1,1), (1,0,1), (1,1,1)]
    B12 = [(0,2,1), (0,3,1), (1,2,1), (1,3,1)]
    B21 = [(2,0,1), (2,1,1), (3,0,1), (3,1,1)]
    B22 = [(2,2,1), (2,3,1), (3,2,1), (3,3,1)]
    
    def merge(A1, A2, sign2=1):
        res = {}
        for i, j, c in A1:
            res[(i,j)] = c
        for i, j, c in A2:
            res[(i,j)] = res.get((i,j), 0) + sign2 * c
        return [(i, j, c) for (i, j), c in res.items() if c != 0]
    
    def mergeB(B1, B2, sign2=1):
        res = {}
        for j, k, c in B1:
            res[(j,k)] = c
        for j, k, c in B2:
            res[(j,k)] = res.get((j,k), 0) + sign2 * c
        return [(j, k, c) for (j, k), c in res.items() if c != 0]

    # p1
    A_p1 = merge(A11, A22)
    B_p1 = mergeB(B11, B22)
    add_strassen_term(A_p1, B_p1, {'TL': 1, 'BR': 1})
    
    # p2
    A_p2 = merge(A21, A22)
    B_p2 = B11
    add_strassen_term(A_p2, B_p2, {'BL': 1, 'BR': -1})
    
    # p3
    A_p3 = A11
    B_p3 = mergeB(B12, B22, -1)
    add_strassen_term(A_p3, B_p3, {'TR': 1, 'BR': 1})
    
    # p4
    A_p4 = A22
    B_p4 = mergeB(B21, B11, -1)
    add_strassen_term(A_p4, B_p4, {'TL': 1, 'BL': 1})
    
    # p5
    A_p5 = merge(A11, A12)
    B_p5 = B22
    add_strassen_term(A_p5, B_p5, {'TL': -1, 'TR': 1})
    
    # p6
    A_p6 = merge(A21, A11, -1)
    B_p6 = mergeB(B11, B12)
    add_strassen_term(A_p6, B_p6, {'BR': 1})
    
    # p7
    A_p7 = merge(A12, A22, -1)
    B_p7 = mergeB(B21, B22)
    add_strassen_term(A_p7, B_p7, {'TL': 1})

    return {"U": U, "V": V, "W": W}
