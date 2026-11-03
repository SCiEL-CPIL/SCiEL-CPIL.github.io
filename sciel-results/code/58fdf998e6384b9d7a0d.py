import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank is sum of ranks of components
    # M_3,3,3 (Rank 27) + M_3,3,2 (Rank 17)
    # M_3,3,2 = M_3,2,2 (Rank 11) + M_3,1,2 (Rank 6)
    # M_3,2,2 = M_2,2,2 (Rank 7) + M_1,2,2 (Rank 4)
    # Total 27 + 7 + 4 + 6 = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val_u=1, val_v=1, val_w=1):
        nonlocal col
        U[u_idx][col] += val_u
        V[v_idx][col] += val_v
        W[w_idx][col] += val_w
        col += 1

    # --- Part 1: M_3,3,3 (Schoolbook, Rank 27) ---
    # i in 0..2, j in 0..2, k in 0..2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u_idx = i * 3 + j
                v_idx = j * 3 + k
                w_idx = k * 3 + i
                add_term(u_idx, v_idx, w_idx)

    # --- Part 2: M_3,3,2 (Rank 17) ---
    # k in 3..4 (offset 3)
    # Split j into J1={0,1} and J2={2}
    
    # --- Part 2a: j in {0,1} (M_3,2,2, Rank 11) ---
    # Split i into I1={0,1} and I2={2}
    
    # --- Part 2a-i: i in {0,1}, j in {0,1} (M_2,2,2, Rank 7, Strassen) ---
    # A: rows 0,1; cols 0,1. B: rows 0,1; cols 3,4. C: rows 0,1; cols 3,4.
    # Strassen Multiplications
    # M1 = (A00+A11)(B00+B11)
    add_term(0*3+0, 0*3+3, 0, 1, 1, 0) # Dummy, we need to accumulate into W later? 
    # No, the "add_term" adds to a specific column. 
    # Strassen requires combining terms in W.
    # It is easier to generate the rank-1 terms for Strassen directly.
    
    # Let's reset the approach for Strassen.
    # Strassen uses 7 intermediate products.
    # We need to define the U, V, W vectors for each of the 7 products.
    
    # Reset col to start of Part 2a-i? No, we just continue.
    # We need to undo the dummy add_term above. 
    # Better to plan the columns.
    
    # Let's restart the construction cleanly.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val_u=1, val_v=1, val_w=1):
        nonlocal col
        U[u_idx][col] += val_u
        V[v_idx][col] += val_v
        W[w_idx][col] += val_w
        col += 1

    # --- Part 1: M_3,3,3 (Schoolbook, Rank 27) ---
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u_idx = i * 3 + j
                v_idx = j * 3 + k
                w_idx = k * 3 + i
                add_term(u_idx, v_idx, w_idx)
                
    # --- Part 2: M_3,3,2 (Rank 17) ---
    # k indices are 3, 4.
    
    # --- Part 2a: j in {0,1} (Rank 11) ---
    # --- Part 2a-i: i in {0,1}, j in {0,1} (Strassen, Rank 7) ---
    # A indices: (0,0), (0,1), (1,0), (1,1) -> u_idx: 0, 1, 3, 4
    # B indices: (0,3), (0,4), (1,3), (1,4) -> v_idx: 3, 4, 6, 7
    # C indices: (0,3), (0,4), (1,3), (1,4) -> w_idx: 9, 10, 12, 13 (k*n+i: 3*3+0=9, 4*3+0=12, 3*3+1=10, 4*3+1=13)
    
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 3, 0, 1, 1, 0) # Wrong W index logic in add_term? 
    # W index is k*n+i.
    # For C00 (i=0, k=3): w_idx = 3*3+0 = 9.
    # For C01 (i=0, k=4): w_idx = 4*3+0 = 12.
    # For C10 (i=1, k=3): w_idx = 3*3+1 = 10.
    # For C11 (i=1, k=4): w_idx = 4*3+1 = 13.
    
    # We need to add contributions to W for each C element.
    # The "add_term" function adds 1 to a specific W entry.
    # For Strassen, the same product M1 contributes to multiple C entries.
    # So we need multiple calls to add_term for the same U,V but different W?
    # No, a rank-1 tensor is $u \otimes v \otimes w$.
    # If $w$ has multiple non-zero entries, it's not rank 1 in the standard basis?
    # Yes, $w$ is a vector. If $w$ has entries at 9 and 12, it's a single vector with two non-zeros.
    # So one column in U, V, W can have multiple non-zeros in W.
    
    # Let's redefine add_term to allow arbitrary W vector?
    # Or just call add_term multiple times? No, that increases rank.
    # We must construct the vectors directly.
    
    # Let's clear and use a more flexible builder.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_rank1_term(u_dict, v_dict, w_dict):
        """
        u_dict: {index: value}
        v_dict: {index: value}
        w_dict: {index: value}
        """
        nonlocal col
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
        col += 1

    # --- Part 1: M_3,3,3 (Schoolbook, Rank 27) ---
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u_idx = i * 3 + j
                v_idx = j * 3 + k
                w_idx = k * 3 + i
                add_rank1_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})

    # --- Part 2: M_3,3,2 (Rank 17) ---
    
    # --- Part 2a: j in {0,1} (Rank 11) ---
    # --- Part 2a-i: i in {0,1}, j in {0,1} (Strassen, Rank 7) ---
    
    # Indices:
    # A: 00(0), 01(1), 10(3), 11(4)
    # B: 03(3), 04(4), 13(6), 14(7)
    # C: 03(9), 04(12), 13(10), 14(13)
    
    # M1 = (A00+A11)(B00+B11)
    add_rank1_term(
        {0: 1, 4: 1}, 
        {3: 1, 6: 1}, 
        {9: 1, 13: 1} # C00 and C11? 
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # So M1 contributes to C00 and C11.
    )
    
    # M2 = (A10+A11)B00
    add_rank1_term(
        {3: 1, 4: 1}, 
        {3: 1}, 
        {10: 1, 13: 1} # C10 and C11
    )
    
    # M3 = A00(B01-B11)
    add_rank1_term(
        {0: 1}, 
        {4: 1, 6: -1}, 
        {12: 1, 13: 1} # C01 and C11
    )
    
    # M4 = A11(B10+B11)
    add_rank1_term(
        {4: 1}, 
        {6: 1, 7: 1}, 
        {9: 1, 10: 1} # C00 and C10
    )
    
    # M5 = (A00+A01)B11
    add_rank1_term(
        {0: 1, 1: 1}, 
        {7: 1}, 
        {9: -1, 12: 1} # C00 (neg) and C01
    )
    
    # M6 = (A10-A00)(B00+B01)
    add_rank1_term(
        {3: 1, 0: -1}, 
        {3: 1, 4: 1}, 
        {13: 1} # C11
    )
    
    # M7 = (A01-A11)(B10+B11)
    add_rank1_term(
        {1: 1, 4: -1}, 
        {6: 1, 7: 1}, 
        {9: 1} # C00
    )
    
    # --- Part 2a-ii: i=2, j in {0,1} (Rank 4) ---
    # A: 20(6), 21(7)
    # B: 03(3), 04(4), 13(6), 14(7)
    # C: 23(15), 24(18)
    
    # C23 = A20*B03 + A21*B13
    # C24 = A20*B04 + A21*B14
    
    add_rank1_term({6: 1}, {3: 1}, {15: 1})
    add_rank1_term({7: 1}, {6: 1}, {15: 1})
    add_rank1_term({6: 1}, {4: 1}, {18: 1})
    add_rank1_term({7: 1}, {7: 1}, {18: 1})
    
    # --- Part 2b: j=2 (Rank 6) ---
    # A: 02(2), 12(5), 22(8)
    # B: 23(9), 24(10)
    # C: 03(9), 04(12), 13(10), 14(13), 23(15), 24(18)
    
    # C03 = A02*B23
    add_rank1_term({2: 1}, {9: 1}, {9: 1})
    # C04 = A02*B24
    add_rank1_term({2: 1}, {10: 1}, {12: 1})
    # C13 = A12*B23
    add_rank1_term({5: 1}, {9: 1}, {10: 1})
    # C14 = A12*B24
    add_rank1_term({5: 1}, {10: 1}, {13: 1})
    # C23 = A22*B23
    add_rank1_term({8: 1}, {9: 1}, {15: 1})
    # C24 = A22*B24
    add_rank1_term({8: 1}, {10: 1}, {18: 1})
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
