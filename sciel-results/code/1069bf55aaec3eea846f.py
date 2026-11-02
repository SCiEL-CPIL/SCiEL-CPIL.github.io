"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 5
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, val_u=1, val_v=1, val_w=1):
        nonlocal rank
        U.append([0] * (rank + 1))
        V.append([0] * (rank + 1))
        W.append([0] * (rank + 1))
        U[u_idx][rank] = val_u
        V[v_idx][rank] = val_v
        W[w_idx][rank] = val_w
        rank += 1

    # 1. i in {0, 1} (2x5x5 part)
    # 1a. j in {0..3}, k in {0..3} (2x4x4 core)
    # Use 4 Strassen blocks (2x2x2 each)
    # Strassen factors for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M4 + M7
    # C21 = M2 + M6 - M5
    # C22 = M1 - M2 + M3 + M6
    
    strassen_u = [
        [1, 1, 0, 0, 0, 0, 0], # M1: A11+A22
        [0, 0, 0, 0, 0, 1, 0], # M2: A21+A22 (Wait, indices)
        # Let's define standard Strassen U, V, W for 2x2x2
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(g-e)
        # M4 = d(f-h)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # i = M1 - M4 + M5 - M7
        # j = M3 + M4 + M7
        # k = M2 + M6 - M5
        # l = M1 - M2 + M3 + M6
        
        # U factors (for A):
        # M1: [1, 0, 1, 0] (a,d)
        # M2: [0, 0, 1, 1] (c,d)
        # M3: [1, 0, 0, 0] (a)
        # M4: [0, 0, 0, 1] (d)
        # M5: [1, 1, 0, 0] (a,b)
        # M6: [-1, 0, 1, 0] (c-a)
        # M7: [0, 1, 0, -1] (b-d)
        
        # V factors (for B):
        # M1: [1, 0, 1, 0] (e,h)
        # M2: [1, 0, 0, 0] (e)
        # M3: [-1, 0, 1, 0] (g-e)
        # M4: [0, 1, 0, -1] (f-h)
        # M5: [0, 0, 0, 1] (h)
        # M6: [1, 1, 0, 0] (e+f)
        # M7: [0, 1, 1, 1] (g+h)
        
        # W factors (for C):
        # i (0,0): [1, 0, 0, 0] for M1, [0, 1, 0, 0] for M4, [0, 0, 1, 0] for M5, [0, 0, 0, 1] for M7
        # i = M1 - M4 + M5 - M7
        # So W_i = [1, -1, 1, -1] for the 4 terms? No, each M is a separate rank term.
        # In the decomposition, each M_r is a separate rank-1 tensor.
        # So we add 7 terms per block.
        
        # Term 1 (M1):
        # U: a=1, d=1. V: e=1, h=1. W: i=1, l=1.
        # C11 += M1, C22 += M1.
        # So W[0,0] += 1, W[1,1] += 1.
        
        # Term 2 (M2):
        # U: c=1, d=1. V: e=1. W: k=1, l=-1.
        # C21 += M2, C22 -= M2.
        
        # Term 3 (M3):
        # U: a=1. V: g=1, e=-1. W: j=1, k=1.
        # C12 += M3, C21 += M3.
        
        # Term 4 (M4):
        # U: d=1. V: f=1, h=-1. W: i=-1, j=1.
        # C11 -= M4, C12 += M4.
        
        # Term 5 (M5):
        # U: a=1, b=1. V: h=1. W: i=1, k=-1.
        # C11 += M5, C21 -= M5.
        
        # Term 6 (M6):
        # U: c=1, a=-1. V: e=1, f=1. W: k=1, l=1.
        # C21 += M6, C22 += M6.
        
        # Term 7 (M7):
        # U: b=1, d=-1. V: g=1, h=1. W: i=-1, j=1.
        # C11 -= M7, C12 += M7.
    ]
    
    # We will implement a helper to add a 2x2x2 Strassen block
    def add_strassen_block(i_offset, j_offset, k_offset):
        nonlocal rank, U, V, W
        # i_offset: 0 or 1 (index in the 2x... matrix)
        # j_offset: 0, 2 (start of 2x2 block in j)
        # k_offset: 0, 2 (start of 2x2 block in k)
        
        # The 7 terms
        # 1. M1
        add_term(i_offset * 5 + j_offset, j_offset * 5 + k_offset, k_offset * 3 + i_offset, 1, 1, 1) # a, e, i
        add_term(i_offset * 5 + j_offset + 1, j_offset * 5 + k_offset + 1, k_offset * 3 + i_offset + 1, 1, 1, 1) # d, h, l (Wait, W index is k*n+i)
        # W index: k * n + i. n=3.
        # For i_offset=0: W[0,0] is k=0, i=0 -> 0*3+0=0. W[1,1] is k=1, i=0 -> 1*3+0=3.
        # Let's re-verify W index.
        # W[k*n+i, r].
        # For block i_offset=0:
        # i=0.
        # C[0,0] (i=0, k=0): W[0*3+0] = W[0].
        # C[0,1] (i=0, k=1): W[1*3+0] = W[3].
        # C[1,0] (i=1, k=0): W[0*3+1] = W[1].
        # C[1,1] (i=1, k=1): W[1*3+1] = W[4].
        
        # Let's rewrite add_term to handle multiple indices if needed, or just call it multiple times.
        # But add_term creates a new rank column.
        # For Strassen, one term (e.g., M1) contributes to multiple entries in C.
        # So for M1, we need to add to W[0] and W[3] (for i=0 block).
        # But U and V are also specific.
        # M1: U has non-zeros at (i, j) and (i+1, j+1).
        # V has non-zeros at (j, k) and (j+1, k+1).
        # W has non-zeros at (k, i) and (k+1, i+1).
        
        # We need a function that adds a single rank term with multiple non-zeros.
        
    def add_rank_term(u_sp, v_sp, w_sp):
        nonlocal rank, U, V, W
        # u_sp, v_sp, w_sp are dicts {index: value}
        U.append([0] * (rank + 1))
        V.append([0] * (rank + 1))
        W.append([0] * (rank + 1))
        for idx, val in u_sp.items():
            U[idx][rank] = val
        for idx, val in v_sp.items():
            V[idx][rank] = val
        for idx, val in w_sp.items():
            W[idx][rank] = val
        rank += 1

    def add_strassen_block(i_off, j_off, k_off):
        # i_off is 0 or 1
        # j_off is 0 or 2
        # k_off is 0 or 2
        
        # Term 1: M1 = (A11+A22)(B11+B22)
        # U: (i_off, j_off)=1, (i_off+1, j_off+1)=1
        # V: (j_off, k_off)=1, (j_off+1, k_off+1)=1
        # W: (k_off, i_off)=1, (k_off+1, i_off+1)=1
        add_rank_term(
            {i_off*5 + j_off: 1, (i_off+1)*5 + j_off+1: 1},
            {j_off*5 + k_off: 1, (j_off+1)*5 + k_off+1: 1},
            {k_off*3 + i_off: 1, (k_off+1)*3 + i_off+1: 1}
        )
        
        # Term 2: M2 = (A21+A22)B11
        # U: (i_off+1, j_off)=1, (i_off+1, j_off+1)=1
        # V: (j_off, k_off)=1
        # W: (k_off, i_off+1)=1, (k_off+1, i_off+1)=-1
        add_rank_term(
            {(i_off+1)*5 + j_off: 1, (i_off+1)*5 + j_off+1: 1},
            {j_off*5 + k_off: 1},
            {k_off*3 + i_off+1: 1, (k_off+1)*3 + i_off+1: -1}
        )
        
        # Term 3: M3 = A11(B21-B11)
        # U: (i_off, j_off)=1
        # V: (j_off+1, k_off)=1, (j_off, k_off)=-1
        # W: (k_off+1, i_off)=1, (k_off, i_off+1)=1
        add_rank_term(
            {i_off*5 + j_off: 1},
            {(j_off+1)*5 + k_off: 1, j_off*5 + k_off: -1},
            {(k_off+1)*3 + i_off: 1, k_off*3 + i_off+1: 1}
        )
        
        # Term 4: M4 = A22(B12-B22)
        # U: (i_off+1, j_off+1)=1
        # V: (j_off, k_off+1)=1, (j_off+1, k_off+1)=-1
        # W: (k_off, i_off)=-1, (k_off+1, i_off)=1
        add_rank_term(
            {(i_off+1)*5 + j_off+1: 1},
            {j_off*5 + k_off+1: 1, (j_off+1)*5 + k_off+1: -1},
            {k_off*3 + i_off: -1, (k_off+1)*3 + i_off: 1}
        )
        
        # Term 5: M5 = (A11+A12)B22
        # U: (i_off, j_off)=1, (i_off, j_off+1)=1
        # V: (j_off+1, k_off+1)=1
        # W: (k_off, i_off)=1, (k_off, i_off+1)=-1
        add_rank_term(
            {i_off*5 + j_off: 1, i_off*5 + j_off+1: 1},
            {(j_off+1)*5 + k_off+1: 1},
            {k_off*3 + i_off: 1, k_off*3 + i_off+1: -1}
        )
        
        # Term 6: M6 = (A21-A11)(B11+B12)
        # U: (i_off+1, j_off)=1, (i_off, j_off)=-1
        # V: (j_off, k_off)=1, (j_off, k_off+1)=1
        # W: (k_off, i_off+1)=1, (k_off+1, i_off+1)=1
        add_rank_term(
            {(i_off+1)*5 + j_off: 1, i_off*5 + j_off: -1},
            {j_off*5 + k_off: 1, j_off*5 + k_off+1: 1},
            {k_off*3 + i_off+1: 1, (k_off+1)*3 + i_off+1: 1}
        )
        
        # Term 7: M7 = (A12-A22)(B21+B22)
        # U: (i_off, j_off+1)=1, (i_off+1, j_off+1)=-1
        # V: (j_off+1, k_off)=1, (j_off+1, k_off+1)=1
        # W: (k_off, i_off)=-1, (k_off+1, i_off)=1
        add_rank_term(
            {i_off*5 + j_off+1: 1, (i_off+1)*5 + j_off+1: -1},
            {(j_off+1)*5 + k_off: 1, (j_off+1)*5 + k_off+1: 1},
            {k_off*3 + i_off: -1, (k_off+1)*3 + i_off: 1}
        )

    # 1a. Core 2x4x4
    for i_off in [0, 1]:
        for j_off in [0, 2]:
            for k_off in [0, 2]:
                add_strassen_block(i_off, j_off, k_off)

    # 1b. Strip j in {0..3}, k=4 (2x4x1)
    # C_i,4 = sum_j A_i,j B_j,4
    # Rank 8
    for i in [0, 1]:
        for j in range(4):
            add_rank_term(
                {i*5 + j: 1},
                {j*5 + 4: 1},
                {4*3 + i: 1}
            )

    # 1c. Strip j=4, k in {0..3} (2x1x4)
    # C_i,k = A_i,4 B_4,k
    # Rank 1?
    # T_i,4,k = A_i,4 B_4,k
    # This is a single rank-1 tensor for the whole strip?
    # U_i,4 = A_i,4. V_4,k = B_4,k.
    # We need one term where U has non-zeros at i=0,1 and j=4.
    # V has non-zeros at j=4 and k=0..3.
    # W has non-zeros at k=0..3 and i=0,1.
    # But U and V are fixed vectors in the decomposition.
    # We can use 1 term.
    # U: {0*5+4: 1, 1*5+4: 1}
    # V: {4*5+0: 1, 4*5+1: 1, 4*5+2: 1, 4*5+3: 1}
    # W: {0*3+0: 1, 1*3+0: 1, 2*3+0: 1, 3*3+0: 1, 0*3+1: 1, 1*3+1: 1, 2*3+1: 1, 3*3+1: 1}
    # This works because the product is separable.
    add_rank_term(
        {0*5+4: 1, 1*5+4: 1},
        {4*5+k: 1 for k in range(4)},
        {k*3+i: 1 for k in range(4) for i in range(2)}
    )

    # 1d. Corner j=4, k=4 (2x1x1)
    # C_i,4 = A_i,4 B_4,4
    # Rank 1
    add_rank_term(
        {0*5+4: 1, 1*5+4: 1},
        {4*5+4: 1},
        {4*3+0: 1, 4*3+1: 1}
    )

    # 2. i=2 (1x5x5 part)
    # Rank 5
    for j in range(5):
        add_rank_term(
            {2*5+j: 1},
            {j*5+k: 1 for k in range(5)},
            {k*3+2: 1 for k in range(5)}
        )
        # Wait, the term is A_2,j B_j,k.
        # U: {2*5+j: 1}
        # V: {j*5+k: 1 for k in 0..4}
        # W: {k*3+2: 1 for k in 0..4}
        # This gives sum_k (1 * 1 * 1) A_2,j B_j,k = A_2,j B_j,k.
        # Summing over j gives the result.

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
