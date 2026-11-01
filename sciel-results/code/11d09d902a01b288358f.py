"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Use a recursive decomposition based on Strassen for 2x2x2 base cases
    # combined with schoolbook for small remaining parts.
    #
    # Dimensions: n=2, m=4, p=5
    # We split the middle dimension m=4 into two blocks of size 2: m1=2, m2=2
    # We split the last dimension p=5 into two blocks: p1=4, p2=1
    #
    # A is 2x4, B is 4x5, C is 2x5
    # C = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :]
    #
    # For each term: (2x2) @ (2x5) -> 2x5
    # For a 2x2 @ 2x5, we can further split the 5 columns into 4+1.
    # (2x2) @ (2x4) can use Strassen-like decomposition? Actually 2x2x2 Strassen
    # applies to 2x2 @ 2x2. For 2x2 @ 2x4, we split the 4 columns into two 2x2 blocks.
    #
    # So for (2x2) @ (2x5):
    #   (2x2) @ (2x4) + (2x2) @ (2x1)
    #   (2x2) @ (2x4) = (2x2)@(2x2) + (2x2)@(2x2)  -> 2 Strassen blocks of rank 7 each = 14
    #   (2x2) @ (2x1) = schoolbook 2x2x1 = rank 4
    #   Total for one (2x2)@(2x5): 14 + 4 = 18
    #
    # Two such terms (for m1 and m2): 18 + 18 = 36
    #
    # But we can do better. Let's use a more careful construction.
    #
    # Actually, let's just build the Strassen decomposition for each 2x2x2 subproblem
    # and schoolbook for the rest.
    #
    # Full decomposition:
    # C = A0 @ B0 + A1 @ B1
    # where A0 = A[:, 0:2], A1 = A[:, 2:4]
    #       B0 = B[0:2, :], B1 = B[2:4, :]
    #
    # A0 @ B0: (2x2) @ (2x5)
    #   Split B0 into B0a (2x4) and B0b (2x1)
    #   B0a split into B0a1 (2x2) and B0a2 (2x2)
    #   A0 @ B0a1: Strassen rank 7
    #   A0 @ B0a2: Strassen rank 7
    #   A0 @ B0b: schoolbook 2x2x1 rank 4
    #   Subtotal: 18
    #
    # A1 @ B1: same structure, rank 18
    #
    # Total rank: 36
    #
    # Let's implement this.

    n, m, p = DIMENSIONS
    # n=2, m=4, p=5

    # We'll build U, V, W as lists of lists, accumulating columns.
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (20, rank)
    # W has shape (p*n, rank) = (10, rank)

    # We'll build a list of (u_vec, v_vec, w_vec) triples, then pack.
    triples = []

    def add_triple(u_idx_list, v_idx_list, w_idx_list):
        """u_idx_list: list of (index, value) for U
           v_idx_list: list of (index, value) for V
           w_idx_list: list of (index, value) for W"""
        triples.append((u_idx_list, v_idx_list, w_idx_list))

    # Strassen 2x2x2 decomposition
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A@B
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5],
    #      [M2+M4,        M1-M2+M3+M6]]

    def strassen_2x2x2(base_u_offset, base_v_offset, base_w_offset):
        """Add 7 Strassen triples for a 2x2x2 block.
           U indices: base_u_offset + [0,1,2,3] for A[0,0], A[0,1], A[1,0], A[1,1]
           V indices: base_v_offset + [0,1,2,3] for B[0,0], B[0,1], B[1,0], B[1,1]
           W indices: base_w_offset + [0,1,2,3] for C[0,0], C[0,1], C[1,0], C[1,1]
        """
        a, b, c, d = 0, 1, 2, 3
        e, f, g, h = 0, 1, 2, 3
        c00, c01, c10, c11 = 0, 1, 2, 3

        # M1 = (a+d)(e+h)
        add_triple([(base_u_offset+a, 1), (base_u_offset+d, 1)],
                   [(base_v_offset+e, 1), (base_v_offset+h, 1)],
                   [(base_w_offset+c00, 1), (base_w_offset+c11, -1)])

        # M2 = (c+d)e
        add_triple([(base_u_offset+c, 1), (base_u_offset+d, 1)],
                   [(base_v_offset+e, 1)],
                   [(base_w_offset+c10, 1)])

        # M3 = a(f-h)
        add_triple([(base_u_offset+a, 1)],
                   [(base_v_offset+f, 1), (base_v_offset+h, -1)],
                   [(base_w_offset+c01, 1)])

        # M4 = d(g-f)
        add_triple([(base_u_offset+d, 1)],
                   [(base_v_offset+g, 1), (base_v_offset+f, -1)],
                   [(base_w_offset+c00, 1), (base_w_offset+c10, -1)])

        # M5 = (a+b)h
        add_triple([(base_u_offset+a, 1), (base_u_offset+b, 1)],
                   [(base_v_offset+h, 1)],
                   [(base_w_offset+c00, -1), (base_w_offset+c01, 1)])

        # M6 = (c-a)(e+f)
        add_triple([(base_u_offset+c, 1), (base_u_offset+a, -1)],
                   [(base_v_offset+e, 1), (base_v_offset+f, 1)],
                   [(base_w_offset+c11, 1)])

        # M7 = (b-d)(g+h)
        add_triple([(base_u_offset+b, 1), (base_u_offset+d, -1)],
                   [(base_v_offset+g, 1), (base_v_offset+h, 1)],
                   [(base_w_offset+c00, 1), (base_w_offset+c11, 1)])

    # Schoolbook 2x2x1: A (2x2) @ B (2x1) -> C (2x1)
    # C[0] = a*e + b*f
    # C[1] = c*e + d*f
    # 4 terms
    def schoolbook_2x2x1(base_u_offset, base_v_offset, base_w_offset):
        """
        U indices: base_u_offset + [0,1,2,3] for A[0,0], A[0,1], A[1,0], A[1,1]
        V indices: base_v_offset + [0,1] for B[0,0], B[1,0]
        W indices: base_w_offset + [0,1] for C[0,0], C[1,0]
        """
        a, b, c, d = 0, 1, 2, 3
        e, f = 0, 1
        c0, c1 = 0, 1

        # a*e
        add_triple([(base_u_offset+a, 1)], [(base_v_offset+e, 1)], [(base_w_offset+c0, 1)])
        # b*f
        add_triple([(base_u_offset+b, 1)], [(base_v_offset+f, 1)], [(base_w_offset+c0, 1)])
        # c*e
        add_triple([(base_u_offset+c, 1)], [(base_v_offset+e, 1)], [(base_w_offset+c1, 1)])
        # d*f
        add_triple([(base_u_offset+d, 1)], [(base_v_offset+f, 1)], [(base_w_offset+c1, 1)])

    # Now build the full decomposition.
    # U indices: i*m + j where i in [0,1], j in [0,3]
    #   A0: i in [0,1], j in [0,1] -> U indices 0,1,2,3
    #   A1: i in [0,1], j in [2,3] -> U indices 4,5,6,7
    #
    # V indices: j*p + k where j in [0,3], k in [0,4]
    #   B0: j in [0,1], k in [0,4]
    #     B0a: k in [0,3]
    #       B0a1: k in [0,1]
    #       B0a2: k in [2,3]
    #     B0b: k in [4]
    #   B1: j in [2,3], k in [0,4]
    #     Same structure
    #
    # W indices: k*n + i where k in [0,4], i in [0,1]
    #   For C[0:2, 0:5], W index for C[i,k] is k*n+i = k*2+i
    #   C0: columns 0-4 from A0@B0
    #   C1: columns 0-4 from A1@B1
    #   Total C has 5 columns, so W indices 0-9.
    #   Wait, W has shape (p*n, rank) = (10, rank).
    #   W[k*n+i] where k in [0,4], i in [0,1].
    #   For the first product (A0@B0), the result contributes to C[i,k] for all i,k.
    #   For the second product (A1@B1), same.
    #   So W indices are shared between the two products.
    #   W[k*2+i] for k in [0,4], i in [0,1].
    #   For A0@B0a1: C[0:2, 0:2] -> W[0*2+0]=0, W[0*2+1]=1, W[1*2+0]=2, W[1*2+1]=3
    #   For A0@B0a2: C[0:2, 2:4] -> W[2*2+0]=4, W[2*2+1]=5, W[3*2+0]=6, W[3*2+1]=7
    #   For A0@B0b: C[0:2, 4:5] -> W[4*2+0]=8, W[4*2+1]=9
    #   Similarly for A1@B1.

    # A0 @ B0a1: Strassen 2x2x2
    # A0: U[0:4], B0a1: V[j*5+k] for j in [0,1], k in [0,1]
    #   j=0,k=0 -> V[0], j=0,k=1 -> V[1], j=1,k=0 -> V[5], j=1,k=1 -> V[6]
    # B0a1 V indices: [0, 1, 5, 6] -> but Strassen expects contiguous [0,1,2,3] offset.
    # We need to map. Let's handle this carefully.
    #
    # Actually, my strassen_2x2x2 function assumes contiguous V indices.
    # Let me redefine to take explicit index lists.

    def strassen_2x2x2_explicit(u_idxs, v_idxs, w_idxs):
        """u_idxs, v_idxs, w_idxs: lists of 4 indices each for A[0,0],A[0,1],A[1,0],A[1,1] etc."""
        a, b, c, d = u_idxs
        e, f, g, h = v_idxs
        c00, c01, c10, c11 = w_idxs

        add_triple([(a, 1), (d, 1)], [(e, 1), (h, 1)], [(c00, 1), (c11, -1)])
        add_triple([(c, 1), (d, 1)], [(e, 1)], [(c10, 1)])
        add_triple([(a, 1)], [(f, 1), (h, -1)], [(c01, 1)])
        add_triple([(d, 1)], [(g, 1), (f, -1)], [(c00, 1), (c10, -1)])
        add_triple([(a, 1), (b, 1)], [(h, 1)], [(c00, -1), (c01, 1)])
        add_triple([(c, 1), (a, -1)], [(e, 1), (f, 1)], [(c11, 1)])
        add_triple([(b, 1), (d, -1)], [(g, 1), (h, 1)], [(c00, 1), (c11, 1)])

    def schoolbook_2x2x1_explicit(u_idxs, v_idxs, w_idxs):
        """u_idxs: 4 indices for A, v_idxs: 2 indices for B, w_idxs: 2 indices for C."""
        a, b, c, d = u_idxs
        e, f = v_idxs
        c0, c1 = w_idxs

        add_triple([(a, 1)], [(e, 1)], [(c0, 1)])
        add_triple([(b, 1)], [(f, 1)], [(c0, 1)])
        add_triple([(c, 1)], [(e, 1)], [(c1, 1)])
        add_triple([(d, 1)], [(f, 1)], [(c1, 1)])

    # Build indices
    # A0 U indices: [0, 1, 2, 3] (A[0,0], A[0,1], A[1,0], A[1,1])
    u_A0 = [0, 1, 2, 3]
    # A1 U indices: [4, 5, 6, 7] (A[0,2], A[0,3], A[1,2], A[1,3])
    u_A1 = [4, 5, 6, 7]

    # B0a1 V indices: j in [0,1], k in [0,1]
    # B[j,k] -> V[j*5+k]
    # B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[5], B[1,1]=V[6]
    v_B0a1 = [0, 1, 5, 6]

    # B0a2 V indices: j in [0,1], k in [2,3]
    # B[0,2]=V[2], B[0,3]=V[3], B[1,2]=V[7], B[1,3]=V[8]
    v_B0a2 = [2, 3, 7, 8]

    # B0b V indices: j in [0,1], k in [4]
    # B[0,4]=V[4], B[1,4]=V[9]
    v_B0b = [4, 9]

    # B1a1 V indices: j in [2,3], k in [0,1]
    # B[2,0]=V[10], B[2,1]=V[11], B[3,0]=V[15], B[3,1]=V[16]
    v_B1a1 = [10, 11, 15, 16]

    # B1a2 V indices: j in [2,3], k in [2,3]
    # B[2,2]=V[12], B[2,3]=V[13], B[3,2]=V[17], B[3,3]=V[18]
    v_B1a2 = [12, 13, 17, 18]

    # B1b V indices: j in [2,3], k in [4]
    # B[2,4]=V[14], B[3,4]=V[19]
    v_B1b = [14, 19]

    # W indices: C[i,k] -> W[k*2+i]
    # C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    # C[0,2]=W[4], C[0,3]=W[5], C[1,2]=W[6], C[1,3]=W[7]
    # C[0,4]=W[8], C[0,5]=W[9]  (but p=5, so k in [0,4])
    w_C_cols01 = [0, 1, 2, 3]  # C[0,0], C[0,1], C[1,0], C[1,1]
    w_C_cols23 = [4, 5, 6, 7]  # C[0,2], C[0,3], C[1,2], C[1,3]
    w_C_col4 = [8, 9]          # C[0,4], C[1,4]

    # A0 @ B0a1: Strassen
    strassen_2x2x2_explicit(u_A0, v_B0a1, w_C_cols01)

    # A0 @ B0a2: Strassen
    strassen_2x2x2_explicit(u_A0, v_B0a2, w_C_cols23)

    # A0 @ B0b: Schoolbook 2x2x1
    schoolbook_2x2x1_explicit(u_A0, v_B0b, w_C_col4)

    # A1 @ B1a1: Strassen
    strassen_2x2x2_explicit(u_A1, v_B1a1, w_C_cols01)

    # A1 @ B1a2: Strassen
    strassen_2x2x2_explicit(u_A1, v_B1a2, w_C_cols23)

    # A1 @ B1b: Schoolbook 2x2x1
    schoolbook_2x2x1_explicit(u_A1, v_B1b, w_C_col4)

    # Now pack triples into U, V, W matrices
    n_rows_u = n * m  # 8
    n_rows_v = m * p  # 20
    n_rows_w = p * n  # 10
    rank = len(triples)

    U = [[0] * rank for _ in range(n_rows_u)]
    V = [[0] * rank for _ in range(n_rows_v)]
    W = [[0] * rank for _ in range(n_rows_w)]

    for col, (u_list, v_list, w_list) in enumerate(triples):
        for idx, val in u_list:
            U[idx][col] = val
        for idx, val in v_list:
            V[idx][col] = val
        for idx, val in w_list:
            W[idx][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
