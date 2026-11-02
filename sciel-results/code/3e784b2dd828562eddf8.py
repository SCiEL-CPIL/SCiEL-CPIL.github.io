"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Hybrid Strassen/Schoolbook decomposition of rank 39 for 2x3x7.
    # Split m=3 into m1=2, m2=1.
    # Part 1: 2x2x7 (A[0:2,0:2] * B[0:2,:])
    #   Split p=7 into (2,2,2,1).
    #   Three 2x2x2 blocks: Strassen rank 7 each -> 21
    #   One 2x2x1 block: schoolbook rank 4 -> 4
    #   Total part 1: 25
    # Part 2: 2x1x7 (A[0:2,2] * B[2,:])
    #   Schoolbook rank 14
    # Total: 39
    n, m, p = DIMENSIONS
    rank = 39
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 39
    V = [[0.0] * rank for _ in range(m * p)]  # 21 x 39
    W = [[0.0] * rank for _ in range(p * n)]  # 14 x 39

    col = 0

    # Part 1: 2x2x7
    # A indices: i in 0..1, j in 0..1 -> U row i*m+j = i*3+j
    # B indices: j in 0..1, k in 0..6 -> V row j*p+k = j*7+k
    # C indices: k in 0..6, i in 0..1 -> W row k*n+i = k*2+i

    # Strassen for 2x2x2 blocks
    # For a 2x2x2 block with A=[[a0,a1],[a2,a3]], B=[[b0,b1],[b2,b3]]
    # Strassen multiplies:
    #   m0 = (a0+a3)(b0+b3)
    #   m1 = (a2+a3)b0
    #   m2 = a0(b1-b3)
    #   m3 = a3(b2-b0)
    #   m4 = (a0+a1)b3
    #   m5 = (a2-a0)(b0+b1)
    #   m6 = (a1-a3)(b2+b3)
    # C = [[m0+m4-m6+m5, m4+m6], [m1+m6, m0-m1+m3]]

    strassen_defs = [
        # (U_coeffs, V_coeffs, W_coeffs) for each of 7 multiplications
        # U_coeffs: list of (row, coeff)
        # V_coeffs: list of (row, coeff)
        # W_coeffs: list of (row, coeff)
        [
            # m0 = (a0+a3)(b0+b3)
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1), (2, -1), (1, 1)]),
            # m1 = (a2+a3)b0
            ([(2, 1), (3, 1)], [(0, 1)], [(3, 1), (1, -1)]),
            # m2 = a0(b1-b3)
            ([(0, 1)], [(1, 1), (3, -1)], [(2, 1)]),
            # m3 = a3(b2-b0)
            ([(3, 1)], [(2, 1), (0, -1)], [(3, 1)]),
            # m4 = (a0+a1)b3
            ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (2, 1)]),
            # m5 = (a2-a0)(b0+b1)
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(2, 1)]),
            # m6 = (a1-a3)(b2+b3)
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, -1), (3, 1)]),
        ]
    ]

    # p blocks: (p_offset, is_strassen)
    p_blocks = [(0, True), (2, True), (4, True), (6, False)]

    for p_off, use_strassen in p_blocks:
        if use_strassen:
            for s_idx in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen_defs[0][s_idx]
                # A local indices: a0=(i=0,j=0)->U row 0, a1=(i=0,j=1)->U row 1
                #                 a2=(i=1,j=0)->U row 3, a3=(i=1,j=1)->U row 4
                # B local indices: b0=(j=0,k=p_off)->V row p_off, b1=(j=0,k=p_off+1)->V row p_off+1
                #                 b2=(j=1,k=p_off)->V row 7+p_off, b3=(j=1,k=p_off+1)->V row 7+p_off+1
                # C local: C[0,0]->W row 2*p_off+0, C[0,1]->W row 2*p_off+1
                #         C[1,0]->W row 2*p_off+2, C[1,1]->W row 2*p_off+3

                # Map local A indices to global U rows
                local_a_to_global = {0: 0, 1: 1, 2: 3, 3: 4}
                local_b_to_global = {0: p_off, 1: p_off+1, 2: 7+p_off, 3: 7+p_off+1}
                local_c_to_global = {0: 2*p_off, 1: 2*p_off+1, 2: 2*p_off+2, 3: 2*p_off+3}

                for (local_a, coeff) in u_coeffs:
                    U[local_a_to_global[local_a]][col] += coeff
                for (local_b, coeff) in v_coeffs:
                    V[local_b_to_global[local_b]][col] += coeff
                for (local_c, coeff) in w_coeffs:
                    W[local_c_to_global[local_c]][col] += coeff
                col += 1
        else:
            # 2x2x1 schoolbook: 4 multiplications
            # C[i][k] = A[i][0]*B[0][k] + A[i][1]*B[1][k] for k=p_off
            # m0 = a0*b0, m1 = a0*b2, m2 = a3*b0, m3 = a3*b2
            # C[0][0] = m0, C[0][1] = m1 (wait, k is single here)
            # Actually for 2x2x1: k is fixed at p_off
            # C[i][k] = sum_j A[i][j] B[j][k]
            # So C[0][k] = A[0][0]B[0][k] + A[0][1]B[1][k]
            #    C[1][k] = A[1][0]B[0][k] + A[1][1]B[1][k]
            # 4 multiplications:
            # m0 = A[0][0]*B[0][k], m1 = A[0][1]*B[1][k]
            # m2 = A[1][0]*B[0][k], m3 = A[1][1]*B[1][k]
            # C[0][k] = m0 + m1, C[1][k] = m2 + m3

            # a0=U[0], a1=U[1], a2=U[3], a3=U[4]
            # b0=V[p_off], b2=V[7+p_off]
            # C[0][k]=W[2*p_off], C[1][k]=W[2*p_off+2]

            # m0 = a0*b0
            U[0][col] += 1
            V[p_off][col] += 1
            W[2*p_off][col] += 1
            col += 1
            # m1 = a1*b2
            U[1][col] += 1
            V[7+p_off][col] += 1
            W[2*p_off][col] += 1
            col += 1
            # m2 = a2*b0
            U[3][col] += 1
            V[p_off][col] += 1
            W[2*p_off+2][col] += 1
            col += 1
            # m3 = a3*b2
            U[4][col] += 1
            V[7+p_off][col] += 1
            W[2*p_off+2][col] += 1
            col += 1

    # Part 2: 2x1x7 schoolbook
    # A[i][2] * B[2][k] for i=0,1 and k=0..6
    # U rows: i*3+2 = 2, 5
    # V rows: 2*7+k = 14+k
    # W rows: k*2+i = 2k, 2k+1

    for k in range(7):
        for i in range(2):
            U[i*3+2][col] += 1
            V[14+k][col] += 1
            W[2*k+i][col] += 1
            col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Convert to integer-like representation (all entries are integers)
    U_out = [[int(round(x)) for x in row] for row in U]
    V_out = [[int(round(x)) for x in row] for row in V]
    W_out = [[int(round(x)) for x in row] for row in W]

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
