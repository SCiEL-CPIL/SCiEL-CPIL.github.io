"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    U_rows = n * m  # 6
    V_rows = m * p  # 30
    W_rows = p * n  # 20

    U_list = []
    V_list = []
    W_list = []

    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    # Part 1: First 2 columns (k=0,1) using Strassen for 2x2 + schoolbook for 2x1x2
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B1 = [[b00, b01], [b10, b11], [b20, b21]]
    # C1 = A*B1 (2x2)

    # Strassen for A1 (2x2) * B11 (2x2) where A1 = [[a00,a01],[a10,a11]], B11 = [[b00,b01],[b10,b11]]
    # M1 = (a00+a11)*(b00+b11)
    u1 = [0]*U_rows; u1[0] = 1; u1[3] = 1  # a00 + a11
    v1 = [0]*V_rows; v1[0] = 1; v1[1] = 1  # b00 + b11
    w1 = [0]*W_rows; w1[0] = 1; w1[1] = 1; w1[2] = 1; w1[3] = 1  # C00 + C01 + C10 + C11
    add_term(u1, v1, w1)

    # M2 = (a10+a11)*b00
    u2 = [0]*U_rows; u2[3] = 1; u2[4] = 1  # a10 + a11
    v2 = [0]*V_rows; v2[0] = 1  # b00
    w2 = [0]*W_rows; w2[2] = 1; w2[3] = 1  # C10 + C11
    add_term(u2, v2, w2)

    # M3 = a00*(b01-b11)
    u3 = [0]*U_rows; u3[0] = 1  # a00
    v3 = [0]*V_rows; v3[1] = 1; v3[4] = -1  # b01 - b11
    w3 = [0]*W_rows; w3[1] = 1  # C01
    add_term(u3, v3, w3)

    # M4 = a11*(b10-b00)
    u4 = [0]*U_rows; u4[4] = 1  # a11
    v4 = [0]*V_rows; v4[2] = 1; v4[0] = -1  # b10 - b00
    w4 = [0]*W_rows; w4[2] = 1  # C10
    add_term(u4, v4, w4)

    # M5 = (a00+a01)*b11
    u5 = [0]*U_rows; u5[0] = 1; u5[1] = 1  # a00 + a01
    v5 = [0]*V_rows; v5[4] = 1  # b11
    w5 = [0]*W_rows; w5[1] = 1; w5[3] = 1  # C01 + C11
    add_term(u5, v5, w5)

    # M6 = (a10-a00)*(b00+b01)
    u6 = [0]*U_rows; u6[3] = 1; u6[0] = -1  # a10 - a00
    v6 = [0]*V_rows; v6[0] = 1; v6[1] = 1  # b00 + b01
    w6 = [0]*W_rows; w6[3] = 1  # C11
    add_term(u6, v6, w6)

    # M7 = (a01-a11)*(b10+b11)
    u7 = [0]*U_rows; u7[1] = 1; u7[4] = -1  # a01 - a11
    v7 = [0]*V_rows; v7[2] = 1; v7[4] = 1  # b10 + b11
    w7 = [0]*W_rows; w7[0] = 1; w7[2] = 1  # C00 + C10
    add_term(u7, v7, w7)

    # Now we need to combine the Strassen terms to get C11, C12, C21, C22
    # C00 (C11 in 1-indexed) = M1 - M5 + M7
    # C01 (C12 in 1-indexed) = M3 + M5
    # C10 (C21 in 1-indexed) = M2 + M4
    # C11 (C22 in 1-indexed) = M1 - M2 + M6

    # Wait, the above add_term calls are adding individual rank-1 tensors, not the combinations.
    # I need to think about this differently. The Strassen algorithm gives 7 multiplications,
    # but each multiplication corresponds to a rank-1 tensor in the decomposition.
    # However, the way I've written it, I'm adding 7 separate rank-1 tensors where each
    # has a specific U, V, W pattern. But that's not quite right for a tensor decomposition.

    # Let me reconsider. In a tensor decomposition, each term is a rank-1 tensor U_r * V_r * W_r.
    # For Strassen, we have 7 terms, but each term involves products of linear combinations.
    # The key insight is that each Strassen term M_r corresponds to a rank-1 tensor where:
    # U_r is the coefficient vector for A in the linear combination
    # V_r is the coefficient vector for B in the linear combination  
    # W_r is the coefficient vector for C that this term contributes to

    # Let me redo this properly.

    U_list = []
    V_list = []
    W_list = []

    # M1 = (a00+a11)*(b00+b11) contributes to C00, C01, C10, C11
    # In the output, this term adds to C[k*n+i] for the relevant (k,i) pairs
    # For C1 (first 2 columns), the output indices are:
    # C00 -> W[0*2+0] = W[0]
    # C01 -> W[0*2+1] = W[1]
    # C10 -> W[1*2+0] = W[2]
    # C11 -> W[1*2+1] = W[3]

    # M1 = (a00+a11)*(b00+b11)
    # This product contributes to all 4 elements of C1 with coefficient 1 each
    # But wait, in Strassen, M1 contributes to C00, C01, C10, C11 with specific signs
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6

    # So M1 contributes +1 to C00 and +1 to C11
    u = [0]*U_rows; u[0] = 1; u[4] = 1  # a00 + a11
    v = [0]*V_rows; v[0] = 1; v[4] = 1  # b00 + b11
    w = [0]*W_rows; w[0] = 1; w[3] = 1  # C00 + C11
    add_term(u, v, w)

    # M2 = (a10+a11)*b00
    # M2 contributes -1 to C11
    u = [0]*U_rows; u[3] = 1; u[4] = 1  # a10 + a11
    v = [0]*V_rows; v[0] = 1  # b00
    w = [0]*W_rows; w[3] = -1  # -C11
    add_term(u, v, w)

    # M3 = a00*(b01-b11)
    # M3 contributes +1 to C01
    u = [0]*U_rows; u[0] = 1  # a00
    v = [0]*V_rows; v[1] = 1; v[4] = -1  # b01 - b11
    w = [0]*W_rows; w[1] = 1  # C01
    add_term(u, v, w)

    # M4 = a11*(b10-b00)
    # M4 contributes +1 to C10
    u = [0]*U_rows; u[4] = 1  # a11
    v = [0]*V_rows; v[2] = 1; v[0] = -1  # b10 - b00
    w = [0]*W_rows; w[2] = 1  # C10
    add_term(u, v, w)

    # M5 = (a00+a01)*b11
    # M5 contributes -1 to C00 and +1 to C01
    u = [0]*U_rows; u[0] = 1; u[1] = 1  # a00 + a01
    v = [0]*V_rows; v[4] = 1  # b11
    w = [0]*W_rows; w[0] = -1; w[1] = 1  # -C00 + C01
    add_term(u, v, w)

    # M6 = (a10-a00)*(b00+b01)
    # M6 contributes +1 to C11
    u = [0]*U_rows; u[3] = 1; u[0] = -1  # a10 - a00
    v = [0]*V_rows; v[0] = 1; v[1] = 1  # b00 + b01
    w = [0]*W_rows; w[3] = 1  # C11
    add_term(u, v, w)

    # M7 = (a01-a11)*(b10+b11)
    # M7 contributes +1 to C00 and +1 to C10
    u = [0]*U_rows; u[1] = 1; u[4] = -1  # a01 - a11
    v = [0]*V_rows; v[2] = 1; v[4] = 1  # b10 + b11
    w = [0]*W_rows; w[0] = 1; w[2] = 1  # C00 + C10
    add_term(u, v, w)

    # Now add the schoolbook terms for A2*B12 (the 3rd row of A times 3rd row of B1)
    # A2 = [a02, a12], B12 = [b20, b21]
    # C00 += a02*b20, C01 += a02*b21, C10 += a12*b20, C11 += a12*b21

    # Term: a02*b20
    u = [0]*U_rows; u[2] = 1  # a02
    v = [0]*V_rows; v[6] = 1  # b20 (row 2 of B, col 0)
    w = [0]*W_rows; w[0] = 1  # C00
    add_term(u, v, w)

    # Term: a02*b21
    u = [0]*U_rows; u[2] = 1  # a02
    v = [0]*V_rows; v[7] = 1  # b21 (row 2 of B, col 1)
    w = [0]*W_rows; w[1] = 1  # C01
    add_term(u, v, w)

    # Term: a12*b20
    u = [0]*U_rows; u[5] = 1  # a12
    v = [0]*V_rows; v[6] = 1  # b20
    w = [0]*W_rows; w[2] = 1  # C10
    add_term(u, v, w)

    # Term: a12*b21
    u = [0]*U_rows; u[5] = 1  # a12
    v = [0]*V_rows; v[7] = 1  # b21
    w = [0]*W_rows; w[3] = 1  # C11
    add_term(u, v, w)

    # Part 2: Remaining 8 columns (k=2 to k=9) using schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                u = [0]*U_rows; u[i*m+j] = 1
                v = [0]*V_rows; v[j*p+k] = 1
                w = [0]*W_rows; w[k*n+i] = 1
                add_term(u, v, w)

    # Convert to list of lists format
    rank = len(U_list)
    U_out = [[U_list[t][r] for t in range(rank)] for r in range(U_rows)]
    V_out = [[V_list[t][r] for t in range(rank)] for r in range(V_rows)]
    W_out = [[W_list[t][r] for t in range(rank)] for r in range(W_rows)]

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
