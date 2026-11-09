def add_strassen_block(U, V, W, rank_offset, i_offset, j_offset, k_offset, n, m, p):
    # i_offset: row index in A
    # j_offset: col index in A / row index in B
    # k_offset: col index in B
    # n=2, m=2, p=2 local dimensions
    
    # m1 = (a+d)(e+h)
    # a=A[i0,j0], d=A[i1,j1]
    # e=B[j0,k0], h=B[j1,k1]
    
    # We need to fill U, V, W for this rank term.
    # U shape (n*m, rank). V shape (m*p, rank). W shape (p*n, rank).
    
    # Helper to set values
    def set_u(i, j, val):
        U[i*m + j][rank_offset] += val
    def set_v(j, k, val):
        V[j*p + k][rank_offset] += val
    def set_w(k, i, val):
        W[k*n + i][rank_offset] += val

    # m1
    set_u(0, 0, 1); set_u(1, 1, 1)
    set_v(0, 0, 1); set_v(1, 1, 1)
    # C11 += m1
    set_w(0, 0, 1)
    # C22 += m1
    set_w(1, 1, 1)
    rank_offset += 1

    # m2 = (c+d)e
    # c=A[i1,j0], d=A[i1,j1]
    # e=B[j0,k0]
    set_u(1, 0, 1); set_u(1, 1, 1)
    set_v(0, 0, 1)
    # C21 += m2
    set_w(0, 1, 1)
    rank_offset += 1

    # m3 = a(f-h)
    # a=A[i0,j0]
    # f=B[j0,k1], h=B[j1,k1]
    set_u(0, 0, 1)
    set_v(0, 1, 1); set_v(1, 1, -1)
    # C12 += m3
    set_w(1, 0, 1)
    # C22 += m3
    set_w(1, 1, 1)
    rank_offset += 1

    # m4 = d(g-e)
    # d=A[i1,j1]
    # g=B[j1,k0], e=B[j0,k0]
    set_u(1, 1, 1)
    set_v(1, 0, 1); set_v(0, 0, -1)
    # C11 += m4
    set_w(0, 0, 1)
    # C21 += m4
    set_w(0, 1, 1)
    rank_offset += 1

    # m5 = (a+b)h
    # a=A[i0,j0], b=A[i0,j1]
    # h=B[j1,k1]
    set_u(0, 0, 1); set_u(0, 1, 1)
    set_v(1, 1, 1)
    # C11 -= m5
    set_w(0, 0, -1)
    # C12 += m5
    set_w(1, 0, 1)
    rank_offset += 1

    # m6 = (c-a)(e+f)
    # c=A[i1,j0], a=A[i0,j0]
    # e=B[j0,k0], f=B[j0,k1]
    set_u(1, 0, 1); set_u(0, 0, -1)
    set_v(0, 0, 1); set_v(0, 1, 1)
    # C22 += m6
    set_w(1, 1, 1)
    rank_offset += 1

    # m7 = (b-d)(g+h)
    # b=A[i0,j1], d=A[i1,j1]
    # g=B[j1,k0], h=B[j1,k1]
    set_u(0, 1, 1); set_u(1, 1, -1)
    set_v(1, 0, 1); set_v(1, 1, 1)
    # C11 += m7
    set_w(0, 0, 1)
    # C22 += m7
    set_w(1, 1, 1)
    rank_offset += 1

    return rank_offset
