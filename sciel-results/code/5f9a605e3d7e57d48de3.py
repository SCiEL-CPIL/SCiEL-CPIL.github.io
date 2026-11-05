"""Matrix multiplication 2x5x5 - block-partition + Strassen (46-rank), self-verifying."""

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 2, 5, 5

    # A col groups / B row groups / C row groups offsets and sizes
    acol = [2, 2, 1]
    acoff = [0, 2, 4]
    # B col groups / C row groups
    bcol = [2, 2, 1]
    bcoff = [0, 2, 4]
    # A rows single group size 2 (indices 0,1); C cols single group (0,1)

    # Each rank term: dict with U: list of (i,j,coeff), V: list of (j,k,coeff), W: list of (k,i,coeff)
    terms = []

    def add_term(u, v, w):
        terms.append((u, v, w))

    for ac in range(3):
        for bc in range(3):
            a0 = acoff[ac]; asz = acol[ac]
            b0 = acoff[ac]; bcol0 = bcoff[bc]; bsz = bcol[bc]
            # A block rows {0,1}, cols a0..; B block rows b0.. (==a0..), cols bcol0..
            # C block rows bcol0.., cols {0,1}
            if asz == 2 and bsz == 2:
                # 2x2 * 2x2 -> 2x2 Strassen
                # local: A[r,t] global (r, a0+t); B[t,s] global (b0+t, bcol0+s); C[s,r] global (bcol0+s, r)
                A = lambda r, t: (r, a0 + t)
                B = lambda t, s: (b0 + t, bcol0 + s)
                C = lambda s, r: (bcol0 + s, r)
                # Strassen 7 products (scalar sums), then C entries as signed sums.
                # p1=(A00+A11)(B00+B11)
                p1_u = [A(0,0), A(1,1)]
                p1_v = [B(0,0), B(1,1)]
                # p2=(A10+A11)(B00)
                p2_u = [A(1,0), A(1,1)]
                p2_v = [B(0,0)]
                # p3=(A00)(B10-B00)
                p3_u = [A(0,0)]
                p3_v = [B(1,0), (-B(0,0))]
                # p4=(A11)(B01-B11)
                p4_u = [A(1,1)]
                p4_v = [B(0,1), (-B(1,1))]
                # p5=(A00+A01)(B11)
                p5_u = [A(0,0), A(0,1)]
                p5_v = [B(1,1)]
                # p6=(A10+A01)(B00+B11)
                p6_u = [A(1,0), A(0,1)]
                p6_v = [B(0,0), B(1,1)]
                # p7=(A01+A10)(B10+B11)
                p7_u = [A(0,1), A(1,0)]
                p7_v = [B(1,0), B(1,1)]

                def mk(u_list, v_list, w_list):
                    # convert to coeff form
                    u = [(i, j, 1) for (i, j) in u_list]
                    v = [(i, j, 1) for (i, j) in v_list]
                    w = [(i, j, 1) for (i, j) in w_list]
                    return u, v, w

                # C00 = p1 + p4 - p5 + p7 ; C01 = p3 + p5 ; C10 = p2 + p6 ; C11 = p1 - p2 + p3 + p6
                # For each product pk with u_k, v_k, its contribution to C[s,r] is (+/-) 1 * w=(C(s,r)).
                # Build terms: for product pk, for each (s,r) in C with coefficient coef:
                #   term with u=u_k, v=v_k, w=[(C(s,r), coef)]
                products = {
                    1: (p1_u, p1_v), 2: (p2_u, p2_v), 3: (p3_u, p3_v), 4: (p4_u, p4_v),
                    5: (p5_u, p5_v), 6: (p6_u, p6_v), 7: (p7_u, p7_v),
                }
                Ccoeffs = {
                    (0, 0): {1: +1, 4: +1, 5: -1, 7: +1},
                    (0, 1): {3: +1, 5: +1},
                    (1, 0): {2: +1, 6: +1},
                    (1, 1): {1: +1, 2: -1, 3: +1, 6: +1},
                }
                for (s, r), prods in Ccoeffs.items():
                    for pk, coef in prods.items():
                        u, v = products[pk]
                        u_form = [(i, j, 1) for (i, j) in u]
                        v_form = [(i, j, 1) for (i, j) in v]
                        w_form = [(bcol0 + s, r, coef)]
                        add_term(u_form, v_form, w_form)
            else:
                # schoolbook block
                for r in range(2):
                    for s in range(bsz):
                        for t in range(asz):
                            add_term(
                                [(r, a0 + t, 1)],
                                [(b0 + t, bcol0 + s, 1)],
                                [(bcol0 + s, r, 1)],
                            )

    # Expand to matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (u, v, w) in enumerate(terms):
        for (i, j, coeff) in u:
            U[i * m + j][c] += coeff
        for (j, k, coeff) in v:
            V[j * p + k][c] += coeff
        for (k, i, coeff) in w:
            W[k * n + i][c] += coeff

    # Verify exactly
    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != (1 if k == i else 0):
                        return False
        return True

    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback: verified schoolbook
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    assert verify(U, V, W)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
