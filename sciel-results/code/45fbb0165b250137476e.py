import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0  # safety margin before the 150s kill

    # Build a rank-46 decomposition:
    #   A is 2x5, B is 5x5, C is 2x5.
    #   Split A into A0 (cols 0..3) 2x4 and A1 (col 4) 2x1.
    #   Split B rows 0..3 into B0 4x5 (cols 0..3) and B1 4x1 (col 4).
    #   C = A0*B0 + A1*B1.
    #   A0*B0 is (2x4)(4x4): partition into four (2x2)(2x2) sub-products,
    #     each computed with a rank-7 Strassen decomposition -> 4*7 = 28 terms.
    #   A1*B1 is (2x1)(1x5): 10 trivial rank-1 terms.
    #   A0*B1 is (2x4)(4x1): 8 trivial rank-1 terms.
    #   Total = 28 + 10 + 8 = 46.
    #
    # Index conventions (official trace convention):
    #   U row index = i*m + j  (i in 0..1, j in 0..4)
    #   V row index = j*p + k  (j in 0..4, k in 0..4)
    #   W row index = k*n + i  (k in 0..4, i in 0..1)
    # A rank-1 term (u, v, w) contributes u*A[i,j]*B[j,k] to C[i,k].

    U = []
    V = []
    W = []

    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)

    def add_trivial(i, j, k, scale=1):
        # term A[i,j]*B[j,k] -> C[i,k]
        add_term(i * m + j, j * p + k, k * n + i)

    # Strassen rank-7 pattern for X (2x2) * Y (2x2) -> Z (2x2).
    # X local row x (0,1), local col y (0,1); Y local row y, local col z.
    # M1 = X00*(Y00+Y11)   -> Z00 += M1, Z11 += M1
    # M2 = (X00+X10)*Y11   -> Z10 += M2, Z11 -= M2
    # M3 = (X01+X11)*Y00   -> Z01 += M3, Z11 += M3
    # M4 = X11*(Y10-Y00)   -> Z00 += M4, Z10 += M4
    # M5 = X00*(Y01-Y11)   -> Z00 -= M5, Z01 += M5
    # M6 = (X00+X01)*Y11   -> Z11 += M6
    # M7 = (X10+X11)*(Y00-Y01) -> Z00 += M7
    def strassen_2x2(i0, i1, ja, jb, ka, kb):
        # X[x][y] = A[i0+x][ja+y], Y[y][z] = B[jb+y][kb+z], Z[x][z] -> C[i0+x][kb+z]
        pairs = [
            # (coeff on X00, coeff on X10, coeff on X01, coeff on X11,
            #  coeff on Y00, coeff on Y10, coeff on Y00b.. ) -- use explicit lists instead.
        ]
        # Define each M as (list of (i_local, y, sign_x), list of (y, z, sign_y))
        M = [
            # M1
            ([(0, 0, +1)], [(0, 0, +1), (1, 1, +1)]),
            # M2
            ([(0, 0, +1), (1, 0, +1)], [(1, 1, +1)]),
            # M3
            ([(0, 1, +1), (1, 1, +1)], [(0, 0, +1)]),
            # M4
            ([(1, 1, +1)], [(1, 0, +1), (0, 0, -1)]),
            # M5
            ([(0, 0, +1)], [(0, 1, +1), (1, 1, -1)]),
            # M6
            ([(0, 0, +1), (0, 1, +1)], [(1, 1, +1)]),
            # M7
            ([(1, 0, +1), (1, 1, +1)], [(0, 0, +1), (0, 1, -1)]),
        ]
        # Contribution of M to Z local (x, z) with sign.
        contrib = [
            # (x, z, sign) per M
            [(0, 0, +1), (1, 1, +1)],   # M1
            [(1, 0, +1), (1, 1, -1)],   # M2
            [(0, 1, +1), (1, 1, +1)],   # M3
            [(0, 0, +1), (1, 0, +1)],   # M4
            [(0, 0, -1), (0, 1, +1)],   # M5
            [(1, 1, +1)],               # M6
            [(0, 0, +1)],               # M7
        ]

        for mi in range(7):
            xs = M[mi][0]
            ys = M[mi][1]
            u = {}
            v = {}
            for (xl, yl, s) in xs:
                i = i0 + xl
                j = ja + yl
                u[i] = u.get(i, 0) + s
            for (yl, zl, s) in ys:
                j = jb + yl
                k = kb + zl
                v[j] = v.get(j, 0) + s
            # Expand into rank-1 terms: for each u-row i and v-row j.
            for i, us in u.items():
                for j, vs in v.items():
                    # w: for each k with the same sign pattern
                    # Actually w depends on k only through C[i,k]; each (i,j) pairs
                    # with each (j,k) giving w = k*n + i.
                    # But v already aggregates over k via its own k. We need to be careful:
                    # The M term is a bilinear form sum_{i in X} a_i A[i][*] * sum_{k in Y} b_k B[*][k].
                    # It equals sum_{i,k} a_i b_k A[i][j_i] B[j_k][k] for the fixed middle index set.
                    # Each (i, j_mid from X side) and (j_mid from Y side, k) must agree on middle j.
                    # So we must pair by middle index j.
                    pass
            # Redo properly: build per-middle-j coefficient.
            add_strassen_M(i0, i1, ja, jb, ka, kb, xs, ys, contrib[mi])

    # Simpler explicit builder for one Strassen M.
    def add_strassen_M(i0, i1, ja, jb, ka, kb, xs, ys, contrib):
        # xs: list of (x_local, y_local, sign) for X = A[i0+x][ja+y]
        # ys: list of (y_local, z_local, sign) for Y = B[jb+y][kb+z]
        # contrib: list of (x_local, z_local, sign) giving Z = C[i0+x][kb+z]
        # The M term = (sum_xl s * A[i0+xl][ja+yl]) * (sum_yl t * B[jb+yl][kb+zl])
        # = sum over xl,yl,zl of s*t * A[i0+xl][ja+yl] * B[jb+yl][kb+zl]
        # For this to be a single rank-1 term, the middle index must be unique:
        # ja+yl (from X) must equal jb+yl (from Y). They use the same yl, so:
        #   ja+yl == jb+yl  => ja == jb. So this only works if ja == jb (same middle block).
        # In our partition, X uses columns ja+yl and Y uses rows jb+yl with the same yl,
        # so we need ja == jb. That's exactly the case: X is A[:, block] and Y is B[block, :].
        # So each M is a product of a linear combo of A rows and a linear combo of B rows
        # that all share the SAME middle index? No: the middle index varies with yl.
        # So M = sum_yl ( (sum_xl s_xl A[i0+xl][ja+yl]) * (sum_zl t_zl B[ja+yl][kb+zl]) ).
        # This is a sum over yl of products, each product has a FIXED middle index ja+yl.
        # So M decomposes into (number of distinct yl) rank-1 terms, one per yl,
        # where for a fixed yl: u-row = (sum_xl s_xl A[i0+xl]) at column j=ja+yl,
        # v-row = (sum_zl t_zl B[.][kb+zl]) at row j=ja+yl.
        # But a rank-1 term uses a SINGLE (u, v, w) with u being a full vector.
        # u has a single nonzero at (i0+xl, ja+yl)? No, the sum over xl means u has multiple rows.
        # A rank-1 term U_c * A * V_c with U_c a vector: U_c[i0+xl] = s_xl for the xl that appear,
        # but only at column ja+yl. Since U_c is a single vector over (i,j), and it can have
        # entries at multiple (i0+xl, ja+yl) for the SAME j=ja+yl, that's fine: U_c[i0+xl] at column
        # ja+yl = s_xl. Similarly V_c[ja+yl][kb+zl] = t_zl. Then U_c*A*V_c =
        #   sum_{xl, zl} s_xl t_zl A[i0+xl][ja+yl] B[ja+yl][kb+zl]. 
        #   (The j index is fixed at ja+yl for both, and the sums are over xl and zl independently.)
        # Yes! Because U_c has nonzero only at column ja+yl (for the i-rows i0+xl), and V_c has
        # nonzero only at row ja+yl (for the k-columns kb+zl). So U_c*A picks out A[.][ja+yl]
        # weighted by s_xl, and (A*V_c) is not needed; the bilinear form is
        #   (U_c^T A) * (B V_c)?? Let's check: (U_c^T A)[i] no... 
        # The rank-1 term in the tensor is: term = sum_{i,j,k} U_c[i,j] A[i,j] V_c[j,k] W_c[k,i]... 
        # Actually the decomposition is C = sum_c (U_c^T (A (elementwise... )) ). Let me use the
        # standard: C[i,k] = sum_c U_c[i,j_c?]... The standard rank-r decomposition of the
        # matmul tensor is C[i,k] = sum_c U_c[i,j] ... no.
        # The correct bilinear form: C = sum_c (u_c^T * A) * (B * v_c) is not right dimensionally.
        # Let me use: C[i,k] = sum_c ( sum_j U_c[i,j] A[i,j] ) * ( sum_j V_c[j,k] B[j,k] )?? 
        # No, that double-counts j. The standard form is:
        #   C[i,k] = sum_c U_c[i, j] ... 
        # The cleanest: a rank-1 term corresponds to choosing a vector u over (i,j),
        # a vector v over (j,k), and the term contributes (u^T (A ⊙ something)). 
        # The official form: C[i,k] = sum_c sum_{i,j,k} T_c[i,j,k] A[i,j] B[j,k], where
        # T_c is rank-1: T_c[i,j,k] = U_c[i,j] * V_c[j,k] * ... no, T_c[i,j,k] = u_c[i,j]*v_c[j,k]
        # is already bilinear in A and B if we write T_c[i,j,k] A[i,j] B[j,k] = u_c[i,j] v_c[j,k] A[i,j] B[j,k].
        # So C[i,k] = sum_c sum_{j} (sum_i? ) ... 
        # For fixed c, the contribution to C[i,k] is sum_j u_c[i,j] v_c[j,k] A[i,j] B[j,k].
        # So the effective (i,k) matrix is (A ⊙ u_c) (B ⊙ v_c) where ⊙ is elementwise:
        #   (A⊙u_c)[i,j] = u_c[i,j] A[i,j], (B⊙v_c)[j,k] = v_c[j,k] B[j,k],
        #   and C_c[i,k] = sum_j (A⊙u_c)[i,j] (B⊙v_c)[j,k].
        # So u_c and v_c can each have multiple nonzeros, and the product is a full matmul of
        # the masked matrices. Good, so my M with multiple xl and zl is a single rank-1 term as
        # long as u_c[i0+xl, ja+yl] = s_xl and v_c[ja+yl, kb+zl] = t_zl for a FIXED yl.
        # Because then (A⊙u_c) is nonzero only in column ja+yl, and (B⊙v_c) nonzero only in row
        # ja+yl, so (A⊙u_c)(B⊙v_c)[i0+x, kb+z] = sum_j (A⊙u_c)[i0+x, j](B⊙v_c)[j, kb+z]
        # = (A⊙u_c)[i0+x, ja+yl](B⊙v_c)[ja+yl, kb+z]  (only j=ja+yl survives)
        # = (s_x A[i0+x][ja+yl]) * (t_z B[ja+yl][kb+z]).
        # Summed over the rank-1 decomposition this gives exactly the M term summed over xl,zl.
        # 
        # So: for a fixed yl (middle index), build u and v.
        # But different yl give DIFFERENT rank-1 terms (different middle index). So M splits into
        # as many rank-1 terms as there are distinct yl in xs (which must equal the distinct yl in ys).
        # For our Strassen M's, xs and ys each involve specific yl values. Let's just handle per yl.

        # Collect, for each middle index j (which equals ja+yl), the u-coeff per i and v-coeff per k.
        # u part: for each (xl, yl, s): i = i0+xl, j = ja+yl, u[(i, j)] += s
        # v part: for each (yl, zl, t): j = jb+yl, k = kb+zl, v[(j, k)] += t
        # For the term to be rank-1 per middle j, we need that for each j, all u entries share that j
        # and all v entries share that j, and then the term is sum over (i at that j) and (k at that j).
        # So group by j.
        from collections import defaultdict
        u_by_j = defaultdict(lambda: defaultdict(int))  # j -> {i: coeff}
        v_by_j = defaultdict(lambda: defaultdict(int))  # j -> {k: coeff}
        for (xl, yl, s) in xs:
            i = i0 + xl
            j = ja + yl
            u_by_j[j][i] = u_by_j[j].get(i, 0) + s
        for (yl, zl, t) in ys:
            j = jb + yl
            k = kb + zl
            v_by_j[j][k] = v_by_j[j].get(k, 0) + t
        # For each j that appears in BOTH u and v, create a rank-1 term.
        for j in set(u_by_j.keys()) & set(v_by_j.keys()):
            u = {}
            v = {}
            w = {}
            for i, c in u_by_j[j].items():
                if c:
                    u[i * m + j] = c
            for k, c in v_by_j[j].items():
                if c:
                    v[j * p + k] = c
            # w: C[i,k] for i in u's i, k in v's k. w row = k*n + i.
            for i in u:
                for k in v:
                    w[k * n + i] = 1
            add_term(u, v, w)

    # Now apply Strassen to the four 2x2x2 sub-products of A0*B0.
    # A0 = A[:, 0:4], B0 = B[0:4, 0:4].
    # Sub-product (a, b): X = A[:, 2a:2a+2] (2x2), Y = B[2b:2b+2, 2b:2b+2] (2x2),
    #   result -> C[:, 2b:2b+2].
    #   i0=0, i1=1, ja=2a, jb=2b, ka=2b, kb=2b.
    for a in range(2):
        for b in range(2):
            if time.time() > deadline:
                break
            strassen_2x2(0, 1, 2 * a, 2 * b, 2 * b, 2 * b)

    # A0 * B1: A0 (2x4) * B1 (4x1) -> C[:, 4] (2x1). Trivial: 8 terms.
    for i in range(2):
        for j in range(4):
            if time.time() > deadline:
                break
            add_trivial(i, j, 4)  # A[i,j]*B[j,4] -> C[i,4]

    # A1 * B1: A1 (2x1) * B1 (1x5) -> C (2x5). Trivial: 10 terms.
    for i in range(2):
        for k in range(5):
            if time.time() > deadline:
                break
            add_trivial(i, 4, k)  # A[i,4]*B[4,k] -> C[i,k]

    # Convert lists of dicts to dense matrix rows.
    rank = len(U)
    nU = n * m
    nV = m * p
    nW = p * n
    Ud = [[0] * rank for _ in range(nU)]
    Vd = [[0] * rank for _ in range(nV)]
    Wd = [[0] * rank for _ in range(nW)]
    for c in range(rank):
        for row, val in U[c].items():
            Ud[row][c] = val
        for row, val in V[c].items():
            Vd[row][c] = val
        for row, val in W[c].items():
            Wd[row][c] = val

    return {"U": Ud, "V": Vd, "W": Wd}
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    import numpy as np
    res = construct()
    U, V, W = np.array(res["U"]), np.array(res["V"]), np.array(res["W"])
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Verify: C[i,k] = sum_c sum_j U[i*m+j, c] * A[i,j] * V[j*p+k, c] * ... 
    # Actually verify via tensor: T[i,j,k] should equal sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c].
    T = np.zeros((n, m, p))
    for c in range(rank):
        T += np.einsum("i,j,k->ijk",
                       U.reshape(n, m),  # placeholder
                       np.ones(1), np.ones(1))
    # Proper check using random A, B.
    rng = np.random.RandomState(0)
    A = rng.randint(-3, 4, size=(n, m))
    B = rng.randint(-3, 4, size=(m, p))
    C = A @ B
    # Reconstruct C from decomposition:
    Crec = np.zeros((n, p))
    for c in range(rank):
        # (A⊙u_c)(B⊙v_c)
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        Ac = (A * uc)
        Bc = (B * vc)
        Crec += Ac @ Bc
    # wc should be all-ones on the (i,k) that are covered; ensure wc == 1 where used.
    ok = np.allclose(Crec, C)
    print("rank:", rank, "verify:", ok)
    if not ok:
        print("max err:", np.max(np.abs(Crec - C)))
