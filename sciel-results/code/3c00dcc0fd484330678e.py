import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    start_time = time.time()
    deadline = start_time + 130.0

    # Build exact multiplication tensor T[a,b,c]
    # a = i*m+l (A[i,l]), b = l*p+k (B[l,k]), c = k*n+i (C[i,k])
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        S = np.zeros((nm, mp, pn), dtype=float)
        for ri in range(r):
            S += np.einsum('a,b,c->abc', U_mat[:, ri], V_mat[:, ri], W_mat[:, ri])
        return float(np.max(np.abs(S - T)))

    def to_list(mat):
        out = []
        for row in mat:
            rl = []
            for val in row:
                rr = round(val * 4) / 4.0
                rl.append(rr if abs(rr - val) < 1e-9 else val)
            out.append(rl)
        return out

    # ===== SEED: recursive Strassen-like construction =====
    # View 2x4 @ 4x4 as:
    #   A = [A00 A01 A02 A03]   (row i=0)
    #       [A10 A11 A12 A13]   (row i=1)
    #   B is 4x4.
    # We split B into 2x2 blocks of 2x2 and A into 1x2 blocks of 2x2 (two rows).
    # Use a "Strassen 3-level" recursive idea: treat as (2x2)@(2x2) with
    # A being a 2x2 matrix of 1x2 blocks, B a 2x2 matrix of 2x2 blocks.
    #
    # Concretely:
    #   A = [[A_TL, A_TR],
    #        [A_BL, A_BR]]   each block is 1x2
    #   B = [[B_TL, B_TR],
    #        [B_BL, B_BR]]   each block is 2x2
    #   C = [[A_TL*B_TL + A_TR*B_BL,  A_TL*B_TR + A_TR*B_BR],
    #        [A_BL*B_TL + A_BR*B_BL,  A_BL*B_TR + A_BR*B_BR]]  each C block is 1x2
    #
    # Strassen on the outer 2x2 (with non-square inner blocks):
    #   M1 = (A_TL + A_BR) * (B_TL + B_BR)
    #   M2 = (A_BL + A_BR) * B_TL
    #   M3 = A_TL * (B_TR - B_BR)
    #   M4 = A_BR * (B_BL - B_TL)
    #   M5 = (A_TL + A_TR) * B_BR
    #   M6 = (A_BL - A_TL) * (B_TL + B_TR)
    #   M7 = (A_TR - A_BR) * (B_BL + B_BR)
    #   C_TL = M1 - M4 + M6 + M7
    #   C_TR = M3 + M1 + M5
    #   C_BL = M2 + M4
    #   C_BR = M3 - M2 + M1 - M5
    #
    # Each inner product is (1x2)*(2x2) = 1x2, which is a 4-scalar product
    # (rank 4) OR we can use a sub-strassen. Since inner blocks are 1x2 @ 2x2,
    # that's just 4 independent scalar multiplications -> rank 4 each.
    # But M1..M7 each involve a (1x2)*(2x2) product = 4 mults.
    # Total = 7 * 4 = 28. Same as before but structured differently.
    #
    # To get below 28, note that the 7 outer mults share structure. Let's just
    # implement the 28-term version but with the recursive structure, which
    # gives different linear dependencies than 4 independent Strassen-7 blocks.

    rank = 28
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    # Index mapping for A: a = i*m + l  ->  i in {0,1}, l in {0,1,2,3}
    #   A_TL = A[0, 0:2] -> a = 0,1
    #   A_TR = A[0, 2:4] -> a = 2,3
    #   A_BL = A[1, 0:2] -> a = 4,5
    #   A_BR = A[1, 2:4] -> a = 6,7
    # Index mapping for B: b = l*p + k -> l in {0,1,2,3}, k in {0,1,2,3}
    #   B_TL = B[0:2, 0:2] -> b = 0,1,4,5
    #   B_TR = B[0:2, 2:4] -> b = 2,3,6,7
    #   B_BL = B[2:4, 0:2] -> b = 8,9,12,13
    #   B_BR = B[2:4, 2:4] -> b = 10,11,14,15
    # Index mapping for C: c = k*n + i -> k in {0,1,2,3}, i in {0,1}
    #   C_TL = C[0:2, 0:2] -> c = 0,1,2,3
    #   C_TR = C[0:2, 2:4] -> c = 4,5,6,7
    #   C_BL = C[2:4, 0:2] -> c = 8,9,10,11
    #   C_BR = C[2:4, 2:4] -> c = 12,13,14,15

    AT = [0, 1]   # A_TL
    AR = [2, 3]   # A_TR
    BT_ = [4, 5]  # A_BL
    BR_ = [6, 7]  # A_BR
    BTL = [0, 1, 4, 5]
    BTR = [2, 3, 6, 7]
    BBL = [8, 9, 12, 13]
    BBR = [10, 11, 14, 15]
    CTL = [0, 1, 2, 3]
    CTR = [4, 5, 6, 7]
    CBL = [8, 9, 10, 11]
    CBR = [12, 13, 14, 15]

    def add_inner(A_rows, B_rows, C_rows, offset):
        # (1x2) @ (2x2) -> 4 independent scalar products, rank 4
        # A has 2 entries (a0,a1), B has 4 entries (b0,b1,b2,b3), C has 2 entries (c0,c1)
        # C[0] = A[0]*B[0] + A[1]*B[2]
        # C[1] = A[0]*B[1] + A[1]*B[3]
        # Term t0: A[0]*B[0] -> C[0]
        U[A_rows[0], offset + 0] += 1
        V[B_rows[0], offset + 0] += 1
        W[C_rows[0], offset + 0] += 1
        # Term t1: A[0]*B[1] -> C[1]
        U[A_rows[0], offset + 1] += 1
        V[B_rows[1], offset + 1] += 1
        W[C_rows[1], offset + 1] += 1
        # Term t2: A[1]*B[2] -> C[0]
        U[A_rows[1], offset + 2] += 1
        V[B_rows[2], offset + 2] += 1
        W[C_rows[0], offset + 2] += 1
        # Term t3: A[1]*B[3] -> C[1]
        U[A_rows[1], offset + 3] += 1
        V[B_rows[3], offset + 3] += 1
        W[C_rows[1], offset + 3] += 1

    def add_sum(A_terms, B_terms, C_rows, offset, nA, nB):
        # Add (sum of A_terms)*(sum of B_terms) using a rank-1-like expansion
        # A is sum of nA unit vectors (with signs), B is sum of nB unit vectors.
        # Product = sum_{a in A_terms} sum_{b in B_terms} sign_a*sign_b * a*b
        # That's nA*nB terms, but we can use a shared factorization only if
        # the product is a single rank-1 outer. For Strassen we keep nA*nB terms
        # but here nA=2, nB=2 or 4 -> up to 8. Too many. Instead we treat
        # (sum A) as a 1x2 vector with 2 non-zero entries and (sum B) as 2x2
        # with up to 4 non-zero entries, giving nA*nB inner terms.
        # This is what add_inner does when we pass the actual rows.
        pass

    # We'll build M1..M7 where each is (1x2 sum) @ (2x2 sum).
    # M1 = (A_TL + A_BR) @ (B_TL + B_BR): A rows {0,1,6,7} (4 nonzeros), B rows {0,1,4,5,10,11,14,15} (8 nonzeros)
    #   That's 4*8 = 32 terms. Too many.
    #
    # Better: keep the 4-block Strassen seed (rank 28) but ALSO build the
    # recursive seed and pick whichever verifies and is lower rank after
    # dependency reduction. The recursive seed's terms are different linear
    # combinations, so the column-dependency reduction may cut more.
    #
    # For simplicity and correctness, build the recursive seed with explicit
    # expansion of each M_i as inner products, accepting higher raw rank,
    # then reduce via linear dependencies.

    # Rebuild: we'll just keep the 4-block Strassen as the proven seed,
    # and add a SECOND seed: the "single outer Strassen with explicit inner
    # expansion" which has a different dependency structure.

    # --- Seed 1: 4 independent 2x2 Strassen (rank 28) ---
    U1 = np.zeros((nm, 28))
    V1 = np.zeros((mp, 28))
    W1 = np.zeros((pn, 28))

    def add_strassen2x2(X_rows, Y_rows, Z_rows, col_offset):
        # Standard Strassen for 2x2 @ 2x2, 7 terms.
        # X_rows = [x00, x01, x10, x11], Y_rows = [y00, y01, y10, y11], Z_rows = [z00, z01, z10, z11]
        # M1 = (x00+x11)(y00+y11)
        U1[X_rows[0], col_offset+0] += 1
        U1[X_rows[3], col_offset+0] += 1
        V1[Y_rows[0], col_offset+0] += 1
        V1[Y_rows[3], col_offset+0] += 1
        W1[Z_rows[0], col_offset+0] += 1
        W1[Z_rows[3], col_offset+0] += 1
        # M2 = (x10+x11)(y00)
        U1[X_rows[2], col_offset+1] += 1
        U1[X_rows[3], col_offset+1] += 1
        V1[Y_rows[0], col_offset+1] += 1
        W1[Z_rows[2], col_offset+1] += 1
        W1[Z_rows[3], col_offset+1] -= 1
        # M3 = (x00)(y10-y00)
        U1[X_rows[0], col_offset+2] += 1
        V1[Y_rows[1], col_offset+2] += 1
        V1[Y_rows[3], col_offset+2] -= 1
        W1[Z_rows[1], col_offset+2] += 1
        W1[Z_rows[3], col_offset+2] += 1
        # M4 = (x11)(y01-y10)
        U1[X_rows[3], col_offset+3] += 1
        V1[Y_rows[2], col_offset+3] += 1
        V1[Y_rows[0], col_offset+3] -= 1
        W1[Z_rows[0], col_offset+3] += 1
        W1[Z_rows[2], col_offset+3] += 1
        # M5 = (x00+x01)(y11)
        U1[X_rows[0], col_offset+4] += 1
        U1[X_rows[1], col_offset+4] += 1
        V1[Y_rows[3], col_offset+4] += 1
        W1[Z_rows[0], col_offset+4] -= 1
        W1[Z_rows[1], col_offset+4] += 1
        # M6 = (x10-x00)(y00+y01)
        U1[X_rows[2], col_offset+5] += 1
        U1[X_rows[0], col_offset+5] -= 1
        V1[Y_rows[0], col_offset+5] += 1
        V1[Y_rows[1], col_offset+5] += 1
        W1[Z_rows[3], col_offset+5] += 1
        # M7 = (x01-x11)(y10+y11)
        U1[X_rows[1], col_offset+6] += 1
        U1[X_rows[3], col_offset+6] -= 1
        V1[Y_rows[2], col_offset+6] += 1
        V1[Y_rows[3], col_offset+6] += 1
        W1[Z_rows[0], col_offset+6] += 1

    # Block 1: A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
    add_strassen2x2([0,1,4,5], [0,1,4,5], [0,2,1,3], 0)
    # Block 2: A[0:2,2:4] @ B[2:4,0:2] -> C[2:4,0:2]  (wait, need to match C layout)
    # C[2:4, 0:2] means k in {2,3}, i in {0,1} -> c = k*2+i: c=4,5,6,7? No.
    # c = k*n + i = k*2 + i. k=2,i=0->4; k=2,i=1->5; k=3,i=0->6; k=3,i=1->7. So C[2:4,0:2] = c in {4,5,6,7}.
    # But I labeled CBL = [8,9,10,11] earlier. Let me recompute C layout carefully.
    # c = k*n + i, n=2, i in {0,1}, k in {0,1,2,3}
    # k=0: c=0,1   (C row k=0, i=0,1)
    # k=1: c=2,3
    # k=2: c=4,5
    # k=3: c=6,7
    # So C is laid out column-major in k, i.e. c = k*2 + i.
    # C_TL (k=0,1; i=0,1) = c in {0,1,2,3}
    # C_TR (k=2,3; i=0,1) = c in {4,5,6,7}
    # So for 2x4 @ 4x4 with C[i,k], the 2x2 blocks of C (rows=i, cols=k) are:
    #   C_TL = i in {0,1}, k in {0,1} -> c = k*2+i: (k0i0=0,k0i1=1,k1i0=2,k1i1=3) = [0,1,2,3]
    #   C_TR = i in {0,1}, k in {2,3} -> [4,5,6,7]
    #   C_BL = i in {0,1}, k in {0,1} but i=1? No, i only has 2 values {0,1}, already used.
    # Wait, n=2 means only 2 rows. So C is 2x4. Splitting into 2x2 blocks:
    #   C has 2 rows, 4 cols. Two 2x2 blocks side by side: C_left (cols 0,1) and C_right (cols 2,3).
    # C_left = [0,1,2,3], C_right = [4,5,6,7].
    #
    # For A (2x4): A_left (cols 0,1) = a in {0,1,4,5}, A_right (cols 2,3) = a in {2,3,6,7}
    # For B (4x4): split into 2x2 blocks:
    #   B_TL (rows 0,1; cols 0,1) = b in {0,1,4,5}
    #   B_TR (rows 0,1; cols 2,3) = b in {2,3,6,7}
    #   B_BL (rows 2,3; cols 0,1) = b in {8,9,12,13}
    #   B_BR (rows 2,3; cols 2,3) = b in {10,11,14,15}
    #
    # C_left = A_left @ B_TL + A_right @ B_BL
    # C_right = A_left @ B_TR + A_right @ B_BR
    #
    # 4 products of (2x2)@(2x2):
    add_strassen2x2([0,1,4,5], [0,1,4,5], [0,2,1,3], 0)    # A_left @ B_TL -> C_left
    add_strassen2x2([2,3,6,7], [8,9,12,13], [0,2,1,3], 7)  # A_right @ B_BL -> C_left
    add_strassen2x2([0,1,4,5], [2,3,6,7], [4,6,5,7], 14)   # A_left @ B_TR -> C_right
    add_strassen2x2([2,3,6,7], [10,11,14,15], [4,6,5,7], 21)  # A_right @ B_BR -> C_right

    err1 = verify(U1, V1, W1)

    # --- Seed 2: single outer Strassen, inner products explicit (rank 7*4=28 but diff structure) ---
    # Represent A as 2x2 blocks of 1x2, B as 2x2 blocks of 2x2.
    # A_TL=[0,1], A_TR=[2,3], A_BL=[4,5], A_BR=[6,7]
    # B_TL=[0,1,4,5], B_TR=[2,3,6,7], B_BL=[8,9,12,13], B_BR=[10,11,14,15]
    # C_TL=[0,1,2,3], C_TR=[4,5,6,7] (C is 2x4, only left/right)
    # Outer Strassen:
    # C_TL = M1 - M4 + M6 + M7  where these are (1x2)@(2x2) -> 1x2 results, but C_TL is 2x2=4 entries.
    # This doesn't match dimensions cleanly. The outer Strassen works when inner
    # products produce the right-sized blocks. Here inner (1x2)@(2x2) = 1x2, but
    # we need 2x2 for C blocks. So outer Strassen is not directly applicable.
    # Skip seed 2; keep seed 1.

    best_U = U1.copy()
    best_V = V1.copy()
    best_W = W1.copy()
    best_rank = 28

    if err1 > 1e-9:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

    # ===== Refinement: linear-dependency column removal =====
    def build_term_matrix(Uc, Vc, Wc):
        r = Uc.shape[1]
        M = np.zeros((nm * mp * pn, r), dtype=float)
        for ri in range(r):
            M[:, ri] = np.einsum('a,b,c->abc', Uc[:, ri], Vc[:, ri], Wc[:, ri]).flatten()
        return M

    def try_remove(Uc, Vc, Wc, cols):
        remaining = [c for c in range(Uc.shape[1]) if c not in cols]
        if not remaining:
            return None
        Un, Vn, Wn = Uc[:, remaining], Vc[:, remaining], Wc[:, remaining]
        if verify(Un, Vn, Wn) < 1e-9:
            return (Un, Vn, Wn)
        return None

    curU, curV, curW = U1.copy(), V1.copy(), W1.copy()
    cur_rank = 28

    for it in range(40):
        if time.time() > deadline - 30:
            break
        if cur_rank <= best_rank:
            break
        M = build_term_matrix(curU, curV, curW)
        removed = False
        for r_el in range(cur_rank):
            if time.time() > deadline - 25:
                break
            other = [s for s in range(cur_rank) if s != r_el]
            Ms = M[:, other]
            rhs = M[:, r_el]
            try:
                alpha, _, _, _ = np.linalg.lstsq(Ms, rhs, rcond=1e-10)
                resid = float(np.max(np.abs(Ms @ alpha - rhs)))
                if resid < 1e-7:
                    res = try_remove(curU, curV, curW, [r_el])
                    if res is not None:
                        curU, curV, curW = res
                        cur_rank -= 1
                        removed = True
                        if cur_rank < best_rank:
                            best_rank = cur_rank
                            best_U, best_V, best_W = curU.copy(), curV.copy(), curW.copy()
                        break
            except np.linalg.LinAlgError:
                continue
        if not removed:
            break

    # ===== Randomised local search: perturb columns, verify, reduce =====
    # Try adding random low-rank "correction" terms and then reduce, hoping to
    # find a representation that reduces below the current best.
    if time.time() < deadline - 40:
        rng = np.random.RandomState(42)
        baseU, baseV, baseW = U1.copy(), V1.copy(), W1.copy()
        base_rank = 28
        for attempt in range(60):
            if time.time() > deadline - 40:
                break
            # Add k random rank-1 terms with small random entries, then reduce.
            k = 1
            extra_U = rng.uniform(-1, 1, (nm, k)) * 0.5
            extra_V = rng.uniform(-1, 1, (mp, k)) * 0.5
            extra_W = rng.uniform(-1, 1, (pn, k)) * 0.5
            candU = np.hstack([baseU, extra_U])
            candV = np.hstack([baseV, extra_V])
            candW = np.hstack([baseW, extra_W])
            # Round to half-integers to keep exactness potential
            candU = np.round(candU * 2) / 2.0
            candV = np.round(candV * 2) / 2.0
            candW = np.round(candW * 2) / 2.0
            if verify(candU, candV, candW) > 1e-9:
                continue
            cand_rank = candU.shape[1]
            # Reduce
            for _ in range(cand_rank + 5):
                if time.time() > deadline - 35:
                    break
                M = build_term_matrix(candU, candV, candW)
                r = candU.shape[1]
                removed = False
                for r_el in range(r):
                    other = [s for s in range(r) if s != r_el]
                    Ms = M[:, other]
                    rhs = M[:, r_el]
                    try:
                        alpha, _, _, _ = np.linalg.lstsq(Ms, rhs, rcond=1e-10)
                        resid = float(np.max(np.abs(Ms @ alpha - rhs)))
                        if resid < 1e-7:
                            res = try_remove(candU, candV, candW, [r_el])
                            if res is not None:
                                candU, candV, candW = res
                                removed = True
                                break
                    except np.linalg.LinAlgError:
                        continue
                if not removed:
                    break
            if candU.shape[1] < best_rank and verify(candU, candV, candW) < 1e-9:
                best_rank = candU.shape[1]
                best_U, best_V, best_W = candU.copy(), candV.copy(), candW.copy()

    # Final verification
    if verify(best_U, best_V, best_W) > 1e-9:
        best_U, best_V, best_W, best_rank = U1, V1, W1, 28

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
