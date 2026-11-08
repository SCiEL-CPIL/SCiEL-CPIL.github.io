"""Matrix multiplication 2,4,4 - rank-minimal tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8

    def check_decomposition(U, V, W, tol=1e-6):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r)); V = np.zeros((mp, r)); W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def strassen2x2(U, V, W, base):
        """Add 7 Strassen rank-1 terms for 2x2 block product.
        U rows: base, base+1, base+2, base+3  (X11,X12,X21,X22)
        V rows: base, base+1, base+2, base+3  (Y11,Y12,Y21,Y22)
        W rows: base, base+1, base+2, base+3  (C11,C12,C21,C22)
        """
        # M1=(x11+x22)(y11+y22)
        U[base, base] = 1; U[base+2, base] = 1
        V[base, base] = 1; V[base+2, base] = 1
        W[base, base] = 1; W[base+2, base] = -1
        # M2=(x21+x22)y11
        U[base+2, base+1] = 1; U[base+3, base+1] = 1
        V[base, base+1] = 1
        W[base+2, base+1] = 1
        # M3=x11(y12-y22)
        U[base, base+2] = 1
        V[base+1, base+2] = 1; V[base+3, base+2] = -1
        W[base+1, base+2] = 1
        # M4=x22(y21-y11)
        U[base+3, base+3] = 1
        V[base+2, base+3] = 1; V[base, base+3] = -1
        W[base, base+3] = 1; W[base+2, base+3] = 1
        # M5=(x11+x12)y22
        U[base, base+4] = 1; U[base+1, base+4] = 1
        V[base+3, base+4] = 1
        W[base, base+4] = -1; W[base+1, base+4] = 1
        # M6=(x21-x11)(y11+y12)
        U[base+2, base+5] = 1; U[base, base+5] = -1
        V[base, base+5] = 1; V[base+1, base+5] = 1
        W[base+1, base+5] = 1
        # M7=(x12-x22)(y21+y22)
        U[base+1, base+6] = 1; U[base+3, base+6] = -1
        V[base+2, base+6] = 1; V[base+3, base+6] = 1
        W[base, base+6] = 1; W[base+1, base+6] = -1

    def try_block_strassen():
        """View A (2x4) as two 1x2 row-blocks per i, B (4x4) as 2x2 blocks of 2x2.
        For each i (2), for each B-block-column (2), multiply [A row] (1x2) by B block (2x2).
        A 1x2 x 2x2 product is rank 4 schoolbook, but we use the 2x2 Strassen by padding the
        1x2 into a 2x2 with a zero row, giving 7 terms but the zero row cancels -> rank 4.
        Total = 2 (i) * 2 (block cols) * 4 = 16? Verify.
        """
        # Actually a 1x2 x 2x2 = [a,b]·[[p,q],[r,s]] = [ap+br, aq+bs]: rank 4, no saving.
        # So this gives 2*2*4 = 16 only if we share; but i-rows are independent -> 2*2*4=16 terms.
        # Build it: for each i, for each block-col c in {0,1}:
        #   X = A[i, 2c:2c+2] (1x2), Y = B[0:2, 2c:2c+2] (2x2) -> C[i,2c:2c+2] (1x2)
        #   and second block-row: Y2 = B[2:4, 2c:2c+2] -> but A row is 1x4, so:
        #   C[i, 2c:2c+2] = A[i,2c]*B[0,2c:2c+2] + A[i,2c+1]*B[1,2c:2c+2]
        #                   + A[i,2c+2]*B[2,2c:2c+2] + A[i,2c+3]*B[3,2c:2c+2]
        # This is 4 scalar*vector(1x2) = 8 scalar products per (i,c) -> total 2*2*8=32. No saving.
        return None

    # The honest lever: use the 2x2 Strassen on the (i, block-row) x (block-col) 2x2 block product
    # where each block is a 2x2 matrix, and the A "blocks" are 1x2 row vectors.
    # Product of 2x2 (blocks of 1x2 x 2x2) = 2x2 (blocks of 1x2).
    # Each scalar-Strassen term becomes a (1x2)*(2x2) = 1x2 product = rank 4.
    # 7 terms * 4 = 28. That's < 32!
    def try_strassen_blocks():
        # 2x2 block product: X = [[X00,X01],[X10,X11]] (Xab are 1x2 row vectors of A),
        # Y = [[Y00,Y01],[Y10,Y11]] (Yab are 2x2 blocks of B), C = X*Y (Cab 1x2).
        # X00=A[0,0:2], X01=A[0,2:4], X10=A[1,0:2], X11=A[1,2:4]
        # Y00=B[0:2,0:2], Y01=B[0:2,2:4], Y10=B[2:4,0:2], Y11=B[2:4,2:4]
        # C00 = X00*Y00 + X01*Y10, C01 = X00*Y01 + X01*Y11
        # C10 = X10*Y00 + X11*Y10, C11 = X10*Y01 + X11*Y11
        # Each Xab*Yab is (1x2)*(2x2)=(1x2): 4 scalar mults.
        # Strassen 7 terms, each term = (1x2)*(2x2) = 4 scalar mults -> 28 total.
        r = 28
        U = np.zeros((nm, r)); V = np.zeros((mp, r)); W = np.zeros((pn, r))
        # Map: U row for A[i, j] = i*4+j ; V row for B[j,k] = j*4+k ; W row for C[k,i] = k*2+i
        # For a Strassen term, U-vector = (1x2) row of A, V-vector = (2x2) block of B,
        # W-vector = (1x2) row of C.
        def add_term(u_idx, v_idx, w_idx):
            """u_idx: (i, j0) A row pair; v_idx: (j0,k0) B block; w_idx: (i, k0) C row pair."""
            nonlocal r
            i, ja = u_idx
            jb, kb = v_idx
            ic, kc = w_idx
            # U: A[i, ja], A[i, ja+1]
            U[i * 4 + ja, r] = 1
            U[i * 4 + ja + 1, r] = 1
            # V: B[jb, kb], B[jb, kb+1], B[jb+1, kb], B[jb+1, kb+1]
            V[jb * 4 + kb, r] = 1
            V[jb * 4 + kb + 1, r] = 1
            V[(jb + 1) * 4 + kb, r] = 1
            V[(jb + 1) * 4 + kb + 1, r] = 1
            # W: C[kb, ic], C[kb+1, ic]
            W[kb * 2 + ic, r] = 1
            W[(kb + 1) * 2 + ic, r] = 1
            r += 1

        # Strassen terms for 2x2 block product (X blocks are 1x2, Y blocks 2x2):
        # M1=(X00+X11)*(Y00+Y11)  -> u rows {X00,X11}, v blocks {Y00,Y11}, w row C00? No:
        # C00 = M1 + M4 - M5 + M7 (standard), but here each is a 1x2 vector.
        # We must add/subtract per-term. Build term-by-term with signs.
        # Standard 2x2:
        # M1=(x11+x22)(y11+y22)
        # M2=(x21+x22)y11
        # M3=x11(y12-y22)
        # M4=x22(y21-y11)
        # M5=(x11+x12)y22
        # M6=(x21-x11)(y11+y12)
        # M7=(x12-x22)(y21+y22)
        # C11=M1+M4-M5+M7 ; C12=M3+M5 ; C21=M2+M4 ; C22=M1-M2+M3+M6
        # Here X11=X00=A[0,0:2], X12=X01=A[0,2:4], X21=X10=A[1,0:2], X22=X11=A[1,2:4]
        # Y11=Y00=B[0:2,0:2], Y12=Y01=B[0:2,2:4], Y21=Y10=B[2:4,0:2], Y22=Y11=B[2:4,2:4]
        # C11=C00 (i=0,k=0:2), C12=C01 (i=0,k=2:4), C21=C10 (i=1,k=0:2), C22=C11 (i=1,k=2:4)
        r = 0
        # M1=(X00+X11)*(Y00+Y11): u={A[0,0:2],A[1,2:4]} (both +), v={Y00,Y11}(+), contributes to C00,C11
        def add_signed(u_list, v_list, w_list):
            nonlocal r
            for (i, ja) in u_list:
                for (jb, kb) in v_list:
                    for (ic, kc) in w_list:
                        pass  # need single term; instead build one combined term per (u,v,w) triple
        # Each Strassen term is a single rank-1 in the *block* sense = 4 scalar mults.
        # Represent as ONE column where U has the u-row entries, V the v-block entries, W the w-row entries.
        # That's a rank-1 tensor in the (A-block x B-block x C-block) space, which expands to 4 scalars.
        # So each Strassen term = 1 column! Total 7 columns?? That can't be rank 7 for 28 scalar mults.
        # A single column with U (2 nonzero), V (4 nonzero), W (2 nonzero) gives 2*4*2=16 scalar products,
        # but they are NOT independent — it's a rank-1 (outer product) so only 1 "multiplication" in block sense
        # = 4 scalar mults? No: U⊗V⊗W with |supp U|=2,|supp V|=4,|supp W|=2 gives 16 products, all coupled.
        # This is NOT a valid 4-mult block product. A block product (1x2)*(2x2) is NOT rank-1; it's rank 4.
        # So this whole approach is invalid. Return None.
        return None

    best_U, best_V, best_W = schoolbook()
    best_rank = best_U.shape[1]

    # Attempt a verified improvement via random search over small half-integer values for lower ranks.
    random.seed(12345)
    for r in range(16, best_rank):
        if time.time() > deadline - 5:
            break
        for attempt in range(8):
            if time.time() > deadline - 5:
                break
            U = np.random.choice([-1, 0, 1], size=(nm, r))
            V = np.random.choice([-1, 0, 1], size=(mp, r))
            W = np.random.choice([-1, 0, 1], size=(pn, r))
            if check_decomposition(U, V, W):
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break

    def to_list(arr):
        return np.asarray(arr).tolist()

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
