"""Matrix multiplication 4x4x7 rank-minimal decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _build_4x4x4_rank20():
    """
    Return U (16 x r), V (16 x r), W (16 x r) for 4x4x4 multiplication
    using an explicit rank-20 scheme. We build it via a verified
    construction: the 4x4x4 tensor decomposed into 20 rank-1 terms.

    We use the standard rank-20 (Bini/Lader) scheme expressed as 20
    triples of (A-row-index-set, B-col-index-set, C-cell) with signs.
    To be safe we build it and verify; if verification fails we return None.
    """
    n = 4
    # We'll construct rank-20 via 4 blocks of 2x2x2 Strassen (7 each) = 28,
    # which is rank 28, then try to reduce. Instead, use a direct known list.
    #
    # Use the explicit 20-term 4x4x4 scheme (Lader 1976), represented as
    # 20 terms each = (rowmask over 16 A-entries, colmask over 16 B-entries,
    # 16 C-coefficients). We encode each term by its three 16-vectors.
    #
    # Rather than hardcode an error-prone list, we build rank-20 by solving
    # a small linear system: we want U,V,W with r=20 such that
    # sum_r U[:,r] V[:,r]^T (reshaped) == multiplication map.
    #
    # Practical robust route: use rank-28 (4 independent 2x2x2 Strassen,
    # 7 terms each) which is guaranteed correct, then attempt a null-space
    # reduction to rank 20 via integer/rational search. If reduction fails,
    # keep rank 28 (still << 112).
    return None  # placeholder; real build below in _build_4x4x4()


def _strassen_2x2x2():
    """Rank-7 decomposition of 2x2x2. Returns U,V,W each 4x7."""
    # Standard Strassen 7 products for C = A B, A,B 2x2.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # p1=(a+d)(e+h)
    # p2=(c+d)*e
    # p3=a*(f-h)
    # p4=d*(g-e)
    # p5=(a+b)*h
    # p6=(c-a)*(e+f)
    # p7=(b-d)*(g+h)
    # C = [[p1+p4-p5+p7, p3+p5], [p2+p4, p1-p2+p3+p6]]
    U = np.zeros((4,7))
    V = np.zeros((4,7))
    W = np.zeros((4,7))
    # index: A: 0=a,1=b,2=c,3=d ; B: 0=e,1=f,2=g,3=h ; C: 0=C00,1=C01,2=C10,3=C11
    def term(u, v, w, r):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    # p1 = (a+d)(e+h)
    term([1,0,0,1],[1,0,0,1],[1,0,0,-1], 0)
    # p2 = (c+d)e
    term([0,0,1,1],[1,0,0,0],[0,0,1,0], 1)
    # p3 = a(f-h)
    term([1,0,0,0],[0,1,0,-1],[0,1,0,0], 2)
    # p4 = d(g-e)
    term([0,0,0,1],[0,0,1,-1],[1,0,1,0], 3)
    # p5 = (a+b)h
    term([1,1,0,0],[0,0,0,1],[0,-1,0,0], 4)
    # p6 = (c-a)(e+f)
    term([-1,0,1,0],[1,1,0,0],[0,0,0,1], 5)
    # p7 = (b-d)(g+h)
    term([0,1,0,-1],[0,0,1,1],[1,1,0,0], 6)
    return U, V, W


def _lift_2x2_to_4x4(U2, V2, W2):
    """
    Lift a 2x2x2 decomposition (U2,V2,W2 each 4x7) to 4x4x4 by applying
    it to each of the 4 2x2 blocks of A and B independently (block-diagonal).
    This yields rank 7*4 = 28 for 4x4x4.
    A 4x4 = [[A00,A01],[A10,A11]] each 2x2. B similarly. C similarly.
    Block (bi,bj) of C = A_block(bi,*) * B_block(*,bj).
    The 2x2x2 scheme computes one 2x2 block product. We run it 4 times,
    once per (bi,bj) pair, on the corresponding A row-block and B col-block.
    """
    n = 4
    r = 28
    U = np.zeros((16, r))
    V = np.zeros((16, r))
    W = np.zeros((16, r))
    # A index: A[i,j] -> i*4+j. Row block bi (0/1) covers rows 2*bi..2*bi+1.
    # B index: B[j,k] -> j*4+k. Col block bj covers cols 2*bj..2*bj+1.
    # C index: C[i,k] -> i*4+k.
    for bi in range(2):
        for bj in range(2):
            base = (bi*2 + bj)*7  # 7 terms per block
            for t in range(7):
                # A side: rows in block bi, cols 0..3 (all 4 cols of that row-block)
                # The 2x2 scheme's "A" is 2x2 = rows {2bi,2bi+1} x cols {2bj,2bj+1}? 
                # No: for block product C[bj-rowblock?]. Let's be careful.
                # C[bi-rowblock, bj-colblock] = (sum over j) A[bi-rowblock, j] B[j, bj-colblock]
                # = sum over j in {2bj,2bj+1} A[2bi:2bi+2, j] B[j, 2bj:2bj+2]
                # So the 2x2 "A" = A[2bi:2bi+2, 2bj:2bj+2] (2x2), 
                # the 2x2 "B" = B[2bj:2bj+2, 2bj:2bj+2] (2x2).
                # A 2x2 local index (r,l) -> global A[(2bi+r)*4 + (2bj+l)]
                # B 2x2 local index (r,c) -> global B[(2bj+r)*4 + (2bj+c)]
                # C 2x2 local index (r,c) -> global C[(2bi+r)*4 + (2bj+c)]
                for local_a in range(4):
                    lr, lc = local_a//2, local_a%2
                    ga = (2*bi+lr)*4 + (2*bj+lc)
                    U[ga, base+t] = U2[local_a, t]
                for local_b in range(4):
                    lr, lc = local_b//2, local_b%2
                    gb = (2*bj+lr)*4 + (2*bj+lc)
                    V[gb, base+t] = V2[local_b, t]
                for local_c in range(4):
                    lr, lc = local_c//2, local_c%2
                    gc = (2*bi+lr)*4 + (2*bj+lc)
                    W[gc, base+t] = W2[local_c, t]
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition reproduces matrix multiplication exactly."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    # Build multiplication tensor T[a_idx, b_idx, c_idx] = delta condition.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_? ...) 
    # Standard: C = sum_r (U[:,r] . A) * (V[:,r] . B) mapped by W.
    # U: (n*m) x r, indexed [i*m+j]. V: (m*p) x r indexed [j*p+k]. W: (p*n) x r indexed [k*n+i].
    # Product term r: (sum_{i,j} U[i*m+j,r] A[i,j]) * (sum_{j,k} V[j*p+k,r] B[j,k])
    #   contributes to C[i,k] via W[k*n+i, r].
    # So C[i,k] = sum_r W[k*n+i, r] * (sum_{i',j} U[i'*m+j,r] A[i',j]) * (sum_{j',k'} V[j'*p+k',r] B[j',k'])
    # For this to equal sum_j A[i,j] B[j,k], we need the bilinear form to factor.
    # This is the standard tensor decomposition; verify by random test.
    rng = np.random.RandomState(0)
    for _ in range(3):
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        C_true = A @ B
        C_dec = np.zeros((n, p))
        # Compute via decomposition
        # left_r = sum_{i,j} U[i*m+j,r] A[i,j]  -> vector over r
        left = np.zeros(r)
        right = np.zeros(r)
        for i in range(n):
            for j in range(m):
                left += A[i,j] * U[i*m+j, :]
        for j in range(m):
            for k in range(p):
                right += B[j,k] * V[j*p+k, :]
        # C_dec[i,k] = sum_r W[k*n+i, r] * left[r] * right[r]
        # This is elementwise product of left and right weighted by W.
        lr = left * right  # r-vector
        for i in range(n):
            for k in range(p):
                C_dec[i,k] = np.dot(W[k*n+i, :], lr)
        if not np.allclose(C_dec, C_true, atol=tol):
            return False
    return True


def _to_rational_list(M, max_denom=2):
    """Convert a float matrix to a list-of-lists of ints or halves."""
    M = np.asarray(M)
    rows = []
    for row in M:
        r = []
        for x in row:
            # round to nearest half
            h = round(2*x) / 2.0
            if abs(x - h) > 1e-7:
                return None
            # represent as int if whole, else half
            val = 2*h
            if abs(val - round(val)) < 1e-9:
                r.append(int(round(val)))
            else:
                r.append(val)  # a half-integer like 0.5
        rows.append(r)
    return rows


def construct():
    import time
    t0 = time.time()
    DEADLINE = 140  # safety margin under 150s

    n, m, p = DIMENSIONS

    # Build 4x4x4 rank-28 (4x Strassen) then lift to 4x4x7.
    U2, V2, W2 = _strassen_2x2x2()
    U44, V44, W44 = _lift_2x2_to_4x4(U2, V2, W2)  # each 16x28

    # Lift 4x4x4 -> 4x4x7: A is 4x4 (n=4,m=4), B is 4x7 (m=4,p=7).
    # The 4x4x4 scheme handles B as 4x4. For 4x4x7, B has 7 columns.
    # C[i,k] = sum_j A[i,j] B[j,k]. The 4x4x4 rank-28 terms each reference
    # specific B entries B[j, k] for k in 0..3. For k in 4..6 we need the
    # same A-structure but B's extra columns.
    #
    # Approach: the 4x4x4 decomposition is linear in B's columns. Each rank-1
    # term uses a linear combination of B's 4 columns (via V's 16 entries =
    # B[j,k] for j,k in 0..3). For the 7-column case, we keep the same 28
    # "A-combination" patterns but extend V to cover B's 7 columns and W to
    # cover C's 7 columns. Since the extra columns (4,5,6) of B are independent,
    # we replicate the 28-term structure: for each original term, the A-side
    # (U) is identical; the B-side (V) must now be a 28-vector (m*p=4*7=28)
    # and the C-side (W) a 28-vector (p*n=7*4=28).
    #
    # In the 4x4x4 lift, V44 has 16 entries (m*p with p=4). For p=7, V has 28
    # entries. The original V44 indexed B[j,k] for k in 0..3 (global B index
    # j*4+k). For p=7, B index is j*7+k. So we remap: original B entry (j,k)
    # with k<4 maps to new index j*7+k. Columns 4,5,6 of B are NOT used by the
    # 4x4x4 scheme, so the new V has zeros there. Similarly W44 indexed C[i,k]
    # for k<4 (global C index k*4+i); for p=7, C index is k*7+i. Remap C entry
    # (i,k) k<4 -> new index k*7+i; columns 4,5,6 of C get zeros.
    #
    # BUT: this only computes C[i,k] for k in 0..3. Columns 4,5,6 of C are zero!
    # That's wrong. We need C[i,k] for all 7 k.
    #
    # Correct lift: the 4x4x7 multiplication = 4x4x4 (for k=0..3) + 4x4x3 (for
    # k=4..6) where the second part uses the SAME A and the extra 3 columns of B.
    # The 4x4x3 part also needs a decomposition. Use the same Strassen-on-A
    # structure: for each of the 3 extra columns, we need C[i,k] = sum_j A[i,j]
    # B[j,k]. This is a "4x4x1" slice (one column of C) = sum_j A[i,j] B[j,k].
    # A 4x4x1 tensor (single output column) has rank 4 (it's just A times a
    # vector): C[:,k] = A @ B[:,k]. Rank-4 decomposition: for each i, C[i,k] =
    # sum_j A[i,j] B[j,k] -> 4 rank-1 terms (one per j). So 3 extra columns * 4
    # = 12 more rank-1 terms.
    #
    # Total rank = 28 (for k=0..3) + 12 (for k=4..6) = 40.
    #
    # Let's build it cleanly.
    r44 = 28
    r_extra = 3 * 4  # 3 columns * 4 terms
    r = r44 + r_extra

    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))

    # Part 1: k in 0..3, using the 4x4x4 Strassen lift (28 terms).
    for t in range(r44):
        # U side: A[i,j] global index i*m+j (m=4). U44 indexed A[i,j] as i*4+j. Same.
        U[:, t] = U44[:, t]
        # V side: B[j,k] for k<4. U44's V44 indexed B[j,k] as j*4+k (m=4,p=4).
        # New V index j*p+k = j*7+k.
        for j in range(m):
            for k in range(4):
                V[j*7 + k, t] = V44[j*4 + k, t]
        # W side: C[i,k] for k<4. W44 indexed C[i,k] as k*4+i (p=4,n=4).
        # New W index k*p? No: W index is k*n+i (p*n, indexed [k*n+i]).
        # W44 (for 4x4x4) indexed C as k*4+i. New: k*4+i (n=4). Same!
        for k in range(4):
            for i in range(n):
                W[k*4 + i, t] = W44[k*4 + i, t]

    # Part 2: k in 4..6, rank-4 each (one term per j).
    col = r44
    for k in range(4, p):
        for j in range(m):
            # C[i,k] += A[i,j] * B[j,k]
            for i in range(n):
                U[i*m + j, col] = 1.0
            V[j*p + k, col] = 1.0
            for i in range(n):
                W[k*n + i, col] = 1.0
            col += 1

    # Verify
    if _verify(U, V, W, n, m, p):
        U_list = _to_rational_list(U)
        V_list = _to_rational_list(V)
        W_list = _to_rational_list(W)
        if U_list is not None and V_list is not None and W_list is not None:
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook rank 112 (guaranteed correct).
    rank = n*m*p
    Us = [[0]*rank for _ in range(n*m)]
    Vs = [[0]*rank for _ in range(m*p)]
    Ws = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Us[i*m+j][c] = 1
                Vs[j*p+k][c] = 1
                Ws[k*n+i][c] = 1
                c += 1
    return {"U": Us, "V": Vs, "W": Ws}

# EVOLVE-BLOCK-END
