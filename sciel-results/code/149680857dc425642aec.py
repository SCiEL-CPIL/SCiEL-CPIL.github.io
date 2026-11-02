"""Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 138.0  # safety margin below the 150s kill


# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """T[io, u, v] = 1 iff output (i,k) uses a[i,j]*b[j,k]; io=k*n+i, u=i*m+j, v=j*p+k."""
    T = np.zeros((p * n, n * m, m * p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1
    return T


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _bini_2x2x2():
    """
    Correct rank-7 decomposition for 2x2 * 2x2 -> 2x2.
    Convention: u[i*m+j], v[j*p+k], w[k*n+i] with n=m=p=2.
    u: 0=A00 1=A01 2=A10 3=A11
    v: 0=B00 1=B01 2=B10 3=B11
    w: 0=C00 1=C10 2=C01 3=C11
    Each returned triple is a list of (u_vec, v_vec, w_vec) length-4 lists.
    """
    terms = [
        ([1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]),  # A00*B00 -> C00
        ([0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]),  # A01*B10 -> C00
        ([0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]),  # A10*B00 -> C10
        ([0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]),  # A11*B10 -> C10
        ([1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]),  # A00*B01 -> C01
        ([0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]),  # A01*B11 -> C01
        ([0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1]),  # A11*B00 -> C11
    ]
    return terms


def _bini_2x2x1():
    """
    Rank-4 (schoolbook, minimal) decomposition for 2x2 * 2x1 -> 2x1.
    A is 2x2 (u: A00,A01,A10,A11), B is 2x1 (v: B0,B1), C is 2x1 (w: C0,C1).
    C0 = A00*B0 + A01*B1 ; C1 = A10*B0 + A11*B1.
    """
    terms = [
        ([1, 0, 0, 0], [1, 0], [1, 0]),  # A00*B0 -> C0
        ([0, 1, 0, 0], [0, 1], [1, 0]),  # A01*B1 -> C0
        ([0, 0, 1, 0], [1, 0], [0, 1]),  # A10*B0 -> C1
        ([0, 0, 0, 1], [0, 1], [0, 1]),  # A11*B1 -> C1
    ]
    return terms


def _construct_blockwise(n, m, p):
    """
    Partition the 4x4 * 4x9 product into 2x2 micro-blocks.
    For each (bi, bk) block of the output (2x2):
      - if bk < 4: full 2x2*2x2 -> 2x2 (rank 7), summed over bj=0,1
      - if bk == 4: 2x2*2x1 -> 2x1 (rank 4), summed over bj=0,1
    Total rank = 8*7 + 4*4 = 56 + 16 = 72.
    """
    U = []
    V = []
    W = []

    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    bini22 = _bini_2x2x2()
    bini21 = _bini_2x2x1()

    for bi in range(2):          # output row block
        for bk in range(5):      # output col block (5 blocks of width 2 cover 9? no: 4 full + 1 partial)
            # Determine which B columns this output block uses.
            # Output col block bk covers global output cols 2*bk .. 2*bk+1.
            # A full 2x2 block needs B cols 2*bk..2*bk+1 (both exist if bk<4).
            # The last block bk=4 covers output cols 8,9 but B only has col 8 (p=9).
            full = (2 * bk + 1) < p  # both output cols map to existing B cols
            for bj in range(2):      # sum over A-col / B-row block
                if full:
                    for u, v, w in bini22:
                        u_full = [0] * (n * m)
                        v_full = [0] * (m * p)
                        w_full = [0] * (p * n)
                        for idx in range(4):
                            i2 = idx // 2
                            j2 = idx % 2
                            if u[idx] != 0:
                                u_full[(2 * bi + i2) * m + (2 * bj + j2)] = u[idx]
                            if v[idx] != 0:
                                v_full[(2 * bj + i2) * p + (2 * bk + j2)] = v[idx]
                            if w[idx] != 0:
                                w_full[(2 * bk + i2) * n + (2 * bi + j2)] = w[idx]
                        add_term(u_full, v_full, w_full)
                else:
                    # Partial: only output col 2*bk (global col 8) is valid.
                    # 2x2 * 2x1 -> 2x1: A rows (2bi,2bi+1) x B col 8.
                    for u, v, w in bini21:
                        u_full = [0] * (n * m)
                        v_full = [0] * (m * p)
                        w_full = [0] * (p * n)
                        for idx in range(4):
                            i2 = idx // 2
                            j2 = idx % 2
                            if u[idx] != 0:
                                u_full[(2 * bi + i2) * m + (2 * bj + j2)] = u[idx]
                        for idx in range(2):
                            i2 = idx  # B row = 2*bj + i2, B col = 8
                            if v[idx] != 0:
                                v_full[(2 * bj + i2) * p + (2 * bk)] = v[idx]
                        for idx in range(2):
                            i2 = idx  # C row = 2*bi + i2, C col = 8
                            if w[idx] != 0:
                                w_full[(2 * bk) * n + (2 * bi + i2)] = w[idx]
                        add_term(u_full, v_full, w_full)

    rank = len(U)
    U_out = [[0] * rank for _ in range(n * m)]
    V_out = [[0] * rank for _ in range(m * p)]
    W_out = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        for u in range(n * m):
            U_out[u][c] = U[c][u]
        for v in range(m * p):
            V_out[v][c] = V[c][v]
        for w in range(p * n):
            W_out[w][c] = W[c][w]
    return {"U": U_out, "V": V_out, "W": W_out}


def _to_arrays(res, n, m, p):
    U = np.array(res["U"], dtype=object)
    V = np.array(res["V"], dtype=object)
    W = np.array(res["W"], dtype=object)
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Exact verification. Returns True if the decomposition is correct."""
    rank = U.shape[1]
    # Precompute nonzero structure: for each c, the (u,v) pairs where U[u,c]*V[v,c]!=0
    for io in range(p * n):
        # W[io, c]
        w_row = W[io]
        acc = np.zeros((n * m, m * p), dtype=object)
        for c in range(rank):
            wc = w_row[c]
            if wc == 0:
                continue
            uc = U[:, c]
            vc = V[:, c]
            # outer product uc * vc * wc
            for u in range(n * m):
                uc_val = uc[u]
                if uc_val == 0:
                    continue
                uwc = uc_val * wc
                for v in range(m * p):
                    vc_val = vc[v]
                    if vc_val == 0:
                        continue
                    acc[u, v] = (acc[u, v] or 0) + uwc * vc_val
        # Compare with T
        for i in range(n):
            for j in range(m):
                u = i * m + j
                for k in range(p):
                    v = j * p + k
                    expected = 1
                    got = acc[u, v]
                    if got is None:
                        got = 0
                    if got != expected:
                        return False
    return True


def _try_merge(U, V, W, a, b, n, m, p, tol=1e-9):
    """
    Try to replace terms a and b with a single rank-1 term.
    We seek scalars alpha, beta and a rank-1 tensor (u,v,w) such that
        alpha*(U_a,V_a,W_a) + beta*(U_b,V_b,W_b) = (u,v,w)  (rank-1).
    Equivalently, the sum tensor S = U_a*V_a*W_a + ... must be expressible as
    alpha*U_a*V_a*W_a + beta*U_b*V_b*W_b being rank-1.
    We search over the span: any element of span{term_a, term_b} that is rank-1.
    A tensor in the 2-dim span is rank-1 iff there exists (alpha,beta) with the
    Kronecker structure. We solve: find (alpha,beta) such that the resulting
    3-tensor is a single rank-1 term.
    Practical test: the 2D span contains a rank-1 element iff the two terms
    'share' enough structure. We test by checking whether the combined support
    allows a rank-1 factor.
    Returns (new_u, new_v, new_w) lists or None.
    """
    # Build the two tensors as flat arrays over (io,u,v).
    # A term is rank-1: T[io,u,v] = U[u]*V[v]*W[io].
    # We want alpha*Ta + beta*Tb to be rank-1.
    # Let's represent each term by its (u,v,w) vectors (already rank-1).
    ua, va, wa = U[:, a], V[:, a], W[:, a]
    ub, vb, wb = U[:, b], V[:, b], W[:, b]
    # The combined tensor C[io,u,v] = alpha*ua[u]*va[v]*wa[io] + beta*ub[u]*vb[v]*wb[io].
    # For C to be rank-1, it must equal x[u]*y[v]*z[io] for some x,y,z.
    # This is a system. We look for (alpha,beta) in a 2D space.
    # Strategy: pick a reference (io0,u0,v0) where the structure is determined.
    # We'll try: assume the rank-1 factor equals a scaled version. Use the fact that
    # rank-1 3-tensors have all 2x2 minors (in any unfolding) equal to zero.
    # Simpler: brute force over a small set of candidate (alpha,beta) ratios derived
    # from entries, then verify.
    # Collect candidate ratios from entries where both terms are nonzero.
    candidates = set()
    nz = []
    for io in range(p * n):
        for u in range(n * m):
            for v in range(m * p):
                ta = (ua[u] or 0) * (va[v] or 0) * (wa[io] or 0)
                tb = (ub[u] or 0) * (vb[v] or 0) * (wb[io] or 0)
                if ta != 0 or tb != 0:
                    nz.append((io, u, v, ta, tb))
    if not nz:
        return None
    # Candidate (alpha,beta): we need C to be rank-1. Try to determine from two
    # independent entries. Use the first entry with ta!=0 to fix scale.
    # Let C = x*y*z. From entry e: C_e = alpha*ta_e + beta*tb_e = x_ue*y_ve*z_ie.
    # This is underdetermined; instead test rank-1 condition via minors.
    # Build candidate: set beta = 1, solve alpha from one entry, then verify all.
    # Better: the span is 2D; a rank-1 element exists iff the 2D subspace intersects
    # the rank-1 variety. We test by trying alpha/beta from each entry pair.
    # For efficiency, sample a few (alpha,beta) and verify rank-1 by unfolding minors.
    def is_rank1(alpha, beta):
        # Build C and check that it factors.
        # Determine x,y,z up to scale using a nonzero entry.
        # Find first entry with C != 0.
        found = False
        for (io, u, v, ta, tb) in nz:
            cval = alpha * ta + beta * tb
            if abs(cval) > tol:
                # Set z[io]=1, then x[u]*y[v]=cval.
                # We need consistency. Reconstruct using the structure.
                # For a rank-1 tensor, C[io,u,v] = C[io,u,v0]*C[io,u0,v]/C[io,u0,v0]
                # and C[io,u0,v0] nonzero.
                break
        if not found:
            return None
        # Use the found (io,u,v) as reference with Cref=cval.
        # Verify all entries: C[io',u',v'] should equal x[u']y[v']z[io'].
        # We don't know x,y,z separately, but for rank-1 the 3D array has the
        # property that for any (io1,io2,u1,u2,v1,v2) the 2x2 minors vanish in
        # every unfolding. We check the (io,u) unfolding minors for a fixed v.
        # Simpler robust check: build full C and check matrix rank of each unfolding ==1.
        C = np.zeros((p * n, n * m, m * p), dtype=float)
        for (io, u, v, ta, tb) in nz:
            C[io, u, v] = alpha * ta + beta * tb
        # Check unfolding (io, u) for each v: rank <=1
        for v in range(m * p):
            M = C[:, :, v]
            # rank-1 iff all 2x2 minors zero
            # Use SVD-free: check M == M[:,0]*M[0,:] / M[0,0] pattern via minors
            # Find a nonzero pivot.
            pivot = None
            for a2 in range(p * n):
                for b2 in range(n * m):
                    if abs(M[a2, b2]) > tol:
                        pivot = (a2, b2)
                        break
                if pivot:
                    break
            if pivot is None:
                continue
            a0, b0 = pivot
            m00 = M[a0, b0]
            for a2 in range(p * n):
                for b2 in range(n * m):
                    if abs(M[a2, b2] * m00 - M[a2, b0] * M[a0, b2]) > tol * max(1, abs(M[a2,b2]*m00)):
                        return None
        # Also check (io,v) unfolding for a fixed u
        for u in range(n * m):
            M = C[:, u, :]
            pivot = None
            for a2 in range(p * n):
                for b2 in range(m * p):
                    if abs(M[a2, b2]) > tol:
                        pivot = (a2, b2)
                        break
                if pivot:
                    break
            if pivot is None:
                continue
            a0, b0 = pivot
            m00 = M[a0, b0]
            for a2 in range(p * n):
                for b2 in range(m * p):
                    if abs(M[a2, b2] * m00 - M[a2, b0] * M[a0, b2]) > tol * max(1, abs(M[a2,b2]*m00)):
                        return None
        # Reconstruct x,y,z from C
        # find reference
        ref = None
        for (io, u, v, ta, tb) in nz:
            if abs(alpha * ta + beta * tb) > tol:
                ref = (io, u, v)
                break
        if ref is None:
            return None
        io0, u0, v0 = ref
        c0 = C[io0, u0, v0]
        x = np.zeros(n * m, dtype=float)
        y = np.zeros(m * p, dtype=float)
        z = np.zeros(p * n, dtype=float)
        z[io0] = 1.0
        for u in range(n * m):
            x[u] = C[io0, u, v0] / c0
        for v in range(m * p):
            y[v] = C[io0, u0, v] / c0
        # verify full reconstruction
        for io in range(p * n):
            z[io] = (C[io, u0, v0] / c0) if abs(c0) > tol else 0.0
        for io in range(p * n):
            for u in range(n * m):
                for v in range(m * p):
                    if abs(C[io, u, v] - x[u] * y[v] * z[io]) > tol * max(1, abs(C[io, u, v])):
                        return None
        return (x, y, z)

    # Generate candidate (alpha,beta) pairs.
    # Use entries to propose ratios.
    cand = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0)]
    # Add ratios from entries: for entries where tb != 0, alpha = -beta*ta/tb makes C=0 there;
    # but we want a rank-1 element, not zero. Instead, sample alpha/beta = ta/tb patterns.
    for (io, u, v, ta, tb) in nz[:40]:
        if abs(tb) > tol:
            cand.append((ta, tb))
            cand.append((ta, -tb))
        if abs(ta) > tol:
            cand.append((1.0, ta))
    seen = set()
    for (alpha, beta) in cand:
        key = (round(alpha / (abs(alpha) + abs(beta) + 1e-12), 6),
               round(beta / (abs(alpha) + abs(beta) + 1e-12), 6))
        if key in seen:
            continue
        seen.add(key)
        r = is_rank1(alpha, beta)
        if r is not None:
            x, y, z = r
            # Convert to object lists
            xu = [0] * (n * m)
            yv = [0] * (m * p)
            zw = [0] * (p * n)
            for i in range(n * m):
                xu[i] = float(x[i])
            for i in range(m * p):
                yv[i] = float(y[i])
            for i in range(p * n):
                zw[i] = float(z[i])
            return (xu, yv, zw)
    return None


def _reduce_rank(res, n, m, p, deadline):
    """
    Iteratively try to merge pairs of rank-1 terms into one, reducing rank.
    Returns a (possibly lower-rank) valid decomposition.
    """
    U, V, W = _to_arrays(res, n, m, p)
    rank = U.shape[1]
    # Work with list-of-columns representation for easy deletion.
    # Keep as 2D arrays; track active columns.
    active = list(range(rank))

    improved = True
    while improved and time.time() < deadline:
        improved = False
        nactive = len(active)
        for ai in range(nactive):
            if time.time() > deadline:
                break
            for bi in range(ai + 1, nactive):
                a = active[ai]
                b = active[bi]
                merged = _try_merge(U, V, W, a, b, n, m, p)
                if merged is not None:
                    xu, yv, zw = merged
                    # Verify the merge exactly before committing.
                    # Replace columns a and b with a new column, drop b.
                    new_rank = rank - 1
                    U2 = np.zeros((n * m, new_rank), dtype=object)
                    V2 = np.zeros((m * p, new_rank), dtype=object)
                    W2 = np.zeros((p * n, new_rank), dtype=object)
                    # copy all active columns except b, and append merged as last
                    new_active = [c for c in active if c != b]
                    for idx, c in enumerate(new_active):
                        U2[:, idx] = U[:, c]
                        V2[:, idx] = V[:, c]
                        W2[:, idx] = W[:, c]
                    U2[:, new_rank - 1] = xu
                    V2[:, new_rank - 1] = yv
                    W2[:, new_rank - 1] = zw
                    # Full verification of the reduced decomposition.
                    if _verify(U2, V2, W2, n, m, p):
                        U, V, W = U2, V2, W2
                        rank = new_rank
                        active = list(range(rank))
                        improved = True
                        break
            if improved:
                break
    # Build output
    U_out = [[0] * rank for _ in range(n * m)]
    V_out = [[0] * rank for _ in range(m * p)]
    W_out = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        for u in range(n * m):
            U_out[u][c] = U[u, c]
        for v in range(m * p):
            V_out[v][c] = V[v, c]
        for w in range(p * n):
            W_out[w][c] = W[w, c]
    return {"U": U_out, "V": V_out, "W": W_out}


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin

    # 1) Build the blockwise Bini decomposition (rank 72).
    try:
        base = _construct_blockwise(n, m, p)
        U, V, W = _to_arrays(base, n, m, p)
        valid = _verify(U, V, W, n, m, p)
        if not valid:
            return _schoolbook(n, m, p)
        best = base
        best_rank = U.shape[1]
    except Exception:
        return _schoolbook(n, m, p)

    # 2) Try to reduce rank via pair-merging search.
    try:
        if time.time() < deadline:
            reduced = _reduce_rank(base, n, m, p, deadline)
            Ur, Vr, Wr = _to_arrays(reduced, n, m, p)
            if _verify(Ur, Vr, Wr, n, m, p):
                rrank = Ur.shape[1]
                if rrank < best_rank:
                    best = reduced
                    best_rank = rrank
    except Exception:
        pass

    return best

# EVOLVE-BLOCK-END
