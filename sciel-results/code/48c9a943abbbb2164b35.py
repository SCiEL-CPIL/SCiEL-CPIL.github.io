"""Matrix multiplication 4,4,9 tensor decomposition - structured support search + Strassen fallback."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]


def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 4, 4, 9
    n_dim = n * m   # 16
    m_dim = m * p   # 36
    p_dim = p * n   # 36

    # Build full target tensor (flat)
    target = np.zeros(n_dim * m_dim * p_dim, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a * (m_dim * p_dim) + b * p_dim + c] = 1.0

    def verify(U, V, W):
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T.ravel(), target, atol=1e-9)

    def snap_halfint(M, tol=1e-6):
        s = M * 2.0
        if not np.allclose(s, np.round(s), atol=tol):
            return None
        return np.round(s) / 2.0

    def solve_w(U, V, target, n_dim, m_dim, p_dim, rank):
        """Given U (n_dim x rank), V (m_dim x rank), solve for W (p_dim x rank) exactly."""
        # D[a*m_dim + b, r] = U[a,r]*V[b,r];  T2[a*m_dim+b, c] = target
        # Need W[c,:] @ D_row = T2_row  =>  D @ W^T = T2^T
        D = np.empty((n_dim * m_dim, rank), dtype=np.float64)
        # vectorized build
        for a in range(n_dim):
            D[a * m_dim:(a + 1) * m_dim, :] = U[a, :][None, :] * V
        T2 = target.reshape(n_dim * m_dim, p_dim)
        try:
            X, res, rk, sv = np.linalg.lstsq(D, T2.T, rcond=None)
        except Exception:
            return None
        W = X.T  # (p_dim, rank)
        return W

    # ---- Structured support generator for the full 4x4x9 tensor ----
    # For each rank term r, choose a "core" (i,j,k) so that the natural 1-positions
    # are a=i*m+j, b=j*p+k, c=k*n+i. Then scatter +/-1 on a small random subset of
    # rows/cols that are index-consistent with that core, so the linear system for W
    # is much more likely to admit an exact integer solution.
    def gen_supports(rank, rng, u_sp, v_sp, w_sp):
        U = np.zeros((n_dim, rank))
        V = np.zeros((m_dim, rank))
        W = np.zeros((p_dim, rank))
        for r in range(rank):
            i = rng.randrange(n)
            j = rng.randrange(m)
            k = rng.randrange(p)
            a0 = i * m + j
            b0 = j * p + k
            c0 = k * n + i
            # A entries: rows i', cols j' ; keep column j fixed-ish or row i fixed-ish
            a_rows = [i] * min(u_sp, 1) + [rng.randrange(n) for _ in range(u_sp - 1)]
            a_cols = [j] * min(u_sp, 1) + [rng.randrange(m) for _ in range(u_sp - 1)]
            seen = set()
            cnt = 0
            for _ in range(u_sp * 3):
                ar = rng.choice(a_rows)
                ac = rng.choice(a_cols)
                pos = ar * m + ac
                if pos not in seen:
                    seen.add(pos)
                    U[pos, r] = rng.choice([-1, 1])
                    cnt += 1
                    if cnt >= u_sp:
                        break
            # B entries: rows j', cols k'
            b_rows = [j] * min(v_sp, 1) + [rng.randrange(m) for _ in range(v_sp - 1)]
            b_cols = [k] * min(v_sp, 1) + [rng.randrange(p) for _ in range(v_sp - 1)]
            seen = set()
            cnt = 0
            for _ in range(v_sp * 3):
                br = rng.choice(b_rows)
                bc = rng.choice(b_cols)
                pos = br * p + bc
                if pos not in seen:
                    seen.add(pos)
                    V[pos, r] = rng.choice([-1, 1])
                    cnt += 1
                    if cnt >= v_sp:
                        break
            # W entries: rows k', cols i'  (c = k'*n + i')
            w_rows = [k] * min(w_sp, 1) + [rng.randrange(p) for _ in range(w_sp - 1)]
            w_cols = [i] * min(w_sp, 1) + [rng.randrange(n) for _ in range(w_sp - 1)]
            seen = set()
            cnt = 0
            for _ in range(w_sp * 3):
                wr = rng.choice(w_rows)
                wc = rng.choice(w_cols)
                pos = wr * n + wc
                if pos not in seen:
                    seen.add(pos)
                    W[pos, r] = rng.choice([-1, 1])
                    cnt += 1
                    if cnt >= w_sp:
                        break
        return U, V, W

    def search_full(rank, max_iters, seed=0):
        rng = random.Random(seed)
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            u_sp = rng.randint(2, 4)
            v_sp = rng.randint(2, 4)
            w_sp = rng.randint(2, 4)
            U, V, W = gen_supports(rank, rng, u_sp, v_sp, w_sp)
            Wsol = solve_w(U, V, target, n_dim, m_dim, p_dim, rank)
            if Wsol is None:
                continue
            # exactness check
            T = np.einsum('ar,br,cr->abc', U, V, Wsol)
            if not np.allclose(T.ravel(), target, atol=1e-8):
                continue
            Us = snap_halfint(U)
            Vs = snap_halfint(V)
            Ws = snap_halfint(Wsol)
            if Us is None or Vs is None or Ws is None:
                continue
            if verify(Us, Vs, Ws):
                return Us, Vs, Ws
        return None

    # ---- Strassen 4x4x4 construction (rank 49) as verified fallback building block ----
    def build_strassen_4x4x4():
        # 2x2x2 Strassen patterns: each term is (u_dict, v_dict, w_dict) over block locs 0..3
        S22 = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        U = []
        V = []
        W = []
        # 4x4x4 as 2x2 blocks of 2x2. Each outer Strassen term (u,v,w) over 2x2 blocks
        # must itself be realized by a 2x2x2 Strassen at the element level.
        # For outer term with u-blocks set Uset, v-blocks Vset, w-blocks Wset,
        # the product (sum_{a in Uset} A_block)(sum_{b in Vset} B_block)(sum_{c in Wset} C_block)
        # We expand using 2x2x2 Strassen for the 2x2 element multiplication, but the
        # block selection determines which 2x2 sub-matrices participate.
        # Simpler correct approach: treat the 4x4x4 tensor directly and use the 7x7=49
        # Kronecker structure: 4x4x4 = (2x2x2) \otimes (2x2x2) in tensor rank sense,
        # rank = 7*7 = 49. Build via Kronecker of the 2x2x2 Strassen tensors.
        # 2x2x2 Strassen tensor T2[a,b,c] with a=(i,j) i*m+j, b=(j,k), c=(k*n+i) for 2x2x2.
        def strassen_222_tensor():
            n2, m2, p2 = 2, 2, 2
            T = np.zeros((4, 4, 4))
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        a = i * m2 + j
                        b = j * p2 + k
                        c = k * n2 + i
                        T[a, b, c] = 1.0
            # Strassen decomposition of 2x2x2 (rank 7)
            # returns U,V,W each (4,7)
            # Standard Strassen:
            # M1=(A00+A11)(B00+B11)
            # M2=(A10+A11)B00
            # M3=A00(B01-B11)
            # M4=A11(B10-B00)
            # M5=(A00+A10)B11
            # M6=(A11-A10)(B00+B10)
            # M7=(A01-A11)(B10+B11)
            U2 = np.zeros((4, 7))
            V2 = np.zeros((4, 7))
            W2 = np.zeros((4, 7))
            # A index: a = i*2+j ; B: b=j*2+k ; C: c=k*2+i
            # term patterns (u on A blocks, v on B blocks, w on C blocks)
            # A blocks: 00=0,01=1,10=2,11=3 ; B blocks: 00=0,01=1,10=2,11=3 ; C blocks: 00=0,01=1,10=2,11=3
            terms = [
                ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
                ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
                ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
                ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
                ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
                ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
                ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
            ]
            for r, (ud, vd, wd) in enumerate(terms):
                for loc, cf in ud.items():
                    U2[loc, r] += cf
                for loc, cf in vd.items():
                    V2[loc, r] += cf
                for loc, cf in wd.items():
                    W2[loc, r] += cf
            # verify
            Tc = np.einsum('ar,br,cr->abc', U2, V2, W2)
            assert np.allclose(Tc, T, atol=1e-9), "Strassen 2x2x2 failed"
            return U2, V2, W2
        U2, V2, W2 = strassen_222_tensor()
        # Kronecker: 4x4x4 tensor = T2 (Kron) T2. Rank 7*7=49.
        # For index a4 = a_hi*2 + a_lo (a_hi is the 2x2 block row/col index in {0,1}^2 flattened)
        # Actually 4x4x4 indices: a = i*4+j with i,j in 0..3. Write i=2*i1+i0, j=2*j1+j0.
        # Then a = (2*i1+i0)*4 + (2*j1+j0) = i1*8 + i0*4 + j1*2 + j0.
        # The 2x2x2 block structure: outer 2x2x2 uses (i1,j1,k1), inner uses (i0,j0,k0).
        # a_outer = i1*2 + j1 ; a_inner = i0*2 + j0 ; a4 = a_outer*4 + a_inner? check:
        # a_outer*4 + a_inner = (i1*2+j1)*4 + (i0*2+j0) = i1*8 + j1*4 + i0*2 + j0. But we need i1*8 + i0*4 + j1*2 + j0. Mismatch.
        # So the mapping isn't a simple kron on the flattened index. Instead build directly:
        # T4[a4,b4,c4] = T2[a_o,b_o,c_o] * T2[a_i,b_i,c_i] where the decomposition of a4 into
        # (a_o,a_i) must be consistent with the matrix-multiplication index structure.
        # Let's just build T4 explicitly and verify the kron factorization by construction:
        # a4 = i*4+j, i=2*i1+i0, j=2*j1+j0  => a4 = i1*8 + i0*4 + j1*2 + j0
        # a_o (2x2 A index) = i1*2 + j1 ; a_i = i0*2 + j0
        # b4 = j*4+k, j=2*j1+j0, k=2*k1+k0 => b4 = j1*8 + j0*4 + k1*2 + k0
        # b_o = j1*2 + k1 ; b_i = j0*2 + k0
        # c4 = k*4+i, k=2*k1+k0, i=2*i1+i0 => c4 = k1*8 + k0*4 + i1*2 + i0
        # c_o = k1*2 + i1 ; c_i = k0*2 + i0
        # So a4 = i1*8 + i0*4 + j1*2 + j0 ; we need a function a4 -> (a_o,a_i).
        # a_o = i1*2+j1, a_i = i0*2+j0.
        # Given a4, extract bits: a4 = 8*i1 + 4*i0 + 2*j1 + j0.
        # i1 = a4//8, i0 = (a4//4)%2, j1=(a4//2)%2, j0=a4%2.
        # a_o = i1*2 + j1, a_i = i0*2 + j0.
        def decomp4(a4):
            i1 = a4 // 8
            i0 = (a4 // 4) % 2
            j1 = (a4 // 2) % 2
            j0 = a4 % 2
            a_o = i1 * 2 + j1
            a_i = i0 * 2 + j0
            return a_o, a_i
        # Build T4 to verify kron
        T4 = np.zeros((16, 16, 16))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    a4 = i * 4 + j
                    b4 = j * 4 + k
                    c4 = k * 4 + i
                    T4[a4, b4, c4] = 1.0
        # kron factor: U4[a4, (ro,ri)] = U2[a_o(ro), ro]*U2[a_i(ri), ri]
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        for a4 in range(16):
            ao, ai = decomp4(a4)
            for ro in range(7):
                for ri in range(7):
                    U4[a4, ro * 7 + ri] = U2[ao, ro] * U2[ai, ri]
        for b4 in range(16):
            bo, bi = decomp4(b4)
            for ro in range(7):
                for ri in range(7):
                    V4[b4, ro * 7 + ri] = V2[bo, ro] * V2[bi, ri]
        for c4 in range(16):
            co, ci = decomp4(c4)
            for ro in range(7):
                for ri in range(7):
                    W4[c4, ro * 7 + ri] = W2[co, ro] * W2[ci, ri]
        Tc = np.einsum('ar,br,cr->abc', U4, V4, W4)
        assert np.allclose(Tc, T4, atol=1e-9), "Strassen 4x4x4 kron failed"
        # snap to half-integers
        U4 = snap_halfint(U4)
        V4 = snap_halfint(V4)
        W4 = snap_halfint(W4)
        assert U4 is not None and V4 is not None and W4 is not None
        return U4, V4, W4

    # ---- Try to beat 49 for 4x4x4 via search, then assemble 4x4x9 ----
    best_full = None

    # 1) Direct full 4x4x9 search at ranks below 114
    for rank in [113, 112, 111, 110, 109, 108, 107, 106, 105]:
        if time.time() > deadline - 25.0:
            break
        res = search_full(rank, max_iters=30, seed=rank * 7 + 1)
        if res is not None:
            best_full = res
            break

    # 2) Search 4x4x4 below 49, then assemble
    def search_444(rank, max_iters, seed=0):
        n_s, m_s, p_s = 4, 4, 4
        n_d = m_d = p_d = 16
        t4 = np.zeros(n_d * m_d * p_d, dtype=np.float64)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    a = i * 4 + j
                    b = j * 4 + k
                    c = k * 4 + i
                    t4[a * (m_d * p_d) + b * p_d + c] = 1.0
        rng = random.Random(seed)
        for it in range(max_iters):
            if time.time() > deadline - 15.0:
                break
            u_sp = rng.randint(2, 4)
            v_sp = rng.randint(2, 4)
            # structured supports for 4x4x4
            U = np.zeros((16, rank))
            V = np.zeros((16, rank))
            for r in range(rank):
                i = rng.randrange(4); j = rng.randrange(4); k = rng.randrange(4)
                a0 = i * 4 + j; b0 = j * 4 + k
                seen = set(); cnt = 0
                for _ in range(u_sp * 3):
                    ar = i if rng.random() < 0.5 else rng.randrange(4)
                    ac = j if rng.random() < 0.5 else rng.randrange(4)
                    pos = ar * 4 + ac
                    if pos not in seen:
                        seen.add(pos); U[pos, r] = rng.choice([-1, 1]); cnt += 1
                        if cnt >= u_sp: break
                seen = set(); cnt = 0
                for _ in range(v_sp * 3):
                    br = j if rng.random() < 0.5 else rng.randrange(4)
                    bc = k if rng.random() < 0.5 else rng.randrange(4)
                    pos = br * 4 + bc
                    if pos not in seen:
                        seen.add(pos); V[pos, r] = rng.choice([-1, 1]); cnt += 1
                        if cnt >= v_sp: break
            Wsol = solve_w(U, V, t4, 16, 16, 16, rank)
            if Wsol is None:
                continue
            T = np.einsum('ar,br,cr->abc', U, V, Wsol)
            if not np.allclose(T.ravel(), t4, atol=1e-8):
                continue
            Us = snap_halfint(U); Vs = snap_halfint(V); Ws = snap_halfint(Wsol)
            if Us is None or Vs is None or Ws is None:
                continue
            if np.allclose(np.einsum('ar,br,cr->abc', Us, Vs, Ws).ravel(), t4, atol=1e-9):
                return Us, Vs, Ws
        return None

    best444 = None
    for rank in [48, 47, 46, 45]:
        if time.time() > deadline - 20.0:
            break
        res = search_444(rank, max_iters=25, seed=5000 + rank)
        if res is not None:
            best444 = res
            break

    # 3) Assemble 4x4x9 from a 4x4x4 block (rank r4) + second 4x4x4 block + 4x4x1 schoolbook
    def assemble(U4, V4, W4, rank4):
        total = rank4 * 2 + 16
        Uf = np.zeros((16, total))
        Vf = np.zeros((36, total))
        Wf = np.zeros((36, total))
        # block 1: k in 0..3
        Uf[:, 0:rank4] = U4
        Vf[:, 0:rank4] = V4
        Wf[0:16, 0:rank4] = W4
        # block 2: k in 4..7  (shift B col index by +4, C row index by +4*4=16)
        Uf[:, rank4:2 * rank4] = U4
        V4b = np.zeros((36, rank4))
        for j in range(4):
            for kl in range(4):
                V4b[j * 9 + 4 + kl, :] = V4[j * 4 + kl, :]
        Vf[:, rank4:2 * rank4] = V4b
        W4b = np.zeros((36, rank4))
        for k in range(4):
            for i in range(4):
                W4b[16 + k * 4 + i, :] = W4[k * 4 + i, :]
        Wf[16:32, rank4:2 * rank4] = W4b
        # block 3: k = 8 schoolbook
        for t in range(16):
            i = t // 4; j = t % 4
            Uf[i * 4 + j, 2 * rank4 + t] = 1.0
            Vf[j * 9 + 8, 2 * rank4 + t] = 1.0
            Wf[8 * 4 + i, 2 * rank4 + t] = 1.0
        return Uf, Vf, Wf

    candidate = None
    if best444 is not None:
        U4, V4, W4 = best444
        r4 = U4.shape[1]
        Uf, Vf, Wf = assemble(U4, V4, W4, r4)
        if verify(Uf, Vf, Wf):
            candidate = (Uf, Vf, Wf)
    if candidate is None:
        # fallback: Strassen 4x4x4 rank 49
        if time.time() < deadline - 12.0:
            try:
                U4, V4, W4 = build_strassen_4x4x4()
                r4 = U4.shape[1]
                Uf, Vf, Wf = assemble(U4, V4, W4, r4)
                if verify(Uf, Vf, Wf):
                    candidate = (Uf, Vf, Wf)
            except Exception:
                candidate = None

    if best_full is not None:
        Uf, Vf, Wf = best_full
    elif candidate is not None:
        Uf, Vf, Wf = candidate
    else:
        # last-resort: full schoolbook rank 144 (always valid)
        r = 144
        Uf = np.zeros((16, r)); Vf = np.zeros((36, r)); Wf = np.zeros((36, r))
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    t = i * 36 + j * 9 + k
                    Uf[i * 4 + j, t] = 1.0
                    Vf[j * 9 + k, t] = 1.0
                    Wf[k * 4 + i, t] = 1.0
        assert verify(Uf, Vf, Wf)

    def to_list(M):
        return [[float(M[row, col]) for col in range(M.shape[1])] for row in range(M.shape[0])]

    return {"U": to_list(Uf), "V": to_list(Vf), "W": to_list(Wf)}
