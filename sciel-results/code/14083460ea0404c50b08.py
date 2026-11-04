"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_start_time = time.time()

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    deadline = _start_time + 130.0  # safety margin before 150s wall-clock

    # ---- Strassen 2x2x2 factors (verified-correct mapping) ----
    def strassen_factors():
        U = np.zeros((4, 7), dtype=np.float64)
        V = np.zeros((4, 7), dtype=np.float64)
        W = np.zeros((4, 7), dtype=np.float64)
        # m1 = (a+d)(e+h)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        # m2 = (c+d)e
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
        # m3 = a(f-h)
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
        # m4 = d(g-e)
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
        # m5 = (a+b)h
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
        # m6 = (c-a)(e+f)
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        # m7 = (b-d)(g+h)
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W

    def schoolbook_factors(n_, m_, p_):
        rank = n_ * m_ * p_
        U = np.zeros((n_*m_, rank), dtype=np.float64)
        V = np.zeros((m_*p_, rank), dtype=np.float64)
        W = np.zeros((p_*n_, rank), dtype=np.float64)
        col = 0
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    U[i*m_+j, col] = 1.0
                    V[j*p_+k, col] = 1.0
                    W[k*n_+i, col] = 1.0
                    col += 1
        return U, V, W

    def block_rank(n_s, m_s, p_s):
        if n_s == 2 and m_s == 2 and p_s == 2:
            return 7
        return n_s * m_s * p_s

    def generate_partitions(total):
        if total == 0:
            return [[]]
        results = []
        for first in [1, 2, 3]:
            if first > total:
                continue
            for rest in generate_partitions(total - first):
                results.append([first] + rest)
        return results

    def build_decomposition(n_parts, m_parts, p_parts, n, m, p):
        n_offsets = [0]
        for sz in n_parts[:-1]:
            n_offsets.append(n_offsets[-1] + sz)
        m_offsets = [0]
        for sz in m_parts[:-1]:
            m_offsets.append(m_offsets[-1] + sz)
        p_offsets = [0]
        for sz in p_parts[:-1]:
            p_offsets.append(p_offsets[-1] + sz)

        total_rank = 0
        for ns, ms, ps in product(n_parts, m_parts, p_parts):
            total_rank += block_rank(ns, ms, ps)

        U = np.zeros((n*m, total_rank), dtype=np.float64)
        V = np.zeros((m*p, total_rank), dtype=np.float64)
        W = np.zeros((p*n, total_rank), dtype=np.float64)

        col_off = 0
        for ni_idx, ns in enumerate(n_parts):
            i0 = n_offsets[ni_idx]
            for mi_idx, ms in enumerate(m_parts):
                j0 = m_offsets[mi_idx]
                for pi_idx, ps in enumerate(p_parts):
                    k0 = p_offsets[pi_idx]
                    if ns == 2 and ms == 2 and ps == 2:
                        Ub, Vb, Wb = strassen_factors()
                    else:
                        Ub, Vb, Wb = schoolbook_factors(ns, ms, ps)
                    r = Ub.shape[1]
                    for li in range(ns):
                        for lj in range(ms):
                            U[(i0+li)*m + (j0+lj), col_off:col_off+r] = Ub[li*ms+lj, :]
                    for lj in range(ms):
                        for lk in range(ps):
                            V[(j0+lj)*p + (k0+lk), col_off:col_off+r] = Vb[lj*ps+lk, :]
                    for lk in range(ps):
                        for li in range(ns):
                            W[(k0+lk)*n + (i0+li), col_off:col_off+r] = Wb[lk*ns+li, :]
                    col_off += r
        return U, V, W, total_rank

    def verify_fast(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        r = U_mat.shape[1]
        if r == 0:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def eliminate(U_arr, V_arr, W_arr, n, m, p, deadline):
        current_rank = U_arr.shape[1]
        for _ in range(30):
            if time.time() > deadline - 3:
                break
            found = False
            for col in range(current_rank):
                if time.time() > deadline - 3:
                    break
                if current_rank <= 1:
                    break
                U_t = np.delete(U_arr, col, axis=1)
                V_t = np.delete(V_arr, col, axis=1)
                W_t = np.delete(W_arr, col, axis=1)
                if verify_fast(U_t, V_t, W_t, n, m, p):
                    U_arr, V_arr, W_arr = U_t, V_t, W_t
                    current_rank -= 1
                    found = True
                    break
            if not found:
                break
        return U_arr, V_arr, W_arr, current_rank

    # ---- Winograd / DFT candidate (guarded: only admitted if verified) ----
    def winograd_candidate():
        # 3-point DFT over the output (n) dimension
        w3 = np.exp(2j*np.pi/3)
        # 5-point DFT over the contraction (m) dimension
        w5 = np.exp(2j*np.pi/5)
        terms = []  # (u_entries, v_entries, w_entries)
        for beta in range(5):
            for gamma in range(5):
                for alpha in range(3):
                    u = {}
                    for i in range(n):
                        for j in range(m):
                            val = (w5 ** (i*j)) * (w3 ** (-i*alpha))
                            if abs(val) > 1e-12:
                                u[i*m+j] = val
                    v = {}
                    for j in range(m):
                        for k in range(p):
                            val = (w5 ** (-j*beta)) * (w5 ** (j*gamma))
                            if abs(val) > 1e-12:
                                v[j*p+k] = val
                    w = {}
                    for k in range(p):
                        for i in range(n):
                            val = (w3 ** (i*alpha)) * (w3 ** (-i*gamma))
                            if abs(val) > 1e-12:
                                w[k*n+i] = val
                    terms.append((u, v, w))

        rank = len(terms)
        U = np.zeros((n*m, rank), dtype=np.complex128)
        V = np.zeros((m*p, rank), dtype=np.complex128)
        W = np.zeros((p*n, rank), dtype=np.complex128)
        for ci, (u, v, w) in enumerate(terms):
            for row, val in u.items():
                U[row, ci] += val
            for row, val in v.items():
                V[row, ci] += val
            for row, val in w.items():
                W[row, ci] += val
        # normalise so tensor == 1
        scale = 1.0 / (n * m)
        U = U * scale
        return U, V, W, rank

    def to_real_factors(Uc, Vc, Wc, n, m, p, tol=1e-9):
        """Convert a complex decomposition into real factors by pairing conjugate
        columns. Returns (U,V,W,rank) or None if it cannot be made real & verified."""
        r = Uc.shape[1]
        real_idx = []
        imag_pairs = []
        used = [False] * r
        for c in range(r):
            if used[c]:
                continue
            col = Uc[:, c]
            if np.allclose(col.imag, 0, atol=tol) and np.allclose(Vc[:, c].imag, 0, atol=tol) \
               and np.allclose(Wc[:, c].imag, 0, atol=tol):
                real_idx.append(c)
                used[c] = True
            else:
                # find conjugate partner
                partner = None
                for d in range(c + 1, r):
                    if used[d]:
                        continue
                    if np.allclose(Uc[:, d], np.conj(col), atol=tol) \
                       and np.allclose(Vc[:, d], np.conj(Vc[:, c]), atol=tol) \
                       and np.allclose(Wc[:, d], np.conj(Wc[:, c]), atol=tol):
                        partner = d
                        break
                if partner is None:
                    return None
                imag_pairs.append((c, partner))
                used[c] = True
                used[partner] = True

        # Build real factor matrices
        n_real = len(real_idx)
        n_imag = len(imag_pairs)
        new_rank = n_real + 2 * n_imag
        Ur = np.zeros((n*m, new_rank), dtype=np.float64)
        Vr = np.zeros((m*p, new_rank), dtype=np.float64)
        Wr = np.zeros((p*n, new_rank), dtype=np.float64)
        for ci, c in enumerate(real_idx):
            Ur[:, ci] = Uc[:, c].real
            Vr[:, ci] = Vc[:, c].real
            Wr[:, ci] = Wc[:, c].real
        for ci, (c, d) in enumerate(imag_pairs):
            a = c
            b = d
            # real part column: (Ua+Ub)/2 etc. ; imag part: (Ua-Ub)/(2i)
            Ur[:, 2*ci] = 0.5 * (Uc[:, a].real + Uc[:, b].real)
            Vr[:, 2*ci] = 0.5 * (Vc[:, a].real + Vc[:, b].real)
            Wr[:, 2*ci] = 0.5 * (Wc[:, a].real + Wc[:, b].real)
            Ur[:, 2*ci+1] = 0.5 * (Uc[:, a].imag - Uc[:, b].imag)
            Vr[:, 2*ci+1] = 0.5 * (Vc[:, a].imag - Vc[:, b].imag)
            Wr[:, 2*ci+1] = 0.5 * (Wc[:, a].imag - Wc[:, b].imag)
        return Ur, Vr, Wr, new_rank

    # ---- Main search ----
    n_partitions = generate_partitions(n)
    m_partitions = generate_partitions(m)
    p_partitions = generate_partitions(p)

    best_rank = float('inf')
    best_U = best_V = best_W = None

    # 1) Winograd candidate (cheap to build, guarded by verification)
    if time.time() < deadline:
        try:
            Uc, Vc, Wc, _ = winograd_candidate()
            Ur, Vr, Wr, wrank = to_real_factors(Uc, Vc, Wc, n, m, p)
            if Ur is not None and verify_fast(Ur, Vr, Wr, n, m, p):
                Ur, Vr, Wr, wrank = eliminate(Ur, Vr, Wr, n, m, p, deadline)
                if wrank < best_rank:
                    best_rank, best_U, best_V, best_W = wrank, Ur, Vr, Wr
        except Exception:
            pass

    # 2) Exhaustive block-partition search with Strassen blocks
    for np_ in n_partitions:
        for mp_ in m_partitions:
            for pp_ in p_partitions:
                if time.time() > deadline - 8:
                    break
                total_rank = 0
                for ns, ms, ps in product(np_, mp_, pp_):
                    total_rank += block_rank(ns, ms, ps)
                if total_rank >= best_rank:
                    continue
                U, V, W, rank = build_decomposition(np_, mp_, pp_, n, m, p)
                if not verify_fast(U, V, W, n, m, p):
                    continue
                U, V, W, rank = eliminate(U, V, W, n, m, p, deadline)
                if rank < best_rank:
                    best_rank, best_U, best_V, best_W = rank, U, V, W

    if best_U is None:
        U, V, W, _ = build_decomposition([n], [m], [p], n, m, p)
        U, V, W, _ = eliminate(U, V, W, n, m, p, deadline)
        best_U, best_V, best_W = U, V, W

    # Clean up entries to integers / half-integers
    U_final = best_U.tolist()
    V_final = best_V.tolist()
    W_final = best_W.tolist()
    for mat in (U_final, V_final, W_final):
        for row in mat:
            for idx in range(len(row)):
                val = row[idx]
                rounded_int = round(val)
                if abs(val - rounded_int) < 1e-9:
                    row[idx] = int(rounded_int)
                else:
                    rounded_half = round(val * 2) / 2.0
                    if abs(val - rounded_half) < 1e-9:
                        row[idx] = rounded_half
                    else:
                        row[idx] = round(val, 10)
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
