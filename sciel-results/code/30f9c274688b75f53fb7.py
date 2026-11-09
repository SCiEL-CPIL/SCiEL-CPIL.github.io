import numpy as np
import time

DIMENSIONS = [2, 4, 8]


def _check(U, V, W, n=2, m=4, p=8, tol=1e-6):
    """Verify sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    ok = True
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                e = abs(s - 1.0)
                if e > max_err:
                    max_err = e
                if e > tol:
                    ok = False
    return ok, max_err


def _to_int_lists(U, V, W):
    """Convert float arrays to lists with exact half-integer (multiple of 1/4) entries."""
    def conv(A):
        B = []
        for row in A:
            out = []
            for x in row:
                # round to nearest multiple of 1/4
                q = round(x * 4.0)
                out.append(q / 4.0)
            B.append(out)
        return B
    return conv(U), conv(V), conv(W)


def _schoolbook(n=2, m=4, p=8):
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2():
    # Classic Strassen 2x2x2: returns U,V,W each 4x7 (or lists)
    # U = A combos (rows indexed by A entry a00,a01,a10,a11)
    # V = B combos (rows indexed by b00,b01,b10,b11)
    # W = C combos (rows indexed by c00,c01,c10,c11), W[k*n+i] with n=2 -> c_{k,i}
    U = np.array([
        [1, 0, 0, 0, 0, 0, 0],  # a00
        [0, 1, 0, 0, 0, 0, 0],  # a01
        [0, 0, 1, 0, 0, 0, 0],  # a10
        [0, 0, 0, 1, 0, 0, 0],  # a11
    ], dtype=float)
    V = np.array([
        [1, 0, 0, 0, 0, 0, 0],  # b00
        [0, 1, 0, 0, 0, 0, 0],  # b01
        [0, 0, 1, 0, 0, 0, 0],  # b10
        [0, 0, 0, 1, 0, 0, 0],  # b11
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 0, 0, 0, 0],  # c00
        [0, 1, 0, 0, 0, 0, 0],  # c01
        [0, 0, 1, 0, 0, 0, 0],  # c10
        [0, 0, 0, 1, 0, 0, 0],  # c11
    ], dtype=float)
    # Strassen: 7 products M1..M7
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1+M4-M5+M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M6
    # We need to express each U/V/W row as linear combo of the 7 products.
    # Each product c corresponds to a (u_c, v_c, w_c) triple of scalars.
    # For product M_t, the A-part and B-part are fixed vectors; the C-part is the
    # contribution to each c. So U[row a, col t] = coefficient of a in M_t's A-vector,
    # V[row b, col t] = coefficient of b in M_t's B-vector,
    # W[row c, col t] = coefficient of c in M_t's C-contribution.
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    # A vectors (a00,a01,a10,a11)
    A = {
        0: [1, 0, 1, 0],  # M1 (a00+a11)
        1: [0, 0, 1, 1],  # M2 (a10+a11)
        2: [1, 0, 0, 0],  # M3 a00
        3: [0, 0, 0, 1],  # M4 a11
        4: [1, 1, 0, 0],  # M5 (a00+a01)
        5: [1, 0, -1, 0], # M6 (a10-a00)
        6: [0, 1, 0, -1], # M7 (a01-a11)
    }
    # B vectors (b00,b01,b10,b11)
    B = {
        0: [1, 0, 0, 1],  # M1 (b00+b11)
        1: [1, 0, 0, 0],  # M2 b00
        2: [0, 1, 0, -1], # M3 (b01-b11)
        3: [0, 0, 1, -1], # M4 (b10-b00)
        4: [0, 0, 0, 1],  # M5 b11
        5: [1, 1, 0, 0],  # M6 (b00+b01)
        6: [0, 0, 1, 1],  # M7 (b10+b11)
    }
    # C contributions (c00,c01,c10,c11)
    C = {
        0: [1, 0, 0, 1],   # M1 -> c00 + c11
        1: [0, 0, 1, -1],  # M2 -> c10 - c11
        2: [0, 1, 0, 1],   # M3 -> c01 + c11
        3: [1, 0, 1, 0],   # M4 -> c00 + c10
        4: [-1, 1, 0, 0],  # M5 -> -c00 + c01
        5: [0, 0, 0, 1],   # M6 -> c11
        6: [1, 0, 0, 0],   # M7 -> c00
    }
    for t in range(7):
        for a in range(4):
            U[a, t] = A[t][a]
        for b in range(4):
            V[b, t] = B[t][b]
        for c in range(4):
            W[c, t] = C[t][c]
    return U, V, W


def _embed_2x2x2(U2, V2, W2, ai, aj, bk):
    """Embed a 2x2x2 Strassen block into the 2x4x8 tensor.
    ai: which 2 rows of A (rows ai, ai+1); aj: which 2 cols of A (cols aj, aj+1);
    bk: which 2 cols of B (cols bk, bk+1).
    A rows indexed i in {0,1}, A cols j in {0,1} (local) -> global col = aj + j.
    B rows j in {0,1} local -> global row = aj + j.
    B cols k in {0,1} local -> global col = bk + k.
    C rows i, cols k.
    """
    nm, mp, pn = 8, 32, 16
    r = U2.shape[1]
    U = np.zeros((nm, r))
    V = np.zeros((mp, r))
    W = np.zeros((pn, r))
    for local_i in range(2):
        for local_j in range(2):
            gA = (ai + local_i) * 4 + (aj + local_j)  # i*m + j
            for t in range(r):
                U[gA, t] += U2[local_i * 2 + local_j, t]
    for local_j in range(2):
        for local_k in range(2):
            gV = (aj + local_j) * 8 + (bk + local_k)  # j*p + k
            for t in range(r):
                V[gV, t] += V2[local_j * 2 + local_k, t]
    for local_k in range(2):
        for local_i in range(2):
            gW = (bk + local_k) * 2 + (ai + local_i)  # k*n + i
            for t in range(r):
                W[gW, t] += W2[local_k * 2 + local_i, t]
    return U, V, W


def _strassen_full(n=2, m=4, p=8):
    U2, V2, W2 = _strassen_2x2()
    U = np.zeros((n * m, 7))
    V = np.zeros((m * p, 7))
    W = np.zeros((p * n, 7))
    for ai in range(0, n, 2):
        for aj in range(0, m, 2):
            for bk in range(0, p, 2):
                Ub, Vb, Wb = _embed_2x2x2(U2, V2, W2, ai, aj, bk)
                U += Ub
                V += Vb
                W += Wb
    return U, V, W


def _rank4_full(n=2, m=4, p=8):
    # C = A B. Build rank-4 B blocks: for each k, B[:,k] = u * v_k^T (u in R^4, v_k in R^4)
    # Then C[:,k] = (A u) v_k^T.
    # Tensor: C[i,k] = sum_t (sum_j A[i,j] u[j]) * v_k[t]
    # = sum_j sum_t A[i,j] (u[j] v_k[t]) ... but v depends on k.
    # Standard: C[i,k] = sum_j A[i,j] B[j,k].
    # With B[j,k] = u[j] * v_k[j]? No. Let B = u v^T where u in R^4 (rows j), v in R^4 (cols k).
    # B[j,k] = u[j] v[k]. Then C[i,k] = sum_j A[i,j] u[j] v[k] = (sum_j A[i,j] u[j]) * v[k].
    # So for a single rank-1 B, we need: A-part vector a[i] = sum_j A[i,j] u[j] -> U[i,j] = u[j].
    # B-part: V[j,k] = v[k] (independent of j). C-part: W[k,i] = 1 (independent of i,k).
    # That's a rank-1 term in the tensor? U[i,j]=u[j], V[j,k]=v[k], W[k,i]=1.
    # Check: sum_j sum_k U[i,j] V[j,k] W[k,i]?? No, the tensor contraction is
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]. For a single term c:
    # = U[i,j] V[j,k] W[k,i] (no sum over j,k; j,k are fixed by the equation).
    # Wait, the equation is: for FIXED (i,j,k), sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1.
    # So each (i,j,k) is a separate equation, and c is the only sum. So a rank-1 term
    # contributes U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] to equation (i,j,k).
    # For the rank-1-B construction to work per (i,j,k):
    # We want sum_j A[i,j] B[j,k] reproduced. But the tensor has NO sum over j!
    # The tensor T[i,j,k]=1 means: the (i,j,k) entry of the multiplication tensor is 1,
    # and the bilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k].
    # The rank-1 decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c where
    # (u_c ⊗ v_c ⊗ w_c)[i,j,k] = u_c[i] v_c[j] w_c[k], and we need
    # sum_c u_c[i] v_c[j] w_c[k] = 1 for all i,j,k.
    # In the interface, U[i*m+j, c] = u_c[i,j]? No. Let me re-read the convention.
    # U has shape (n*m, rank), so U[i*m+j, c]. V has shape (m*p, rank): V[j*p+k, c].
    # W has shape (p*n, rank): W[k*n+i, c].
    # The condition: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k.
    # This is a 3-tensor with entries indexed by (i,j) for U, (j,k) for V, (k,i) for W.
    # This is the standard matrix multiplication tensor where the three factors share
    # indices j (between U and V), k (between V and W), i (between W and U).
    # So it's: sum_c U_{(i,j),c} V_{(j,k),c} W_{(k,i),c} = 1.
    # This is exactly the bilinear form: sum_{i,j,k} (sum_c U_{ij,c} V_{jk,c} W_{ki,c}) A_{ij} B_{jk}
    # = sum_{i,k} (sum_j sum_c ...) ... The coefficient of A_{ij} B_{jk} is
    # sum_c U_{ij,c} V_{jk,c} W_{ki,c} which must equal 1 (the (i,j,k) entry of the tensor,
    # which is 1). Yes! So the condition is correct.
    # Now, for a rank-1 term: U_{ij,c}=u_c[i]v_c[j], V_{jk,c}=v_c[j]w_c[k], W_{ki,c}=w_c[k]u_c[i].
    # Product = u_c[i]^2 v_c[j]^2 w_c[k]^2. Sum over c must = 1 for all i,j,k.
    # That's very restrictive. The schoolbook uses u_c[i]=delta, v_c[j]=delta, w_c[k]=delta.
    # So a "rank-4 matrix" idea doesn't directly map. Let me instead use the
    # bilinear circuit view: the tensor rank = number of scalar multiplications in a
    # bilinear algorithm. For C = A B with B = sum_t u_t v_t^T (rank r_B),
    # C = sum_t (A u_t) v_t^T. The bilinear algorithm:
    #   For each t: compute x_t = A u_t (this is r_B linear combos, no mults in bilinear count? 
    #   Actually A u_t involves multiplications A[i,j]*u_t[j]).
    # In bilinear rank, each scalar mult counts. A u_t: for each i, x_t[i] = sum_j A[i,j] u_t[j]
    # = r_B? No, u_t is a fixed vector (part of the algorithm), so x_t[i] = sum_j A[i,j] u_t[j]
    # requires m multiplications per i (A[i,j]*u_t[j]), total n*m per t.
    # Then C[i,k] = sum_t x_t[i] v_t[k]: n*p per t.
    # Total per t: n*m + n*p = n(m+p). For r_B = 4: 4 * 2*(4+8) = 4*24 = 96. Worse.
    # So rank-4-B is not helpful. Skip.
    # Return schoolbook as fallback.
    return _schoolbook(n, m, p)


def construct():
    start = time.time()
    deadline = start + 140.0

    n, m, p = DIMENSIONS
    candidates = []

    # Candidate 1: Strassen-based (rank 56)
    try:
        U, V, W = _strassen_full(n, m, p)
        ok, err = _check(U, V, W, n, m, p)
        if ok:
            Ui, Vi, Wi = _to_int_lists(U, V, W)
            candidates.append((56, Ui, Vi, Wi, "strassen"))
    except Exception:
        pass

    # Candidate 2: Rank-4-B construction (may or may not be valid)
    if time.time() < deadline - 20.0:
        try:
            U, V, W = _rank4_full(n, m, p)
            ok, err = _check(U, V, W, n, m, p)
            if ok:
                Ui, Vi, Wi = _to_int_lists(U, V, W)
                candidates.append((U.shape[1], Ui, Vi, Wi, "rank4"))
        except Exception:
            pass

    # Candidate 3: Schoolbook (rank 64) - always valid fallback
    U, V, W = _schoolbook(n, m, p)
    ok, err = _check(U, V, W, n, m, p)
    if ok:
        Ui, Vi, Wi = _to_int_lists(U, V, W)
        candidates.append((64, Ui, Vi, Wi, "schoolbook"))

    if not candidates:
        # Should not happen; force schoolbook
        U, V, W = _schoolbook(n, m, p)
        Ui, Vi, Wi = _to_int_lists(U, V, W)
        return {"U": Ui, "V": Vi, "W": Wi}

    # Pick the one with smallest rank
    candidates.sort(key=lambda x: x[0])
    _, best_U, best_V, best_W, _name = candidates[0]
    return {"U": best_U, "V": best_V, "W": best_W}


if __name__ == "__main__":
    # Self-test
    n, m, p = DIMENSIONS
    U, V, W = _strassen_full(n, m, p)
    ok, err = _check(U, V, W, n, m, p)
    print("Strassen valid:", ok, "max_err:", err, "rank:", U.shape[1])
    out = construct()
    ok2, err2 = _check(out["U"], out["V"], out["W"], n, m, p)
    print("construct valid:", ok2, "max_err:", err2,
          "rank:", len(out["U"][0]) if out["U"] else None)
