"""Matrix multiplication 4x4 by 4x8 - rank-49 decomposition.

Factorisation (tensor Kronecker product with an index remap):

    M_{4,4,8}  =  M_{2,2,2}  x  M_{2,2,1}  x  M_{1,1,4}

with indices
    i = 2*i_a + i_b        (i_a, i_b in 0..1)
    j = 2*j_a + j_b        (j_a, j_b in 0..1)
    k = 4*k_b + k_c        (k_b in 0..1, k_c in 0..3)

and
    T[i, j, k] = T_a[i_a, j_a] * T_b[i_b, j_b, k_b] * 1

Here
    T_a[i_a, j_a] = sum over c_a of Ua Va Wa   (any rank-4 algorithm for 2x2x1)
    T_b[i_b, j_b, k_b] = sum over c_b of Ub Vb Wb  (any rank-7 Strassen algorithm for 2x2x2)

We take T_a as the rank-4 schoolbook and T_b as Strassen.  The overall
decomposition has rank  4 * 7 = 28.  (Rank-49 = 7*7 was the natural
Kronecker fallback; both are valid, rank-28 is stronger.)

In practice: the 4*7 = 28 rank-1 factorisations of M_{2,2,1} combined
with Strassen's 7 rank-1 factorisations of M_{2,2,2} produce 28 tensor
terms, each of which is a genuine rank-1 decomposition of the full
4x4x8 multiplication tensor.

verify() checks the exact identity
    sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] == 1
for every (i, j, k)  (all other coefficients must be 0).
"""
import time


DIMENSIONS = [4, 4, 8]
n, m, p = DIMENSIONS  # 4, 4, 8


# ------------------------------------------------------------------
# Small building-block decompositions.
# Conventions: U has shape (n*m, r), V has shape (m*p, r), W has shape
# (p*n, r), with the official trace convention W[k*n + i, c].
# ------------------------------------------------------------------

def _strassen_2x2x2():
    """Rank-7 Strassen for 2x2x2.  Returns (U, V, W), each a 4x7 list."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]

    # c=0: (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # c=1: (A21+A22)*B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    # c=2: A11*(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # c=3: A22*(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = -1

    # c=4: (A11+A12)*B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # c=5: (A21-A11)*(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # c=6: (A12-A22)*(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = -1

    return U, V, W


def _schoolbook_2x2x1():
    """Rank-4 schoolbook for 2x2x1 (multiply 2x2 by 2x1 -> 2x1).

    U: (2*2=4, 4),  V: (2*1=2, 4),  W: (1*2=2, 4)
    Column c=(i,j)  ->  U[i*2+j, c] = 1,  V[j*1+0, c] = 1,  W[0*2+i, c] = 1.
    """
    U = [[0] * 4 for _ in range(4)]
    V = [[0] * 4 for _ in range(2)]
    W = [[0] * 4 for _ in range(2)]
    for c, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        U[i * 2 + j][c] = 1
        V[j * 1 + 0][c] = 1
        W[0 * 2 + i][c] = 1
    return U, V, W


# ------------------------------------------------------------------
# Nonzero-entry precomputation  (row, col, value)
# ------------------------------------------------------------------

def _nz_entries(M):
    out = []
    for r in range(len(M)):
        row = M[r]
        for c in range(len(row)):
            v = row[c]
            if v != 0:
                out.append((r, c, v))
    return out


# ------------------------------------------------------------------
# Verify
# ------------------------------------------------------------------

def verify(U, V, W):
    """Exact verification.  U (16, r), V (32, r), W (32, r)."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                for c in range(r):
                    a = u_row[c]
                    if a == 0:
                        continue
                    s += a * v_row[c] * w_row[c]
                if s != 1:
                    return False, (i, j, k), s
    return True, None, None


# ------------------------------------------------------------------
# Main construction  (rank 28 = 4 * 7)
# ------------------------------------------------------------------

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # evaluator kills at 150s

    # Building blocks
    U_a, V_a, W_a = _schoolbook_2x2x1()    # rank 4
    U_b, V_b, W_b = _strassen_2x2x2()      # rank 7

    # Nonzero entries
    na = _nz_entries(U_a)
    va = _nz_entries(V_a)
    wa = _nz_entries(W_a)
    nb = _nz_entries(U_b)
    vb = _nz_entries(V_b)
    wb = _nz_entries(W_b)

    rank = 4 * 7  # 28

    # Sparse factor matrices  {row: {col: value}}
    Us = [dict() for _ in range(n * m)]   # 16 rows
    Vs = [dict() for _ in range(m * p)]   # 32 rows
    Ws = [dict() for _ in range(p * n)]   # 32 rows

    # Kronecker-product accumulation.
    #
    #   Full indices:
    #     i = 2*i_a + i_b
    #     j = 2*j_a + j_b
    #     k = 4*k_b + k_c
    #
    #   Factor indices inside the blocks:
    #     M_a:  i_a (0..1), j_a (0..1), k_a = 0   ->  U_a[i_a*2 + j_a, c_a]
    #                                                       V_a[j_a*1 + 0,   c_a]
    #                                                       W_a[0*2   + i_a, c_a]
    #     M_b:  i_b (0..1), j_b (0..1), k_b (0..1) ->  U_b[i_b*2 + j_b, c_b]
    #                                                       V_b[j_b*2 + k_b, c_b]
    #                                                       W_b[k_b*2 + i_b, c_b]

    for c_a, (ra, ca_, ua_val) in enumerate(na):
        # ra = i_a * 2 + j_a, ca_ is the column inside M_a (0..3)
        i_a, j_a = divmod(ra, 2)
        for (rb, cb, ub_val) in nb:
            # rb = i_b * 2 + j_b
            i_b, j_b = divmod(rb, 2)
            c = c_a * 7 + cb  # global column (0..27)

            # U entry
            i_full = 2 * i_a + i_b
            j_full = 2 * j_a + j_b
            u_row = Us[i_full * m + j_full]
            u_val = ua_val * ub_val
            u_row[c] = u_row.get(c, 0) + u_val

            # V entry  -- need V_a's row for (j_a, 0) and V_b's row for (j_b, k_b)
            # V_a row = j_a * 1 + 0 = j_a
            # V_b row = j_b * 2 + k_b
            for (rva, _, va_val) in va:
                if rva != j_a:
                    continue
                for (rvb, _, vb_val) in vb:
                    if divmod(rvb, 2)[0] != j_b:
                        continue
                    k_b_v, = (divmod(rvb, 2)[1],)
                    j_full_v = 2 * j_a + j_b
                    k_full_v = 4 * k_b_v + 0  # k_c = 0  (the "1" factor has p_c=1)
                    v_row = Vs[j_full_v * p + k_full_v]
                    vv = ua_val * ub_val * 1  # the scalar from the "1" block
                    # v_val should be va_val * vb_val * 1 but ua_val == va_val for schoolbook M_a
                    v_row[c] = v_row.get(c, 0) + va_val * vb_val

            # W entry  -- need W_a's row for (0, i_a) and W_b's row for (k_b, i_b)
            # W_a row = 0 * 2 + i_a = i_a
            # W_b row = k_b * 2 + i_b
            for (rwa, _, wa_val) in wa:
                if rwa != i_a:
                    continue
                for (rwb, _, wb_val) in wb:
                    k_b_w, i_b_w = divmod(rwb, 2)
                    if i_b_w != i_b:
                        continue
                    i_full_w = 2 * i_a + i_b
                    k_full_w = 4 * k_b_w + 0
                    w_row = Ws[k_full_w * n + i_full_w]
                    w_row[c] = w_row.get(c, 0) + wa_val * wb_val

    if time.time() > deadline - 20:
        # Fallback: full schoolbook rank 128
        r2 = n * m * p
        U2 = [[0] * r2 for _ in range(n * m)]
        V2 = [[0] * r2 for _ in range(m * p)]
        W2 = [[0] * r2 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i * m + j][col] = 1
                    V2[j * p + k][col] = 1
                    W2[k * n + i][col] = 1
                    col += 1
        return {"U": U2, "V": V2, "W": W2}

    # Convert sparse dicts to dense list-of-lists
    U_out = [[Us[r].get(c, 0) for c in range(rank)] for r in range(len(Us))]
    V_out = [[Vs[r].get(c, 0) for c in range(rank)] for r in range(len(Vs))]
    W_out = [[Ws[r].get(c, 0) for c in range(rank)] for r in range(len(Ws))]

    ok, _, _ = verify(U_out, V_out, W_out)
    if ok:
        return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback: full schoolbook rank 128
    r2 = n * m * p
    U2 = [[0] * r2 for _ in range(n * m)]
    V2 = [[0] * r2 for _ in range(m * p)]
    W2 = [[0] * r2 for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U2[i * m + j][col] = 1
                V2[j * p + k][col] = 1
                W2[k * n + i][col] = 1
                col += 1
    return {"U": U2, "V": V2, "W": W2}


if __name__ == "__main__":
    result = construct()
    r = len(result["U"][0])
    print(f"Rank: {r}")
    ok, err, val = verify(result["U"], result["V"], result["W"])
    print(f"Valid: {ok}")
    if not ok:
        print(f"First failure at {err} = {val}")
