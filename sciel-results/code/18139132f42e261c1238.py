"""Block-decomposition matrix multiplication tensor using Strassen for 2x2x2 blocks."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return (U, V, W) each a list of 7 column-vectors of length 4 for the
    2x2x2 matrix multiplication tensor, in row-major (i*2+j) / (j*2+k) / (k*2+i) order.
    U[i*2+j] uses row i; V[j*2+k] uses column j; W[k*2+i] uses row i."""
    # Each entry: (U_row, V_col, W_row) with coefficient 1 (all +/-1 handled by sign)
    # Strassen:
    # M1=(A11+A22)(B11+B22)
    # M2=(A21+A22)B11
    # M3=A11(B12-B22)
    # M4=A22(B21-B11)
    # M5=(A11+A12)B22
    # M6=(A21-A11)(B11+B12)
    # M7=(A12-A22)(B21+B22)
    # C11=M1+M4-M5+M7
    # C12=M3+M5
    # C21=M2+M4
    # C22=M1-M2+M3+M6
    # U: for each factor, which A row (i) is used (with sign)
    # V: which B col (j)
    # W: which C row (i)
    # We build 7 columns. Each column corresponds to one M.
    # For a factor M = (sum a_i * A_ri)(sum b_j * B_cj), the U vector has entries
    # a_i at positions for row ri (across both columns of that row), V vector has
    # b_j at positions for column cj (across both rows of that col).
    # The W vector: for each output (i,k) in C, coefficient c_{i,k}^{(M)} at position k*2+i.

    U = [ [0]*4 for _ in range(7) ]
    V = [ [0]*4 for _ in range(7) ]
    W = [ [0]*4 for _ in range(7) ]

    # Define each M by (U entries: list of (row_i, coeff) for the A side),
    # (V entries: list of (col_j, coeff) for the B side),
    # (W entries: list of (out_i, out_k, coeff) for C side)
    def aidx(i, j): return i*2 + j   # A row-major index for (i,j)
    def bidx(j, k): return j*2 + k   # B row-major index for (j,k)
    def cidx(k, i): return k*2 + i   # C index per trace convention W[k*n+i]

    # M1 = (A11+A22)(B11+B22)
    # A side: row0 col0 (+1), row1 col1 (+1) -> U positions aidx(0,0)=0, aidx(1,1)=3
    # B side: col0 row0 (+1), col0 row1 (+1) -> V positions bidx(0,0)=0, bidx(0,1)=1
    # C side: C11 += M1 -> (i=0,k=0) coeff +1 ; C22 += M1 -> (i=1,k=1) coeff +1
    U[0][aidx(0,0)] = 1; U[0][aidx(1,1)] = 1
    V[0][bidx(0,0)] = 1; V[0][bidx(0,1)] = 1
    W[0][cidx(0,0)] = 1; W[0][cidx(1,1)] = 1

    # M2 = (A21+A22)B11
    # A side: row1 col0 (+1), row1 col1 (+1) -> aidx(1,0)=2, aidx(1,1)=3
    # B side: col0 row0 (+1) -> bidx(0,0)=0
    # C side: C21 += M2 -> (i=1,k=0) coeff +1
    U[1][aidx(1,0)] = 1; U[1][aidx(1,1)] = 1
    V[1][bidx(0,0)] = 1
    W[1][cidx(0,1)] = 1

    # M3 = A11(B12-B22)
    # A side: row0 col0 (+1) -> aidx(0,0)=0
    # B side: col1 row0 (+1), col1 row1 (-1) -> bidx(1,0)=2, bidx(1,1)=3
    # C side: C12 += M3 -> (i=0,k=1) +1 ; C22 += M3 -> (i=1,k=1) +1
    U[2][aidx(0,0)] = 1
    V[2][bidx(1,0)] = 1; V[2][bidx(1,1)] = -1
    W[2][cidx(1,0)] = 1; W[2][cidx(1,1)] = 1

    # M4 = A22(B21-B11)
    # A side: row1 col1 (+1) -> aidx(1,1)=3
    # B side: col1 row0 (+1), col0 row0 (-1) -> bidx(1,0)=2, bidx(0,0)=0
    # C side: C11 += M4 -> (i=0,k=0) +1 ; C21 += M4 -> (i=1,k=0) +1
    U[3][aidx(1,1)] = 1
    V[3][bidx(1,0)] = 1; V[3][bidx(0,0)] = -1
    W[3][cidx(0,0)] = 1; W[3][cidx(0,1)] = 1

    # M5 = (A11+A12)B22
    # A side: row0 col0 (+1), row0 col1 (+1) -> aidx(0,0)=0, aidx(0,1)=1
    # B side: col1 row1 (+1) -> bidx(1,1)=3
    # C side: C11 -= M5 -> (i=0,k=0) -1 ; C12 += M5 -> (i=0,k=1) +1
    U[4][aidx(0,0)] = 1; U[4][aidx(0,1)] = 1
    V[4][bidx(1,1)] = 1
    W[4][cidx(0,0)] = -1; W[4][cidx(1,0)] = 1

    # M6 = (A21-A11)(B11+B12)
    # A side: row1 col0 (+1), row0 col0 (-1) -> aidx(1,0)=2, aidx(0,0)=0
    # B side: col0 row0 (+1), col1 row0 (+1) -> bidx(0,0)=0, bidx(1,0)=2
    # C side: C22 += M6 -> (i=1,k=1) +1
    U[5][aidx(1,0)] = 1; U[5][aidx(0,0)] = -1
    V[5][bidx(0,0)] = 1; V[5][bidx(1,0)] = 1
    W[5][cidx(1,1)] = 1

    # M7 = (A12-A22)(B21+B22)
    # A side: row0 col1 (+1), row1 col1 (-1) -> aidx(0,1)=1, aidx(1,1)=3
    # B side: col1 row0 (+1), col1 row1 (+1) -> bidx(1,0)=2, bidx(1,1)=3
    # C side: C11 += M7 -> (i=0,k=0) +1
    U[6][aidx(0,1)] = 1; U[6][aidx(1,1)] = -1
    V[6][bidx(1,0)] = 1; V[6][bidx(1,1)] = 1
    W[6][cidx(0,0)] = 1

    return U, V, W


def _block_decompose(n, m, p, ns, ms, ps):
    """Build U,V,W for block decomposition.
    ns: list of row-block sizes summing to n
    ms: list of col-block sizes summing to m
    ps: list of out-block sizes summing to p
    Returns (U, V, W) as lists of lists (row-major) and total rank."""
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n

    U = [[0] * 0 for _ in range(U_rows)]  # placeholder; we build columns then transpose
    # Instead build as list of columns then assemble.
    U_cols = []
    V_cols = []
    W_cols = []

    # Precompute Strassen for 2x2x2
    sU, sV, sW = _strassen_2x2()

    for bi, ni in enumerate(ns):
        for bj, mj in enumerate(ms):
            for bk, pk in enumerate(ps):
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen: 7 columns
                    # A block rows: (i in 0..1, j in 0..1) -> global row i0+i, j0+j
                    # B block cols: (j in 0..1, k in 0..1) -> global col j0+j, k0+k
                    # C block: (k in 0..1, i in 0..1) -> global out k0+k, i0+i
                    i0 = sum(ns[:bi])
                    j0 = sum(ms[:bj])
                    k0 = sum(ps[:bk])
                    for t in range(7):
                        uc = [0] * U_rows
                        vc = [0] * V_rows
                        wc = [0] * W_rows
                        # U: sU[t] has entries at local aidx(i,j) = i*2+j
                        for li in range(2):
                            for lj in range(2):
                                val = sU[t][li*2 + lj]
                                if val != 0:
                                    uc[(i0 + li) * m + (j0 + lj)] = val
                        # V: sV[t] at local bidx(j,k) = j*2+k
                        for lj in range(2):
                            for lk in range(2):
                                val = sV[t][lj*2 + lk]
                                if val != 0:
                                    vc[(j0 + lj) * p + (k0 + lk)] = val
                        # W: sW[t] at local cidx(k,i) = k*2+i
                        for lk in range(2):
                            for li in range(2):
                                val = sW[t][lk*2 + li]
                                if val != 0:
                                    wc[(k0 + lk) * n + (i0 + li)] = val
                        U_cols.append(uc)
                        V_cols.append(vc)
                        W_cols.append(wc)
                else:
                    # Schoolbook: ni*mj*pk columns
                    i0 = sum(ns[:bi])
                    j0 = sum(ms[:bj])
                    k0 = sum(ps[:bk])
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                uc = [0] * U_rows
                                vc = [0] * V_rows
                                wc = [0] * W_rows
                                uc[(i0 + li) * m + (j0 + lj)] = 1
                                vc[(j0 + lj) * p + (k0 + lk)] = 1
                                wc[(k0 + lk) * n + (i0 + li)] = 1
                                U_cols.append(uc)
                                V_cols.append(vc)
                                W_cols.append(wc)

    rank = len(U_cols)
    # Transpose columns to rows
    U = [[U_cols[c][r] for c in range(rank)] for r in range(U_rows)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(V_rows)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(W_rows)]
    return U, V, W, rank


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Build by iterating over c
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u = Ua[:, c]
        v = Va[:, c]
        w = Wa[:, c]
        # u is length n*m, v is length m*p, w is length p*n
        # T[i,j,k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        # Build outer: for each i, u_i (length m); for each j, v_j (length p); for each k, w_k (length n)
        u_reshaped = u.reshape(n, m)
        v_reshaped = v.reshape(m, p)
        w_reshaped = w.reshape(p, n)
        # T[i,j,k] += u_reshaped[i,j] * v_reshaped[j,k] * w_reshaped[k,i]
        # Use einsum: 'ij,jk,ki->ijk'
        contrib = np.einsum('ij,jk,ki->ijk', u_reshaped, v_reshaped, w_reshaped)
        T += contrib
    # Expected: T[i,j,k] = 1 if (i,j,k) valid else 0. Actually the multiplication tensor
    # has T[i,j,k] = 1 for all i,j,k (it's the identity tensor for matmul).
    expected = np.ones((n, m, p), dtype=float)
    return np.max(np.abs(T - expected)) <= tol


def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s

    n, m, p = DIMENSIONS

    # Splits: n=3 as 2+1, m=5 as 2+2+1, p=6 as 2+2+2
    ns = [2, 1]
    ms = [2, 2, 1]
    ps = [2, 2, 2]

    U, V, W, rank = _block_decompose(n, m, p, ns, ms, ps)

    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Should not happen for this fixed construction, but guard
        pass

    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
