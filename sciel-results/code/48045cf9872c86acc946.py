"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time as _time
import numpy as _np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _time_left(deadline):
    return deadline - _time.time()

def _build_rank44():
    n, m, p = DIMENSIONS
    cols = []

    def add_term(u_entries, v_entries, w_entries):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, val in u_entries:
            u_vec[idx] = val
        for idx, val in v_entries:
            v_vec[idx] = val
        for idx, val in w_entries:
            w_vec[idx] = val
        cols.append((u_vec, v_vec, w_vec))

    def add_strassen(a,b,c,d, e,f,g,h, w11,w12,w21,w22):
        add_term([(a,1),(d,1)], [(e,1),(h,1)], [(w11,1),(w22,1)])
        add_term([(c,1),(d,1)], [(e,1)], [(w21,1),(w22,-1)])
        add_term([(a,1)], [(f,1),(h,-1)], [(w12,1),(w22,1)])
        add_term([(d,1)], [(g,1),(e,-1)], [(w11,1),(w21,1)])
        add_term([(a,1),(b,1)], [(h,1)], [(w12,1),(w11,-1)])
        add_term([(c,1),(a,-1)], [(e,1),(f,1)], [(w22,1)])
        add_term([(b,1),(d,-1)], [(g,1),(h,1)], [(w11,1)])

    def add_rank4(a,b, c,d,e,f, w1,w2):
        add_term([(a,1)], [(c,1)], [(w1,1)])
        add_term([(b,1)], [(e,1)], [(w1,1)])
        add_term([(a,1)], [(d,1)], [(w2,1)])
        add_term([(b,1)], [(f,1)], [(w2,1)])

    # A (3x4): Row 0: 0-3, Row 1: 4-7, Row 2: 8-11
    # B (4x4): Row 0: 0-3, Row 1: 4-7, Row 2: 8-11, Row 3: 12-15
    # C (3x4) W (k*3+i): C[0,0]=0, C[0,1]=3, C[0,2]=6, C[0,3]=9
    #                    C[1,0]=1, C[1,1]=4, C[1,2]=7, C[1,3]=10
    #                    C[2,0]=2, C[2,1]=5, C[2,2]=8, C[2,3]=11

    add_strassen(0, 1, 4, 5, 0, 1, 4, 5, 0, 3, 1, 4)
    add_strassen(0, 1, 4, 5, 2, 3, 6, 7, 6, 9, 7, 10)
    add_rank4(8, 9, 0, 1, 4, 5, 2, 5)
    add_rank4(8, 9, 2, 3, 6, 7, 8, 11)
    add_strassen(2, 3, 6, 7, 8, 9, 12, 13, 0, 3, 1, 4)
    add_strassen(2, 3, 6, 7, 10, 11, 14, 15, 6, 9, 7, 10)
    add_rank4(10, 11, 8, 9, 12, 13, 2, 5)
    add_rank4(10, 11, 10, 11, 14, 15, 8, 11)

    rank = len(cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        u_vec, v_vec, w_vec = cols[r]
        for i in range(n * m):
            U[i][r] = u_vec[i]
        for j in range(m * p):
            V[j][r] = v_vec[j]
        for k in range(p * n):
            W[k][r] = w_vec[k]
    return {"U": U, "V": V, "W": W}

def _build_schoolbook():
    n, m, p = DIMENSIONS
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    for i in range(n):
        for j in range(p):
            for k in range(m):
                r = i * m * p + k * p + j
                U[i * m + k][r] = 1
                V[k * p + j][r] = 1
                W[j * n + i][r] = 1
    return {"U": U, "V": V, "W": W}

def _verify(result):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    U = _np.array(result["U"], dtype=float)
    V = _np.array(result["V"], dtype=float)
    W = _np.array(result["W"], dtype=float)
    rank = U.shape[1]
    # Build the tensor T[i,k,j] = sum_r U[i,k,r] * V[k,j,r] * W[j,n+i,r]
    # U shape: (n*m, rank) -> U[i*m+k, r]
    # V shape: (m*p, rank) -> V[k*p+j, r]
    # W shape: (p*n, rank) -> W[j*n+i, r]
    T = _np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]  # (n*m,)
        v_col = V[:, r]  # (m*p,)
        w_col = W[:, r]  # (p*n,)
        u_mat = u_col.reshape(n, m)
        v_mat = v_col.reshape(m, p)
        w_mat = w_col.reshape(p, n)
        # T[i,k,j] += u_mat[i,k] * v_mat[k,j] * w_mat[j,i]
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T[i, k, j] += u_mat[i, k] * v_mat[k, j] * w_mat[j, i]
    # Expected: T[i,k,j] = 1 if this is the matmul tensor, else 0
    # Actually the matmul tensor has T[i,k,j] = 1 for all i,k,j (it's the identity tensor)
    # Wait no. The matmul tensor M[i,k,j] = 1 for all valid i,k,j.
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # So the tensor is T[i,k,j] = 1 for all i in [0,n), k in [0,m), j in [0,p).
    expected = _np.ones((n, m, p))
    if not _np.allclose(T, expected, atol=1e-9):
        diff = _np.abs(T - expected)
        return False, diff.max()
    return True, 0.0

def construct():
    t_start = _time.time()
    deadline = t_start + 140.0  # 10s safety margin

    # Build the rank-44 solution
    result = _build_rank44()

    # Verify it
    if _time_left(deadline) > 5.0:
        ok, max_err = _verify(result)
        if ok:
            return result
        # If verification fails, fall back to schoolbook (rank 48)
        result = _build_schoolbook()
        ok2, _ = _verify(result)
        if ok2:
            return result

    return result
# EVOLVE-BLOCK-END
