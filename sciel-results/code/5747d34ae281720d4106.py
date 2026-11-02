"""Combined Strassen decomposition with ALS refinement for 4x4 * 4x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Verify exact decomposition using numpy."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return np.allclose(T, 1.0, atol=atol)

def _round_to_half(x):
    """Round to nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix(M):
    """Round all entries to nearest half-integer."""
    return np.array([[_round_to_half(x) for x in row] for row in M])

def _strassen_construct(n=4, m=4, p=5):
    """Construct Strassen-based block decomposition (rank 64)."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u.copy())
        V_cols.append(v.copy())
        W_cols.append(w.copy())
    
    def add_strassen_2x2_2(ai, aj, bj, bk):
        """Add 7 Strassen terms for 2x2 * 2x2 -> 2x2."""
        # M1 = (a+d)(e+h)
        u = np.zeros(n*m); u[ai*m+aj] = 1; u[(ai+1)*m+(aj+1)] = 1
        v = np.zeros(m*p); v[bj*p+bk] = 1; v[(bj+1)*p+(bk+1)] = 1
        w = np.zeros(p*n); w[bk*n+ai] = 1; w[(bk+1)*n+(ai+1)] = 1
        add_term(u, v, w)
        
        # M2 = (c+d)e
        u = np.zeros(n*m); u[(ai+1)*m+aj] = 1; u[(ai+1)*m+(aj+1)] = 1
        v = np.zeros(m*p); v[bj*p+bk] = 1
        w = np.zeros(p*n); w[bk*n+(ai+1)] = 1; w[(bk+1)*n+(ai+1)] = -1
        add_term(u, v, w)
        
        # M3 = a(f-h)
        u = np.zeros(n*m); u[ai*m+aj] = 1
        v = np.zeros(m*p); v[bj*p+(bk+1)] = 1; v[(bj+1)*p+(bk+1)] = -1
        w = np.zeros(p*n); w[(bk+1)*n+ai] = 1; w[(bk+1)*n+(ai+1)] = 1
        add_term(u, v, w)
        
        # M4 = d(g-e)
        u = np.zeros(n*m); u[(ai+1)*m+(aj+1)] = 1
        v = np.zeros(m*p); v[(bj+1)*p+bk] = 1; v[bj*p+bk] = -1
        w = np.zeros(p*n); w[bk*n+ai] = 1; w[bk*n+(ai+1)] = 1
        add_term(u, v, w)
        
        # M5 = (a+b)h
        u = np.zeros(n*m); u[ai*m+aj] = 1; u[ai*m+(aj+1)] = 1
        v = np.zeros(m*p); v[(bj+1)*p+(bk+1)] = 1
        w = np.zeros(p*n); w[bk*n+ai] = -1; w[(bk+1)*n+ai] = 1
        add_term(u, v, w)
        
        # M6 = (c-a)(e+f)
        u = np.zeros(n*m); u[(ai+1)*m+aj] = 1; u[ai*m+aj] = -1
        v = np.zeros(m*p); v[bj*p+bk] = 1; v[bj*p+(bk+1)] = 1
        w = np.zeros(p*n); w[(bk+1)*n+(ai+1)] = 1
        add_term(u, v, w)
        
        # M7 = (b-d)(g+h)
        u = np.zeros(n*m); u[ai*m+(aj+1)] = 1; u[(ai+1)*m+(aj+1)] = -1
        v = np.zeros(m*p); v[(bj+1)*p+bk] = 1; v[(bj+1)*p+(bk+1)] = 1
        w = np.zeros(p*n); w[bk*n+ai] = 1
        add_term(u, v, w)

    def add_2x2x3(ai, aj, bj, bk):
        """Add 9 terms: Strassen for first 2 cols + naive for 3rd col."""
        add_strassen_2x2_2(ai, aj, bj, bk)
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[(ai+di)*m+(aj+dj)] = 1
                v = np.zeros(m*p); v[(bj+dj)*p+(bk+2)] = 1
                w = np.zeros(p*n); w[(bk+2)*n+(ai+di)] = 1
                add_term(u, v, w)

    # C00 (2x2) = A00*B00 + A01*B10
    add_strassen_2x2_2(0, 0, 0, 0)
    add_strassen_2x2_2(0, 2, 2, 0)
    
    # C10 (2x2) = A10*B00 + A11*B10
    add_strassen_2x2_2(2, 0, 0, 0)
    add_strassen_2x2_2(2, 2, 2, 0)
    
    # C01 (2x3) = A00*B01 + A01*B11
    add_2x2x3(0, 0, 0, 2)
    add_2x2x3(0, 2, 2, 2)
    
    # C11 (2x3) = A10*B01 + A11*B11
    add_2x2x3(2, 0, 0, 2)
    add_2x2x3(2, 2, 2, 2)

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _als_decompose(n, m, p, rank, seed=42, max_iters=200, deadline=None, init_U=None, init_V=None, init_W=None):
    """ALS decomposition with rounding to half-integers."""
    np.random.seed(seed)
    
    if init_U is not None:
        U = np.array(init_U, dtype=np.float64).copy()
        V = np.array(init_V, dtype=np.float64).copy()
        W = np.array(init_W, dtype=np.float64).copy()
    else:
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Optimize U
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize V
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize W
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (s - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Round to half-integers and verify
    U_r = _round_matrix(best_U)
    V_r = _round_matrix(best_V)
    W_r = _round_matrix(best_W)
    
    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    # Try without rounding
    if _verify(best_U.tolist(), best_V.tolist(), best_W.tolist()):
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline to be safe within 150s limit
    
    # Start with Strassen decomposition (rank 64)
    U_str, V_str, W_str = _strassen_construct(n, m, p)
    rank_str = len(U_str[0])
    
    # Verify Strassen solution
    if _verify(U_str, V_str, W_str):
        best_result = {"U": U_str, "V": V_str, "W": W_str}
        best_rank = rank_str
    else:
        best_result = None
        best_rank = float('inf')
    
    # Try to reduce rank using ALS
    for target_rank in range(60, rank_str + 1):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            break
            
        result = _als_decompose(n, m, p, target_rank, seed=42, max_iters=100, 
                                deadline=deadline, init_U=U_str, init_V=V_str, init_W=W_str)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r):
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                best_rank = target_rank
                break
    
    # If no better solution found, return Strassen
    if best_result is None:
        # Fallback to schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
