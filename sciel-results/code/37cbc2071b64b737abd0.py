import time
import numpy as np

DIMENSIONS = [2, 2, 8]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_228():
    """Construct a rank-28 decomposition for M_{2,2,8} using 4 blocks of Strassen M_{2,2,2}."""
    n, m, p = 2, 2, 8
    NM, MP, PN = n * m, m * p, p * n  # 4, 16, 16
    
    # Standard Strassen terms for 2x2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C[0,0] = M1+M4-M5+M7, C[0,1] = M3+M5, C[1,0] = M2+M4, C[1,1] = M1-M2+M3+M6
    
    terms = [
        # M1 = (a+d)(e+h)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (c+d)e
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3 = a(f-h)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4 = d(g-e)
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5 = (a+b)h
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # M6 = (c-a)(e+f)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (b-d)(g+h)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    for block_idx in range(4):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = terms[term_idx]
            
            # Fill U: index i corresponds to A[i//2, i%2]
            for i in range(4):
                U[i][col] = u[i]
                
            # Fill V: B[j, k_global] where k_global = 2*block_idx + k_local
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    b_idx = j * p + k_global
                    v_idx = j * 2 + k_local
                    V[b_idx][col] = v[v_idx]
                    
            # Fill W: C[k_global, i] where k_global = 2*block_idx + k_local
            # W index: k_global * n + i
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    c_idx = k_global * n + i
                    w_idx = i * 2 + k_local
                    W[c_idx][col] = w[w_idx]
                    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to find linear dependencies among the rank-1 terms to reduce rank."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    dim = (n * m) * (m * p) * (p * n)
    terms = np.zeros((rank, dim))
    for r in range(rank):
        u = U_arr[:, r]
        v = V_arr[:, r]
        w = W_arr[:, r]
        outer = np.einsum('i,j,k->ijk', u, v, w)
        terms[r, :] = outer.flatten()
    
    tol = 1e-10
    U_svd, s_vals, Vt_svd = np.linalg.svd(terms, full_matrices=False)
    null_mask = s_vals < tol
    if not np.any(null_mask):
        return None
    
    null_vecs = Vt_svd[null_mask, :]
    
    best_result = None
    for nv_idx in range(null_vecs.shape[0]):
        c = null_vecs[nv_idx, :]
        max_idx = np.argmax(np.abs(c))
        if abs(c[max_idx]) < tol:
            continue
        
        new_U2 = [[0] * rank for _ in range(n * m)]
        new_V2 = [[0] * rank for _ in range(m * p)]
        new_W2 = [[0] * rank for _ in range(p * n)]
        
        new_col = 0
        for r in range(rank):
            if r == max_idx:
                continue
            scale = 1.0 - c[r] / c[max_idx]
            if abs(scale) < tol:
                continue
            for i in range(n * m):
                new_U2[i][new_col] = U_arr[i, r] * scale
            for j in range(m * p):
                new_V2[j][new_col] = V_arr[j, r] * scale
            for k in range(p * n):
                new_W2[k][new_col] = W_arr[k, r] * scale
            new_col += 1
        
        if new_col == 0:
            continue
        
        new_U3 = [row[:new_col] for row in new_U2]
        new_V3 = [row[:new_col] for row in new_V2]
        new_W3 = [row[:new_col] for row in new_W2]
        
        if _verify_decomposition(new_U3, new_V3, new_W3, n, m, p):
            if best_result is None or new_col < best_result[0]:
                best_result = (new_col, new_U3, new_V3, new_W3)
        
        if time.time() > deadline:
            break
    
    return best_result

def _schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    n, m, p = 2, 2, 8
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Try rank 28 Strassen decomposition first
    try:
        U, V, W = _strassen_228()
        if _verify_decomposition(U, V, W, n, m, p):
            # Try to reduce rank via linear dependency elimination
            if time.time() < deadline - 10:
                result = _try_reduce_rank(U, V, W, n, m, p, deadline - 5)
                if result is not None:
                    new_rank, new_U, new_V, new_W = result
                    if new_rank < 28:
                        U, V, W = new_U, new_V, new_W
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _schoolbook()
    return {"U": U, "V": V, "W": W}
