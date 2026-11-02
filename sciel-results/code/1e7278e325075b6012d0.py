import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE_SECONDS = 120

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that U, V, W exactly decompose the matrix multiplication tensor using einsum."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Compute decomposed tensor
    D = np.einsum('xr,yr,zr->xyz', U_np, V_np, W_np)
    
    # Build expected tensor: T[i*m+j, j*p+k, k*n+i] = 1
    expected = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i * m + j, j * p + k, k * n + i] = 1.0
    
    return np.allclose(D, expected, atol=tol)

def _get_strassen_2x2x2():
    """Returns list of 7 (u, v, w) vectors for 2x2x2 Strassen."""
    terms = []
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    terms.append(([1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]))
    terms.append(([0, 0, 0, 1], [0, 1, 0, 0], [-1, 0, 1, 0]))
    terms.append(([1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]))
    terms.append(([-1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 0, -1]))
    terms.append(([0, 1, 0, -1], [0, 1, 0, 1], [0, 1, 0, -1]))
    return terms

def _get_rank_11_2x2x3():
    """Returns list of 11 (u, v, w) vectors for 2x2x3 with rank 11."""
    terms = []
    s_terms = _get_strassen_2x2x2()
    for u, v, w in s_terms:
        u_g = u
        v_g = [v[0], v[1], 0, v[2], v[3], 0]
        w_g = [w[0], w[1], 0, w[2], w[3], 0]
        terms.append((u_g, v_g, w_g))
    
    terms.append(([1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]))
    terms.append(([0, 1, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]))
    terms.append(([0, 0, 1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]))
    terms.append(([0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0]))
    return terms

def _get_rank_4_1x2x2():
    """Returns list of 4 (u, v, w) vectors for 1x2x2."""
    terms = []
    terms.append(([1, 0], [1, 0, 0, 0], [1, 0]))
    terms.append(([1, 0], [0, 1, 0, 0], [0, 1]))
    terms.append(([0, 1], [0, 0, 1, 0], [1, 0]))
    terms.append(([0, 1], [0, 0, 0, 1], [0, 1]))
    return terms

def _get_rank_6_1x2x3():
    """Returns list of 6 (u, v, w) vectors for 1x2x3."""
    terms = []
    terms.append(([1, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0]))
    terms.append(([1, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0]))
    terms.append(([1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1]))
    terms.append(([0, 1], [0, 0, 0, 1, 0, 0], [1, 0, 0]))
    terms.append(([0, 1], [0, 0, 0, 0, 1, 0], [0, 1, 0]))
    terms.append(([0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 1]))
    return terms

def _try_reduce_rank(U, V, W, n, m, p, max_time=10):
    """Try to reduce the rank by finding linear dependencies using SVD-based approach."""
    start = time.time()
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    R = U_np.shape[1]
    
    # Build the full tensor and try to find a lower-rank decomposition
    # by checking if columns are linearly dependent in a joint sense
    
    # Simple approach: check for zero columns first
    keep = list(range(R))
    
    for r in range(R):
        if time.time() - start > max_time:
            break
        if np.allclose(U_np[:, r], 0, atol=1e-12) or np.allclose(V_np[:, r], 0, atol=1e-12) or np.allclose(W_np[:, r], 0, atol=1e-12):
            keep.remove(r)
    
    if len(keep) < R:
        U_new = U_np[:, keep]
        V_new = V_np[:, keep]
        W_new = W_np[:, keep]
        if _verify_decomposition(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    # Try to find columns that are scalar multiples
    keep = list(range(R))
    for r in range(R):
        if time.time() - start > max_time:
            break
        if r not in keep:
            continue
        for s in keep:
            if s == r:
                continue
            # Check if column r is a scalar multiple of column s
            try:
                # Find a non-zero entry in U to determine ratio
                nz_u = np.nonzero(np.abs(U_np[:, r]) > 1e-12)[0]
                if len(nz_u) == 0:
                    continue
                idx = nz_u[0]
                if np.abs(U_np[idx, s]) > 1e-12:
                    ratio = U_np[idx, r] / U_np[idx, s]
                    if (np.allclose(U_np[:, r], ratio * U_np[:, s], atol=1e-10) and
                        np.allclose(V_np[:, r], ratio * V_np[:, s], atol=1e-10) and
                        np.allclose(W_np[:, r], ratio * W_np[:, s], atol=1e-10)):
                        keep.remove(r)
                        break
            except:
                pass
    
    if len(keep) < R:
        U_new = U_np[:, keep]
        V_new = V_np[:, keep]
        W_new = W_np[:, keep]
        if _verify_decomposition(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(n * m):
            U[i][col] = u_vec[i]
        for i in range(m * p):
            V[i][col] = v_vec[i]
        for i in range(p * n):
            W[i][col] = w_vec[i]
        col += 1
        
    s_terms = _get_strassen_2x2x2()
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    for u, v, w in s_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[0] = u[0]; u_g[1] = u[1]; u_g[4] = u[2]; u_g[5] = u[3]
        v_g[0] = v[0]; v_g[1] = v[1]; v_g[5] = v[2]; v_g[6] = v[3]
        w_g[0] = w[0]; w_g[1] = w[1]; w_g[3] = w[2]; w_g[4] = w[3]
        add_term(u_g, v_g, w_g)
        
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    for u, v, w in s_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[2] = u[0]; u_g[3] = u[1]; u_g[6] = u[2]; u_g[7] = u[3]
        v_g[10] = v[0]; v_g[11] = v[1]; v_g[15] = v[2]; v_g[16] = v[3]
        w_g[0] = w[0]; w_g[1] = w[1]; w_g[3] = w[2]; w_g[4] = w[3]
        add_term(u_g, v_g, w_g)
        
    r11_terms = _get_rank_11_2x2x3()
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    for u, v, w in r11_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[0] = u[0]; u_g[1] = u[1]; u_g[4] = u[2]; u_g[5] = u[3]
        v_g[2] = v[0]; v_g[3] = v[1]; v_g[4] = v[2]; v_g[7] = v[3]; v_g[8] = v[4]; v_g[9] = v[5]
        w_g[2] = w[0]; w_g[3] = w[1]; w_g[4] = w[2]; w_g[5] = w[3]; w_g[6] = w[4]; w_g[7] = w[5]
        add_term(u_g, v_g, w_g)
        
    # Block 4: A[0:2, 2:4] * B[2:4, 2:5] -> C[0:2, 2:5]
    for u, v, w in r11_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[2] = u[0]; u_g[3] = u[1]; u_g[6] = u[2]; u_g[7] = u[3]
        v_g[12] = v[0]; v_g[13] = v[1]; v_g[14] = v[2]; v_g[17] = v[3]; v_g[18] = v[4]; v_g[19] = v[5]
        w_g[2] = w[0]; w_g[3] = w[1]; w_g[4] = w[2]; w_g[5] = w[3]; w_g[6] = w[4]; w_g[7] = w[5]
        add_term(u_g, v_g, w_g)
        
    r4_terms = _get_rank_4_1x2x2()
    
    # Block 5: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    for u, v, w in r4_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[8] = u[0]; u_g[9] = u[1]
        v_g[0] = v[0]; v_g[1] = v[1]; v_g[5] = v[2]; v_g[6] = v[3]
        w_g[6] = w[0]; w_g[7] = w[1]
        add_term(u_g, v_g, w_g)
        
    # Block 6: A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2]
    for u, v, w in r4_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[10] = u[0]; u_g[11] = u[1]
        v_g[10] = v[0]; v_g[11] = v[1]; v_g[15] = v[2]; v_g[16] = v[3]
        w_g[6] = w[0]; w_g[7] = w[1]
        add_term(u_g, v_g, w_g)
        
    r6_terms = _get_rank_6_1x2x3()
    
    # Block 7: A[2, 0:2] * B[0:2, 2:5] -> C[2, 2:5]
    for u, v, w in r6_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[8] = u[0]; u_g[9] = u[1]
        v_g[2] = v[0]; v_g[3] = v[1]; v_g[4] = v[2]; v_g[7] = v[3]; v_g[8] = v[4]; v_g[9] = v[5]
        w_g[8] = w[0]; w_g[9] = w[1]; w_g[10] = w[2]
        add_term(u_g, v_g, w_g)
        
    # Block 8: A[2, 2:4] * B[2:4, 2:5] -> C[2, 2:5]
    for u, v, w in r6_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[10] = u[0]; u_g[11] = u[1]
        v_g[12] = v[0]; v_g[13] = v[1]; v_g[14] = v[2]; v_g[17] = v[3]; v_g[18] = v[4]; v_g[19] = v[5]
        w_g[8] = w[0]; w_g[9] = w[1]; w_g[10] = w[2]
        add_term(u_g, v_g, w_g)
        
    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")
        
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    remaining_time = DEADLINE_SECONDS - (time.time() - start_time)
    if remaining_time > 5:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, max_time=min(remaining_time - 2, 15))
    
    return {"U": U, "V": V, "W": W}
