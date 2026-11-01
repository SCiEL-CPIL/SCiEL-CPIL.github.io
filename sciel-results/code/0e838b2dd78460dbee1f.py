import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=1e-10)

def _als_decompose(T_flat_u, T_flat_v, T_flat_w, rank, max_iters=100, tol=1e-12):
    n_u, n_v, n_w = T_flat_u.shape[0], T_flat_v.shape[0], T_flat_w.shape[0]
    
    # Initialize
    U = np.random.randn(n_u, rank) * 0.1
    V = np.random.randn(n_v, rank) * 0.1
    W = np.random.randn(n_w, rank) * 0.1
    
    prev_err = np.inf
    for it in range(max_iters):
        # Update U
        # M_ZW: rank x (n_v * n_w)
        # M_ZW[r, v*n_w + w] = V[v, r] * W[w, r]
        M_ZW = np.einsum('vr,wr->vwr', V, W).reshape(rank, n_v * n_w)
        U_new = np.linalg.lstsq(M_ZW.T, T_flat_u, rcond=None)[0]
        
        # Update V
        # M_UW: rank x (n_u * n_w)
        # M_UW[r, u*n_w + w] = U[u, r] * W[w, r]
        M_UW = np.einsum('ur,wr->uwr', U, W).reshape(rank, n_u * n_w)
        V_new = np.linalg.lstsq(M_UW.T, T_flat_v, rcond=None)[0]
        
        # Update W
        # M_UV: rank x (n_u * n_v)
        # M_UV[r, u*n_v + v] = U[u, r] * V[v, r]
        M_UV = np.einsum('ur,vr->uvr', U, V).reshape(rank, n_u * n_v)
        W_new = np.linalg.lstsq(M_UV.T, T_flat_w, rcond=None)[0]
        
        U, V, W = U_new, V_new, W_new
        
        # Calculate error
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        # We need the full tensor T to calculate error properly, or use the flat versions
        # But we only have flats. Let's reconstruct the flat error.
        # Error in U-space: || T_flat_u - U M_ZW ||_F
        err_u = np.linalg.norm(T_flat_u - U @ M_ZW)
        err_v = np.linalg.norm(T_flat_v - V @ M_UW)
        err_w = np.linalg.norm(T_flat_w - W @ M_UV)
        err = np.sqrt(err_u**2 + err_v**2 + err_w**2)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100

    # Build T flats
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_flat_u = T.reshape(n*m, m*p * p*n)
    T_flat_v = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
    T_flat_w = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)

    # Try Rank 38
    if time.time() < deadline - 20:
        for attempt in range(20):
            if time.time() > deadline - 20: break
            U, V, W = _als_decompose(T_flat_u, T_flat_v, T_flat_w, 38, max_iters=50)
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _verify(U_r, V_r, W_r):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }

    # Try Rank 37
    if time.time() < deadline - 10:
        for attempt in range(10):
            if time.time() > deadline - 10: break
            U, V, W = _als_decompose(T_flat_u, T_flat_v, T_flat_w, 37, max_iters=50)
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _verify(U_r, V_r, W_r):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }

    # Fallback: Rank 39 Construction
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    r = 0

    def add_strassen_block(k0, k1):
        nonlocal r
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1

        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1

        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1

        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1

        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1

        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1

        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1

        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1

        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1

    for block in range(5):
        if time.time() > deadline - 5:
            break
        k0 = block * 2
        k1 = block * 2 + 1
        add_strassen_block(k0, k1)

    if r < rank:
        k2 = 10
        v_e2 = k2
        v_g2 = 11 + k2
        w_c02 = 2 * k2
        w_c12 = 2 * k2 + 1

        U[0, r] += 1
        V[v_e2, r] += 1
        W[w_c02, r] += 1
        r += 1

        U[1, r] += 1
        V[v_g2, r] += 1
        W[w_c02, r] += 1
        r += 1

        U[2, r] += 1
        V[v_e2, r] += 1
        W[w_c12, r] += 1
        r += 1

        U[3, r] += 1
        V[v_g2, r] += 1
        W[w_c12, r] += 1
        r += 1

    if time.time() < deadline - 2:
        if _verify(U, V, W):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback Schoolbook
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
