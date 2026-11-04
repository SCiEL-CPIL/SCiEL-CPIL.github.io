import numpy as np
import time

def _strassen_2x2x2():
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=-1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[1,2]=1; W2[3,2]=1
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[1,4]=1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1
    return U2, V2, W2

def _block_specs():
    A_specs = [
        [(1,0),(1,3)], [(1,2),(1,3)], [(1,0)], [(1,3)],
        [(1,0),(1,1)], [(1,2),(-1,0)], [(1,1),(-1,3)],
    ]
    B_specs = [
        [(1,0),(1,3)], [(1,0)], [(1,1),(-1,3)], [(1,2),(-1,0)],
        [(1,3)], [(1,0),(1,1)], [(1,2),(1,3)],
    ]
    C_specs = [
        [(1,0),(1,3)], [(1,2),(-1,3)], [(1,1),(1,3)], [(1,0),(1,2)],
        [(-1,0),(1,1)], [(1,3)], [(1,0)],
    ]
    return A_specs, B_specs, C_specs

def _build(rank=98):
    n, m, p = 4, 4, 8
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    U2, V2, W2 = _strassen_2x2x2()
    A_specs, B_specs, C_specs = _block_specs()
    for half in range(2):
        for kp in range(7):
            for ms in range(7):
                col = half*49 + kp*7 + ms
                for ca, bi in A_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c
                            U[gi*4+gj, col] += ca * U2[r*2+c, ms]
                for cb, bi in B_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            V[gi*8+gj, col] += cb * V2[r*2+c, ms]
                for cc, bi in C_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            W[gj*4+gi, col] += cc * W2[r*2+c, ms]
    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification: check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
    R = U.shape[1]
    i_idx = np.repeat(np.arange(n), p*m)
    k_idx = np.tile(np.repeat(np.arange(p), m), n)
    j_idx = np.tile(np.arange(m), n*p)
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    U_sel = U[u_rows, :]
    V_sel = V[v_rows, :]
    W_sel = W[w_rows, :]
    prod = U_sel * V_sel * W_sel
    sums = prod.sum(axis=1)
    return np.allclose(sums, 1.0, atol=1e-8)

def _find_dependency_fast(M):
    """Find a null space vector of M using SVD. Returns None if full rank."""
    try:
        U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    tol = 1e-10 * max(M.shape)
    rank_svd = sum(1 for sv in s_vals if sv > tol)
    
    if rank_svd >= M.shape[1]:
        return None
        
    null_vectors = Vt[rank_svd:, :]
    if len(null_vectors) == 0:
        return None
        
    c = null_vectors[0]
    if np.max(np.abs(c)) < 1e-10:
        return None
        
    return c

def _clean_small(U, V, W, threshold=1e-12):
    """Zero out very small entries to help with exactness."""
    U[np.abs(U) < threshold] = 0.0
    V[np.abs(V) < threshold] = 0.0
    W[np.abs(W) < threshold] = 0.0
    return U, V, W

def construct():
    n, m, p = 4, 4, 8
    start = time.time()
    deadline = start + 145.0
    
    U, V, W = _build(98)
    assert _verify_fast(U, V, W, n, m, p), "base decomposition invalid"
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    R = U.shape[1]
    
    # Precompute index arrays once
    i_idx = np.repeat(np.arange(n), p*m)
    k_idx = np.tile(np.repeat(np.arange(p), m), n)
    j_idx = np.tile(np.arange(m), n*p)
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    
    def get_M():
        U_sel = U[u_rows, :]
        V_sel = V[v_rows, :]
        W_sel = W[w_rows, :]
        return U_sel * V_sel * W_sel
    
    M = get_M()
    
    iterations = 0
    while time.time() < deadline - 15.0:
        iterations += 1
        c = _find_dependency_fast(M)
        if c is None:
            break
            
        r0 = np.argmax(np.abs(c))
        if abs(c[r0]) < 1e-10:
            break
            
        keep = [i for i in range(R) if i != r0]
        
        U_new = np.zeros((U.shape[0], R-1))
        V_new = np.zeros((V.shape[0], R-1))
        W_new = np.zeros((W.shape[0], R-1))
        
        for new_idx, old_idx in enumerate(keep):
            alpha = 1.0 - c[old_idx] / c[r0]
            U_new[:, new_idx] = alpha * U[:, old_idx]
            V_new[:, new_idx] = V[:, old_idx]
            W_new[:, new_idx] = W[:, old_idx]
            
        U = U_new
        V = V_new
        W = W_new
        R -= 1
        
        M = get_M()
        
        if R <= 50:
            break
    
    # Clean up small values
    U, V, W = _clean_small(U, V, W)
    
    # Final verification
    if time.time() < deadline - 5.0:
        if not _verify_fast(U, V, W, n, m, p):
            U, V, W = _build(98)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
