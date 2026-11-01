"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 140  # 140s safety margin (evaluator kills at 150s)

def _build_full_tensor(n, m, p):
    """Build the full matrix multiplication tensor."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    return T

def _build_sub_tensor(n, m, p_sub):
    """Build the sub-tensor for a block of p_sub columns.
    Here p_sub=2.
    Returns T_sub of shape (n*m, m*p_sub, p_sub*n).
    """
    nm = n * m
    mp_sub = m * p_sub
    pn_sub = p_sub * n
    T_sub = np.zeros((nm, mp_sub, pn_sub))
    for i in range(n):
        for j in range(m):
            for k in range(p_sub):
                a = i * m + j
                b = j * p_sub + k
                d = k * n + i
                T_sub[a, b, d] = 1.0
    return T_sub

def _check_sub_decomposition(U, V, W, T_sub, tol=1e-8):
    """Check if U, V, W exactly decompose T_sub."""
    T_decomp = np.einsum('at,bt,dt->abd', U, V, W, optimize=True)
    return np.allclose(T_decomp, T_sub, atol=tol, rtol=0)

def _als_sub(T_sub, r, max_iter=1000, time_budget=2.0, seed=42):
    """Run ALS on the sub-tensor to find rank-r decomposition.
    Returns (U, V, W) or None.
    """
    nm, mp, pn = T_sub.shape
    
    # Precompute slices for efficient lstsq
    T_w = T_sub.transpose(2, 0, 1).reshape(pn, nm * mp)
    T_u = T_sub.transpose(0, 1, 2).reshape(nm, mp * pn)
    T_v = T_sub.transpose(1, 0, 2).reshape(mp, nm * pn)
    
    start = time.time()
    np.random.seed(seed)
    
    U = np.random.randn(nm, r) * 0.5
    V = np.random.randn(mp, r) * 0.5
    W = np.random.randn(pn, r) * 0.5
    
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            return None
        
        # Update W
        M_w = U @ V.T
        W_new = np.zeros((pn, r))
        for d in range(pn):
            try:
                W_new[d, :], _, _, _ = np.linalg.lstsq(M_w, T_w[d, :], rcond=None)
            except:
                return None
        W = W_new
        
        # Update U
        M_u = V @ W.T
        U_new = np.zeros((nm, r))
        for a in range(nm):
            try:
                U_new[a, :], _, _, _ = np.linalg.lstsq(M_u, T_u[a, :], rcond=None)
            except:
                return None
        U = U_new
        
        # Update V
        M_v = U @ W.T
        V_new = np.zeros((mp, r))
        for b in range(mp):
            try:
                V_new[b, :], _, _, _ = np.linalg.lstsq(M_v, T_v[b, :], rcond=None)
            except:
                return None
        V = V_new
        
        if _check_sub_decomposition(U, V, W, T_sub, tol=1e-8):
            return U, V, W
            
    return None

def _construct_full_from_sub(U_sub, V_sub, W_sub, n, m, p, r_sub):
    """Construct the full decomposition from the sub-decomposition."""
    p_sub = 2
    num_blocks = p // p_sub  # 4 blocks
    total_rank = num_blocks * r_sub
    
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    for t in range(num_blocks):
        U_full[:, t*r_sub : (t+1)*r_sub] = U_sub
        
        for j in range(m):
            for k_prime in range(p_sub):
                b_global = j * p + (t * p_sub + k_prime)
                b_local = j * p_sub + k_prime
                V_full[b_global, t*r_sub : (t+1)*r_sub] = V_sub[b_local]
                
        for k_prime in range(p_sub):
            for i in range(n):
                d_global = (t * p_sub + k_prime) * n + i
                d_local = k_prime * n + i
                W_full[d_global, t*r_sub : (t+1)*r_sub] = W_sub[d_local]
                
    return U_full, V_full, W_full

def _check_full_decomposition(U, V, W, T_full, tol=1e-8):
    T_decomp = np.einsum('at,bt,dt->abd', U, V, W, optimize=True)
    return np.allclose(T_decomp, T_full, atol=tol, rtol=0)

def _build_schoolbook_fallback(n, m, p):
    """Build a rank n*m*p schoolbook decomposition as fallback."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    T_full = _build_full_tensor(n, m, p)
    
    p_sub = 2
    T_sub = _build_sub_tensor(n, m, p_sub)
    
    # Fallback: Schoolbook rank
    U_fb, V_fb, W_fb = _build_schoolbook_fallback(n, m, p)
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = n * m * p
    
    # Search for sub-rank from 7 up to 12 (schoolbook for sub is 12)
    for r_sub in range(7, 13):
        if time.time() > _DEADLINE - 10:
            break
            
        found = False
        # Try multiple random restarts
        for attempt in range(100):
            if time.time() > _DEADLINE - 5:
                break
                
            seed = 42 + attempt
            result = _als_sub(T_sub, r_sub, max_iter=500, time_budget=1.5, seed=seed)
            if result is not None:
                U_sub, V_sub, W_sub = result
                U_full, V_full, W_full = _construct_full_from_sub(U_sub, V_sub, W_sub, n, m, p, r_sub)
                if _check_full_decomposition(U_full, V_full, W_full, T_full, tol=1e-8):
                    best_U, best_V, best_W = U_full, V_full, W_full
                    best_rank = 4 * r_sub
                    found = True
                    break
        
        if found and r_sub == 7:
            break
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
