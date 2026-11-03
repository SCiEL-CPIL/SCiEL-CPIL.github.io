"""Matrix multiplication 2x4 x 4x7 via Strassen block decomposition + alternating minimization refinement."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]
DEADLINE = 135.0  # safety margin under 150s

def _time_left():
    return DEADLINE - time.time()

def _tensor_from(U, V, W):
    # U: (a, r), V: (b, r), W: (c, r) -> tensor (a,b,c)
    return np.einsum('ar,br,cr->abc', U, V, W)

def _check(U, V, W, T, tol=1e-6):
    R = _tensor_from(U, V, W)
    return np.allclose(R, T, atol=tol, rtol=tol)

def _round_half(M):
    return np.round(M * 2.0) / 2.0

def _get_strassen_222():
    """Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7).
    U: (4, 7) indexed by (i,j) -> i*2+j
    V: (4, 7) indexed by (j,k) -> j*2+k
    W: (4, 7) indexed by (k,i) -> k*2+i
    """
    r = 7
    U = np.zeros((4, r))
    V = np.zeros((4, r))
    W = np.zeros((4, r))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_school_221():
    """Returns U, V, W for 2x2x1 multiplication using schoolbook (Rank 4).
    U: (4, 4), V: (2, 4), W: (2, 4)
    """
    r = 4
    U = np.zeros((4, r))
    V = np.zeros((2, r))
    W = np.zeros((2, r))
    
    for c in range(r):
        i = c // 2
        j = c % 2
        U[i * 2 + j, c] = 1.0
        V[j, c] = 1.0
        W[i, c] = 1.0
        
    return U, V, W

def _build_strassen_decomposition():
    """Build the base Strassen block decomposition for 2x4x7.
    
    Split m=4 into two blocks of 2.
    Split p=7 into: [0,2), [2,4), [4,6) use Strassen 2x2x2; [6,7) uses schoolbook 2x2x1.
    Total rank = 2 * (3*7 + 4) = 2 * 25 = 50.
    """
    n, m, p = 2, 4, 7
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # For each m-block (j_offset in [0, 2)), do the same p-structure
    for j_off in [0, 2]:
        # p in [0, 2): Strassen 2x2x2
        U_sub, V_sub, W_sub = _get_strassen_222()
        r_sub = U_sub.shape[1]
        
        for i in range(2):
            for j in range(2):
                U[i * 4 + (j_off + j), col_offset:col_offset + r_sub] = U_sub[i * 2 + j, :]
        for j in range(2):
            for k in range(2):
                V[(j_off + j) * 7 + k, col_offset:col_offset + r_sub] = V_sub[j * 2 + k, :]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i, col_offset:col_offset + r_sub] = W_sub[k * 2 + i, :]
        col_offset += r_sub
        
        # p in [2, 4): Strassen 2x2x2
        U_sub, V_sub, W_sub = _get_strassen_222()
        r_sub = U_sub.shape[1]
        
        for i in range(2):
            for j in range(2):
                U[i * 4 + (j_off + j), col_offset:col_offset + r_sub] = U_sub[i * 2 + j, :]
        for j in range(2):
            for k in range(2):
                V[(j_off + j) * 7 + (k + 2), col_offset:col_offset + r_sub] = V_sub[j * 2 + k, :]
        for k in range(2):
            for i in range(2):
                W[(k + 2) * 2 + i, col_offset:col_offset + r_sub] = W_sub[k * 2 + i, :]
        col_offset += r_sub
        
        # p in [4, 6): Strassen 2x2x2
        U_sub, V_sub, W_sub = _get_strassen_222()
        r_sub = U_sub.shape[1]
        
        for i in range(2):
            for j in range(2):
                U[i * 4 + (j_off + j), col_offset:col_offset + r_sub] = U_sub[i * 2 + j, :]
        for j in range(2):
            for k in range(2):
                V[(j_off + j) * 7 + (k + 4), col_offset:col_offset + r_sub] = V_sub[j * 2 + k, :]
        for k in range(2):
            for i in range(2):
                W[(k + 4) * 2 + i, col_offset:col_offset + r_sub] = W_sub[k * 2 + i, :]
        col_offset += r_sub
        
        # p in [6, 7): Schoolbook 2x2x1
        U_sub, V_sub, W_sub = _get_school_221()
        r_sub = U_sub.shape[1]
        
        for i in range(2):
            for j in range(2):
                U[i * 4 + (j_off + j), col_offset:col_offset + r_sub] = U_sub[i * 2 + j, :]
        for j in range(2):
            V[(j_off + j) * 7 + 6, col_offset:col_offset + r_sub] = V_sub[j, :]
        for i in range(2):
            W[6 * 2 + i, col_offset:col_offset + r_sub] = W_sub[i, :]
        col_offset += r_sub
    
    return U, V, W

def _build_target_tensor():
    """Build the full target tensor T of shape (8, 28, 14)."""
    n, m, p = 2, 4, 7
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _alt_minimize(U, V, W, T, n_trials=10, max_iter=300, r=None):
    """Alternating minimization to refine a rank-r decomposition.
    Returns (U, V, W) if successful, else None.
    """
    a, b, c = T.shape
    if r is None:
        r = U.shape[1]
    
    for trial in range(n_trials):
        if _time_left() < 3:
            return None
        
        # Initialize: use the provided U,V,W as starting point, or perturb
        if trial == 0:
            U_curr = U.copy()
            V_curr = V.copy()
            W_curr = W.copy()
        else:
            rng = np.random.RandomState(trial * 137 + 42)
            # Perturb the initial solution
            U_curr = U + rng.randn(a, r) * 0.1
            V_curr = V + rng.randn(b, r) * 0.1
            W_curr = W + rng.randn(c, r) * 0.1
        
        prev_err = np.inf
        for it in range(max_iter):
            if _time_left() < 2:
                return None
            
            # Optimize W: T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
            # For fixed U,V: for each c, T[:,:,c] (a x b) = sum_d W[c,d] * outer(U[:,d], V[:,d])
            # Flatten: T_flat (a*b*c,) where index = (c*a + i)*b + j or similar
            # Let's use: for each c, we have a*b equations:
            # T[i,j,c] = sum_d W[c,d] * U[i,d] * V[j,d]
            # Stack over (i,j,c): total a*b*c equations
            # D_w[(i*b+j)*c + c, d] = U[i,d] * V[j,d]  -- no, let's think differently
            # For fixed d, the contribution to T[i,j,c] is U[i,d]*V[j,d]*W[c,d]
            # So for optimizing W: fix U,V. The equation for T[i,j,c] is:
            # T[i,j,c] = sum_d (U[i,d]*V[j,d]) * W[c,d]
            # This is a linear system in W. Build D of shape (a*b*c, r):
            # D[idx, d] where idx = (i*b+j)*c+c_val: D[idx, d] = U[i,d]*V[j,d] if c_val matches...
            # No: for each (i,j,c_val), the coefficient of W[c_val, d] is U[i,d]*V[j,d]
            # So D[(i*b+j)*c + c_val, d] = U[i,d]*V[j,d]
            D_w = np.zeros((a * b * c, r))
            for d in range(r):
                for i in range(a):
                    for j in range(b):
                        for c_val in range(c):
                            D_w[(i * b + j) * c + c_val, d] = U_curr[i, d] * V_curr[j, d]
            
            T_flat_w = T.transpose(2, 0, 1).reshape(c, a * b).T.reshape(a * b * c)
            try:
                W_new = np.linalg.lstsq(D_w, T_flat_w, rcond=None)[0].reshape(c, r)
            except:
                W_new = W_curr
            
            # Optimize U: T[i,j,c] = sum_d U[i,d]*V[j,d]*W[c,d]
            # For fixed V,W: for each i, T[i,j,c] = sum_d U[i,d] * (V[j,d]*W[c,d])
            # D_u[(i*a + i_val)*b*c + (j*c+c_val), d]... let's be careful
            # For fixed i_val: T[i_val, j, c] = sum_d U[i_val, d] * V[j,d] * W[c,d]
            # Stack over (j,c): b*c equations per i_val
            # D_u[(i_val*b*c + j*c + c_val), d] = V[j,d] * W[c_val,d]
            D_u = np.zeros((a * b * c, r))
            for d in range(r):
                for i_val in range(a):
                    for j in range(b):
                        for c_val in range(c):
                            D_u[(i_val * b * c + j * c + c_val), d] = V_curr[j, d] * W_new[c_val, d]
            
            T_flat_u = T.reshape(a, b * c).T.reshape(a * b * c)
            try:
                U_new = np.linalg.lstsq(D_u, T_flat_u, rcond=None)[0].reshape(a, r)
            except:
                U_new = U_curr
            
            # Optimize V: T[i,j,c] = sum_d U[i,d]*V[j,d]*W[c,d]
            # For fixed U,W: for each j, T[i,j,c] = sum_d V[j,d] * (U[i,d]*W[c,d])
            # D_v[(j*b + j_val)*a*c + (i*a+c_val), d] = U[i,d]*W[c_val,d]
            D_v = np.zeros((a * b * c, r))
            for d in range(r):
                for j_val in range(b):
                    for i in range(a):
                        for c_val in range(c):
                            D_v[(j_val * a * c + i * c + c_val), d] = U_new[i, d] * W_new[c_val, d]
            
            T_flat_v = T.transpose(1, 0, 2).reshape(b, a * c).T.reshape(b * a * c)
            # Wait, need to match the indexing. T_flat_v[idx] where idx = (j_val*a*c + i*c + c_val)
            # T[i,j,c] with j fixed: index in T.transpose(1,0,2) is (j, i, c) -> flat index j*a*c + i*c + c
            T_flat_v = T.transpose(1, 0, 2).reshape(b * a * c)
            try:
                V_new = np.linalg.lstsq(D_v, T_flat_v, rcond=None)[0].reshape(b, r)
            except:
                V_new = V_curr
            
            U_curr, V_curr, W_curr = U_new, V_new, W_new
            
            # Compute error
            T_rec = _tensor_from(U_curr, V_curr, W_curr)
            err = np.linalg.norm(T_rec - T)
            if err < 1e-10:
                break
            if abs(prev_err - err) < 1e-14 and it > 20:
                break
            prev_err = err
        
        # Check if we found a valid decomposition
        T_rec = _tensor_from(U_curr, V_curr, W_curr)
        if np.allclose(T_rec, T, atol=1e-8, rtol=1e-8):
            # Try rounding to half-integers
            U_r = _round_half(U_curr)
            V_r = _round_half(V_curr)
            W_r = _round_half(W_curr)
            if np.allclose(_tensor_from(U_r, V_r, W_r), T, atol=1e-8, rtol=1e-8):
                return U_r, V_r, W_r
            return U_curr, V_curr, W_curr
    
    return None

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    # Build target tensor
    T = _build_target_tensor()
    
    # Start with Strassen block decomposition (rank 50)
    U_base, V_base, W_base = _build_strassen_decomposition()
    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = U_base.shape[1]  # 50
    
    # Verify base decomposition
    T_rec = _tensor_from(best_U, best_V, best_W)
    if not np.allclose(T_rec, T, atol=1e-6, rtol=1e-6):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_sb
    
    # Try to reduce rank using alternating minimization
    # Focus on ranks 49 and 48 first (most likely to succeed)
    # Then try lower ranks if time permits
    for r in [49, 48, 47, 46, 45, 44, 43, 42, 41, 40]:
        if r >= best_rank:
            continue
        if _time_left() < 10:
            break
        
        # More trials for closer ranks
        n_trials = 8 if r >= 48 else 5
        max_iter = 300 if r >= 48 else 200
        
        result = _alt_minimize(best_U, best_V, best_W, T, n_trials=n_trials, max_iter=max_iter, r=r)
        if result is not None:
            U_new, V_new, W_new = result
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = r
            # Continue searching for even lower ranks
            continue
    
    # If we still have time, try even lower ranks with more aggressive search
    if _time_left() > 15:
        for r in range(best_rank - 1, max(best_rank - 15, 30), -1):
            if _time_left() < 5:
                break
            if r >= best_rank:
                continue
            
            result = _alt_minimize(best_U, best_V, best_W, T, n_trials=5, max_iter=200, r=r)
            if result is not None:
                U_new, V_new, W_new = result
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = r
                break
    
    # Final verification
    T_rec = _tensor_from(best_U, best_V, best_W)
    if not np.allclose(T_rec, T, atol=1e-6, rtol=1e-6):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
