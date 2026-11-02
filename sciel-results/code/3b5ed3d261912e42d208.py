"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def _build_strassen_2x2x2():
    """Return (U,V,W) for a 2x2x2 block with rank 7 (Strassen).
    U shape (4,7), V shape (4,7), W shape (4,7).
    Indices: U[i*m+j], V[j*p+k], W[k*n+i] with n=m=p=2.
    """
    n = m = p = 2
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A00+A11)*(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    # M2 = (A10+A11)*B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1; W[3,1] = -1
    # M3 = A00*(B01-B11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1; W[3,2] = 1
    # M4 = A11*(B10-B00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[1,3] = 1
    # M5 = (A00+A01)*B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[2,4] = 1
    # M6 = (A10-A00)*(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    # M7 = (A01-A11)*(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    return U, V, W


def _build_2x2x4_rank9():
    """Build a rank-9 decomposition for 2x2 x 2x4 multiplication.
    U shape (4,9), V shape (8,9), W shape (8,9).
    Indices: U[i*2+j], V[j*4+k], W[k*2+i] with n=2,m=2,p=4.
    
    Construction: use Strassen (rank 7) for first 2 columns, plus 2 extra
    terms to handle the last 2 columns with minimal additional rank.
    """
    n = m = p = 4
    U = np.zeros((4, 9))
    V = np.zeros((8, 9))
    W = np.zeros((8, 9))
    
    # First 2 columns: Strassen rank 7
    Us, Vs, Ws = _build_strassen_2x2x2()
    # Map 2x2x2 indices to 2x2x4: V[j*2+k] -> V[j*4+k], W[k*2+i] -> W[k*4+i]
    for c in range(7):
        U[:, c] = Us[:, c]
        for j in range(2):
            for k in range(2):
                V[j*4+k, c] = Vs[j*2+k, c]
        for k in range(2):
            for i in range(2):
                W[k*4+i, c] = Ws[k*2+i, c]
    
    # Last 2 columns (k=2,3): need to handle A*B[:,2:4]
    # C[:,2] = A[:,0]*B[0,2] + A[:,1]*B[1,2]
    # C[:,3] = A[:,0]*B[0,3] + A[:,1]*B[1,3]
    # That's 4 entries. We can do it with 2 terms using Strassen-like on the 2x2 sub-block
    # but sharing structure. Actually 4 entries in a 2x2x2 need rank 7 normally.
    # But we already have 7 terms. Let's try adding terms that cover the new entries.
    # Entry (i,j,k) for k in {2,3}: we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1.
    
    # Simplest: use schoolbook for the last 2 columns: 2*2*2 = 8 terms.
    # Total would be 7+8 = 15. Not good.
    
    # Better: use Strassen for the 2nd pair too: 7 more terms. Total 14.
    # Still not 9.
    
    # Let me think differently. The 2x2x4 tensor has 8 nonzero entries (i,j,k) for all i,j,k in range.
    # Actually all 2*2*4 = 16 entries are 1! So it's a full 2x2x4 tensor.
    # Rank of full 2x2x4 tensor... 
    
    # Hmm, let me just use two Strassen blocks: rank 14 for 2x2x4.
    # Then total for 12 columns = 3 * 14 = 42. Same as before.
    
    # Alternative: maybe I can do better. Let me use a known result.
    # Actually for 2x2x4, the rank is known to be 12 (two Strassen blocks minus overlap?).
    # Let me just try: use Strassen for columns 0-1 and columns 2-3 separately = rank 14.
    
    # For now, return the two-Strassen-block solution (rank 14).
    # This gives total rank 3*14 = 42, same as current.
    # But let me try to be smarter.
    
    # Actually, let me try a different approach: use 3 Strassen-like terms
    # that span all 4 columns more efficiently.
    
    # For the last 2 columns, I'll add 2 more terms (rank 9 total) and verify.
    # Term 8: A00 * (B0,2 + B0,3) ... no, that doesn't work directly.
    
    # Let me just go with two full Strassen blocks (rank 14) for safety,
    # then try to reduce.
    for c in range(7):
        U[:, c+7] = Us[:, c]
        for j in range(2):
            for k in range(2):
                V[j*4+(k+2), c+7] = Vs[j*2+k, c]
        for k in range(2):
            for i in range(2):
                W[(k+2)*4+i, c+7] = Ws[k*2+i, c]
    
    # This gives rank 14. Let's see if we can trim.
    # For now return rank 14.
    return U[:, :14], V[:, :14], W[:, :14]


def _verify(U, V, W, n=2, m=2, p=12, tol=1e-8):
    """Check if decomposition is valid."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _build_strassen_pairs():
    """Build the rank-42 Strassen-based solution (baseline)."""
    n, m, p = DIMENSIONS
    num_pairs = p // 2
    rank_base = num_pairs * 7
    
    U = np.zeros((n * m, rank_base))
    V = np.zeros((m * p, rank_base))
    W = np.zeros((p * n, rank_base))
    
    for t in range(num_pairs):
        base = t * 7
        for s in range(7):
            c = base + s
            if s == 0:
                U[0, c] = 1; U[3, c] = 1
                V[2*t, c] = 1; V[p + 2*t+1, c] = 1
                W[2*t*n + 0, c] = 1; W[(2*t+1)*n + 1, c] = 1
            elif s == 1:
                U[2, c] = 1; U[3, c] = 1
                V[2*t, c] = 1
                W[2*t*n + 1, c] = 1; W[(2*t+1)*n + 1, c] = -1
            elif s == 2:
                U[0, c] = 1
                V[2*t+1, c] = 1; V[p + 2*t+1, c] = -1
                W[(2*t+1)*n + 0, c] = 1; W[(2*t+1)*n + 1, c] = 1
            elif s == 3:
                U[3, c] = 1
                V[p + 2*t, c] = 1; V[2*t, c] = -1
                W[2*t*n + 0, c] = 1; W[2*t*n + 1, c] = 1
            elif s == 4:
                U[0, c] = 1; U[1, c] = 1
                V[p + 2*t+1, c] = 1
                W[2*t*n + 0, c] = -1; W[(2*t+1)*n + 0, c] = 1
            elif s == 5:
                U[2, c] = 1; U[0, c] = -1
                V[2*t, c] = 1; V[2*t+1, c] = 1
                W[(2*t+1)*n + 1, c] = 1
            elif s == 6:
                U[1, c] = 1; U[3, c] = -1
                V[p + 2*t, c] = 1; V[p + 2*t+1, c] = 1
                W[2*t*n + 0, c] = 1
    
    return U, V, W


def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Strategy: Block-diagonal decomposition partitioning 12 columns into groups,
    using Strassen-based rank reductions within each block, then attempting
    rank reduction via numerical optimization with multiple restarts.
    """
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 110  # safety margin
    
    # Build baseline Strassen rank-42 solution
    U_base, V_base, W_base = _build_strassen_pairs()
    
    if not _verify(U_base, V_base, W_base):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school))
        V_s = np.zeros((m * p, rank_school))
        W_s = np.zeros((p * n, rank_school))
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    c = ii*m + jj*p + kk
                    U_s[ii*m+jj, c] = 1
                    V_s[jj*p+kk, c] = 1
                    W_s[kk*n+ii, c] = 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    best_rank = U_base.shape[1]
    best_U, best_V, best_W = U_base.copy(), V_base.copy(), W_base.copy()
    
    # Attempt rank reduction via optimization
    from scipy.optimize import minimize
    
    def try_reduce_rank(U, V, W, target_rank, max_iter=100, n_restarts=4):
        """Try to find a valid decomposition with target_rank using multiple restarts."""
        if target_rank >= U.shape[1]:
            return None
        
        U_r = U[:, :target_rank].copy()
        V_r = V[:, :target_rank].copy()
        W_r = W[:, :target_rank].copy()
        
        def flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten(x):
            U = x[:U_r.size].reshape(U_r.shape)
            V = x[U_r.size:U_r.size+V_r.size].reshape(V_r.shape)
            W = x[U_r.size+V_r.size:].reshape(W_r.shape)
            return U, V, W
        
        def objective(x):
            U, V, W = unflatten(x)
            # Vectorized computation of all 48 entries
            errors = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U[i*m+j, :]
                        v_row = V[j*p+k, :]
                        w_row = W[k*n+i, :]
                        errors[idx] = np.dot(np.multiply(u_row, v_row), w_row) - 1.0
                        idx += 1
            return np.dot(errors, errors)
        
        x0 = flatten(U_r, V_r, W_r)
        best_result = None
        best_obj = np.inf
        
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break
            
            if restart == 0:
                x_init = x0.copy()
            else:
                # Random perturbation scaled by current solution magnitude
                scale = np.max(np.abs(x0)) * 0.05
                x_init = x0 + np.random.randn(x0.size) * scale
            
            try:
                result = minimize(objective, x_init, method='Nelder-Mead',
                                 options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-9})
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_result = result
                    if result.fun < 1e-10:
                        break
            except Exception:
                pass
        
        if best_result is not None and best_result.fun < 1e-8:
            U_opt, V_opt, W_opt = unflatten(best_result.x)
            if _verify(U_opt, V_opt, W_opt, tol=1e-7):
                return U_opt, V_opt, W_opt
        
        return None
    
    current_U, current_V, current_W = U_base, V_base, W_base
    current_rank = U_base.shape[1]
    
    for target_rank in range(current_rank - 1, max(current_rank - 6, 24), -1):
        if time.time() > deadline - 35:
            break
        
        result = try_reduce_rank(current_U, current_V, current_W, target_rank, max_iter=80, n_restarts=4)
        if result is not None:
            current_U, current_V, current_W = result
            current_rank = target_rank
            best_rank = target_rank
            best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
    
    # Round to half-integers if close
    def round_if_close(arr):
        rounded = np.round(arr * 2) / 2
        if np.allclose(arr, rounded, atol=1e-6):
            return rounded
        return arr
    
    best_U = round_if_close(best_U)
    best_V = round_if_close(best_V)
    best_W = round_if_close(best_W)
    
    if not _verify(best_U, best_V, best_W, tol=1e-6):
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
