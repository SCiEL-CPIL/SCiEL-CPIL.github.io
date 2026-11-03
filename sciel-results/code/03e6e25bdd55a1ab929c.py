"""Matrix multiplication 2x4 x 4x7: block decomposition + rank minimization search."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # safety margin under 150s

def _time_left():
    return _DEADLINE - time.time()

def _get_strassen_222():
    """Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7)."""
    r = 7
    U = np.zeros((4, r))
    V = np.zeros((4, r))
    W = np.zeros((4, r))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_223_rank9():
    """Returns U, V, W for 2x2x3 multiplication with rank 9."""
    n, m, p = 2, 2, 3
    r = 9
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    U[0, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[4, 0] = 1
    
    U[3, 1] = 1
    V[3, 1] = 1; V[4, 1] = 1; V[5, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1; W[5, 1] = 1
    
    U[0, 2] = 1; U[3, 2] = 1
    V[0, 2] = 1; V[3, 2] = 1
    W[0, 2] = -1; W[1, 2] = -1
    
    U[0, 3] = 1; U[3, 3] = 1
    V[1, 3] = 1; V[4, 3] = 1
    W[2, 3] = -1; W[3, 3] = -1
    
    U[0, 4] = 1; U[3, 4] = 1
    V[2, 4] = 1; V[5, 4] = 1
    W[4, 4] = -1; W[5, 4] = -1
    
    U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1; V[2, 5] = 1
    W[1, 5] = 1; W[3, 5] = 1; W[5, 5] = 1
    
    U[1, 6] = 1
    V[3, 6] = 1; V[4, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1; W[4, 6] = 1
    
    U[2, 7] = 1; U[1, 7] = 1
    V[0, 7] = 1; V[3, 7] = 1
    W[1, 7] = -1; W[0, 7] = -1
    
    U[2, 8] = 1; U[1, 8] = 1
    V[1, 8] = 1; V[4, 8] = 1
    W[3, 8] = -1; W[2, 8] = -1
    
    return U, V, W

def _get_224_rank14():
    """Returns U, V, W for 2x2x4 multiplication with rank 14 (two Strassen 2x2x2)."""
    n, m, p = 2, 2, 4
    r = 14
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    U_s, V_s, W_s = _get_strassen_222()
    
    for i in range(2):
        for j in range(2):
            U[i*2+j, 0:7] = U_s[i*2+j, :]
    for j in range(2):
        for k in range(2):
            V[j*4+k, 0:7] = V_s[j*2+k, :]
    for k in range(2):
        for i in range(2):
            W[k*2+i, 0:7] = W_s[k*2+i, :]
    
    for i in range(2):
        for j in range(2):
            U[i*2+j, 7:14] = U_s[i*2+j, :]
    for j in range(2):
        for k in range(2):
            V[j*4+(k+2), 7:14] = V_s[j*2+k, :]
    for k in range(2):
        for i in range(2):
            W[(k+2)*2+i, 7:14] = W_s[k*2+i, :]
    
    return U, V, W

def _build_rank46():
    """Build the base rank-46 decomposition for 2x4x7."""
    n, m, p = 2, 4, 7
    total_rank = 46
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    m_blocks = [(0, 2), (2, 4)]
    p_blocks = [
        (0, 3, 'rank9_223'),
        (3, 7, 'rank14_224')
    ]
    
    U_r9, V_r9, W_r9 = _get_223_rank9()
    U_r14, V_r14, W_r14 = _get_224_rank14()
    
    for j_start, j_end in m_blocks:
        for k_start, k_end, p_type in p_blocks:
            if p_type == 'rank9_223':
                U_sub, V_sub, W_sub = U_r9, V_r9, W_r9
                r_sub = 9
                p_sub = 3
            else:
                U_sub, V_sub, W_sub = U_r14, V_r14, W_r14
                r_sub = 14
                p_sub = 4
            
            for i in range(n):
                for j_sub in range(j_end - j_start):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            
            for j_sub in range(j_end - j_start):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            
            col_offset += r_sub
    
    return U, V, W

def _build_rank50():
    """Fallback rank-50 decomposition using Strassen blocks."""
    n, m, p = 2, 4, 7
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    m_blocks = [(0, 2), (2, 4)]
    p_blocks = [
        (0, 2, 'strassen'),
        (2, 4, 'strassen'),
        (4, 6, 'strassen'),
        (6, 7, 'school')
    ]
    
    U_s, V_s, W_s = _get_strassen_222()
    
    r_sc = 4
    U_sc = np.zeros((4, r_sc))
    V_sc = np.zeros((2, r_sc))
    W_sc = np.zeros((2, r_sc))
    for c in range(r_sc):
        i = c // 2
        j = c % 2
        U_sc[i * 2 + j, c] = 1.0
        V_sc[j, c] = 1.0
        W_sc[i, c] = 1.0
    
    for j_start, j_end in m_blocks:
        for k_start, k_end, p_type in p_blocks:
            if p_type == 'strassen':
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
                p_sub = 2
            else:
                U_sub, V_sub, W_sub = U_sc, V_sc, W_sc
                r_sub = 4
                p_sub = 1
            
            for i in range(n):
                for j_sub in range(j_end - j_start):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            
            for j_sub in range(j_end - j_start):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            
            col_offset += r_sub
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor identity: sum_c W[k*n+i,c]*U[i*m+j,c]*V[j'*p+k,c] = delta_{j,j'}"""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True

def _build_target_tensor():
    """Build target tensor T of shape (n*m, m*p, p*n) where T[i*m+j, j*p+k, k*n+i] = 1."""
    n, m, p = 2, 4, 7
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _round_half(M):
    """Round to nearest half-integer."""
    return np.round(M * 2.0) / 2.0

def _alt_minimize(U0, V0, W0, T, r_target, n_trials=3, max_iter=150):
    """
    Alternating minimization to find a rank-r_target decomposition.
    T shape: (a, b, c) where a=n*m, b=m*p, c=p*n
    U shape: (a, r), V shape: (b, r), W shape: (c, r)
    Identity: T[i, j, k] = sum_d U[i,d] * V[j,d] * W[k,d]
    """
    a, b, c = T.shape
    r = r_target
    
    best_result = None
    best_err = np.inf
    
    for trial in range(n_trials):
        if _time_left() < 5:
            break
        
        # Initialize
        if trial == 0:
            U_curr = U0[:, :r].copy() if U0.shape[1] >= r else np.zeros((a, r))
            V_curr = V0[:, :r].copy() if V0.shape[1] >= r else np.zeros((b, r))
            W_curr = W0[:, :r].copy() if W0.shape[1] >= r else np.zeros((c, r))
        else:
            rng = np.random.RandomState(trial * 137 + 42)
            if U0.shape[1] >= r:
                U_curr = U0[:, :r] + rng.randn(a, r) * 0.05
                V_curr = V0[:, :r] + rng.randn(b, r) * 0.05
                W_curr = W0[:, :r] + rng.randn(c, r) * 0.05
            else:
                U_curr = rng.randn(a, r) * 0.1
                V_curr = rng.randn(b, r) * 0.1
                W_curr = rng.randn(c, r) * 0.1
        
        prev_err = np.inf
        for it in range(max_iter):
            if _time_left() < 3:
                break
            
            # Optimize W
            M = np.einsum('id,jd->ijd', U_curr, V_curr).reshape(a * b, r)
            T_rhs = T.transpose(2, 0, 1).reshape(c, a * b).T
            try:
                W_new = np.linalg.lstsq(M, T_rhs, rcond=None)[0].T
            except:
                W_new = W_curr
            
            # Optimize U
            V_W = np.einsum('jd,kd->jkd', V_curr, W_curr)
            M_U = np.tile(V_W.reshape(b * c, r), (a, 1, 1))
            T_lhs = T.reshape(a, b * c)
            
            U_new = np.zeros((a, r))
            for i in range(a):
                try:
                    U_new[i, :] = np.linalg.lstsq(M_U[i, :, :].T, T_lhs[i, :], rcond=None)[0]
                except:
                    U_new[i, :] = U_curr[i, :]
            
            # Optimize V
            U_W = np.einsum('id,kd->ikd', U_curr, W_curr)
            M_V = np.tile(U_W.reshape(a * c, r), (b, 1, 1))
            T_rhs2 = T.transpose(0, 2, 1).reshape(a * c, b).T
            
            V_new = np.zeros((b, r))
            for j in range(b):
                try:
                    V_new[j, :] = np.linalg.lstsq(M_V[j, :, :].T, T_rhs2[j, :], rcond=None)[0]
                except:
                    V_new[j, :] = V_curr[j, :]
            
            U_curr, V_curr, W_curr = U_new, V_new, W_new
            
            # Compute error
            T_recon = np.einsum('id,jd,kd->ijk', U_curr, V_curr, W_curr)
            err = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
            
            if err < best_err:
                best_err = err
                best_result = (U_curr.copy(), V_curr.copy(), W_curr.copy())
            
            if err < 1e-10:
                break
            if abs(err - prev_err) < 1e-14:
                break
            prev_err = err
    
    if best_result is not None:
        U_b, V_b, W_b = best_result
        # Try rounding to half-integers
        U_r = _round_half(U_b)
        V_r = _round_half(V_b)
        W_r = _round_half(W_b)
        
        T_recon_r = np.einsum('id,jd,kd->ijk', U_r, V_r, W_r)
        if np.allclose(T_recon_r, T, atol=1e-6, rtol=1e-6):
            return U_r, V_r, W_r, best_err
        else:
            return U_b, V_b, W_b, best_err
    
    return None

def _sparse_support_reduce(U, V, W, T, n, m, p, max_removals=10):
    """
    Greedily try to remove rank-1 terms from the decomposition.
    For each term d, check if the tensor can still be represented
    without it by solving a linear system on the remaining terms.
    """
    a, b, c = T.shape
    r = U.shape[1]
    
    # Build the linear system: T[i,j,k] = sum_d U[i,d]*V[j,d]*W[k,d]
    # Flatten T to a vector of length a*b*c
    T_flat = T.reshape(-1)
    num_eqs = T_flat.shape[0]
    
    # Active set of terms
    active = list(range(r))
    
    for removal_pass in range(max_removals):
        if _time_left() < 5:
            break
        
        removed_this_pass = False
        for d in list(active):
            if _time_left() < 3:
                break
            
            # Try removing term d
            candidate_active = [x for x in active if x != d]
            if len(candidate_active) == 0:
                continue
            
            # Build design matrix A for the remaining terms
            # A[row, col] where row indexes (i,j,k) and col indexes term index
            # T[i,j,k] = sum_{d' in candidate_active} U[i,d']*V[j,d']*W[k,d']
            r_rem = len(candidate_active)
            
            # Build A efficiently using Kronecker-like structure
            # For each term d', the contribution to T[i,j,k] is U[i,d']*V[j,d']*W[k,d']
            # We need to check if T is in the span of these rank-1 tensors
            
            # Build A as (a*b*c, r_rem)
            # A[i*b*c + j*c + k, t] = U[i, candidate_active[t]] * V[j, candidate_active[t]] * W[k, candidate_active[t]]
            
            # Efficient construction using broadcasting
            # U_sel: (a, r_rem), V_sel: (b, r_rem), W_sel: (c, r_rem)
            U_sel = U[:, candidate_active]  # (a, r_rem)
            V_sel = V[:, candidate_active]  # (b, r_rem)
            W_sel = W[:, candidate_active]  # (c, r_rem)
            
            # A[i, j, k, t] = U[i,t] * V[j,t] * W[k,t]
            # Reshape to (a*b*c, r_rem)
            # A_flat[i*b*c + j*c + k, t] = U[i,t] * V[j,t] * W[k,t]
            
            # Use einsum to build this
            # temp[i,j,k,t] = U[i,t]*V[j,t]*W[k,t]
            # Then reshape to (a*b*c, r_rem)
            
            # More efficient: for each t, the tensor is U[:,t] outer V[:,t] outer W[:,t]
            # So A[:, t] = (U[:,t] * V[:,t] * W[:,t]).reshape(-1)
            # i.e., A[:, t] = np.kron(np.kron(U[:,t], V[:,t]), W[:,t])
            
            A = np.zeros((num_eqs, r_rem))
            for t_idx, d_idx in enumerate(candidate_active):
                A[:, t_idx] = np.kron(np.kron(U[:, d_idx], V[:, d_idx]), W[:, d_idx])
            
            # Solve A * x = T_flat
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A, T_flat, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check if the solution is exact (residual ~ 0)
            residual = np.linalg.norm(A @ x - T_flat)
            if residual < 1e-8:
                # Check if x values are close to half-integers
                x_rounded = np.round(x * 2.0) / 2.0
                if np.allclose(A @ x_rounded, T_flat, atol=1e-6):
                    # We can remove term d and use the new coefficients
                    # Update U, V, W for the remaining terms
                    for t_idx, d_idx in enumerate(candidate_active):
                        U[:, d_idx] = x_rounded[t_idx] * U[:, d_idx] / (U[:, d_idx] != 0).sum() if (U[:, d_idx] != 0).sum() > 0 else 0
                        # Actually, we need to rescale properly
                        # The new decomposition is: T = sum_t x_t * U[:,d_t] * V[:,d_t] * W[:,d_t]
                        # We can absorb x_t into one of the matrices
                        # Let's put it into U
                    # Simpler: just verify and keep track
                    active = candidate_active
                    removed_this_pass = True
                    break
        
        if not removed_this_pass:
            break
    
    if len(active) < r:
        # Build reduced matrices
        U_red = U[:, active]
        V_red = V[:, active]
        W_red = W[:, active]
        return U_red, V_red, W_red, len(active)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Build base rank-46 decomposition
    U46, V46, W46 = _build_rank46()
    
    if _verify(U46, V46, W46, n, m, p):
        base_result = {"U": U46.tolist(), "V": V46.tolist(), "W": W46.tolist()}
    else:
        U50, V50, W50 = _build_rank50()
        base_result = {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}
    
    T = _build_target_tensor()
    
    # Strategy 1: Sparse support pattern search - try to remove terms greedily
    if _time_left() > 30:
        reduced = _sparse_support_reduce(U46.copy(), V46.copy(), W46.copy(), T, n, m, p, max_removals=10)
        if reduced is not None:
            U_r, V_r, W_r, r_new = reduced
            if _verify(U_r, V_r, W_r, n, m, p, tol=1e-6):
                U_check = np.round(U_r * 2) / 2
                V_check = np.round(V_r * 2) / 2
                W_check = np.round(W_r * 2) / 2
                if _verify(U_check, V_check, W_check, n, m, p, tol=1e-6):
                    return {"U": U_check.tolist(), "V": V_check.tolist(), "W": W_check.tolist()}
    
    # Strategy 2: Try alternating minimization for rank reduction
    target_ranks = [45, 44, 43, 42, 41, 40, 39, 38, 37, 36]
    
    for r_target in target_ranks:
        if _time_left() < 25:
            break
        
        result = _alt_minimize(U46, V46, W46, T, r_target, n_trials=3, max_iter=120)
        if result is not None:
            U_t, V_t, W_t, err = result
            if _verify(U_t, V_t, W_t, n, m, p, tol=1e-6):
                U_check = np.round(U_t * 2) / 2
                V_check = np.round(V_t * 2) / 2
                W_check = np.round(W_t * 2) / 2
                if np.allclose(U_check, U_t, atol=1e-9) and \
                   np.allclose(V_check, V_t, atol=1e-9) and \
                   np.allclose(W_check, W_t, atol=1e-9):
                    return {"U": U_check.tolist(), "V": V_check.tolist(), "W": W_check.tolist()}
    
    return base_result
