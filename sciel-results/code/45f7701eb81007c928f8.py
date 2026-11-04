"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 5, 6]

def _construct_target_tensor(n, m, p):
    """Construct the target multiplication tensor T."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_fast(U, V, W, T, tol=1e-8):
    """
    Verify if U, V, W decompose T.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(recon, T, atol=tol)

def _try_rational_snapping(U, V, W, T, max_denom=8):
    """
    Try to snap the float solution to rational numbers with small denominators.
    Returns True if a snapped version verifies exactly, and updates U, V, W in place.
    """
    for denom in [1, 2, 4, 8]:
        step = 1.0 / denom
        U_snap = np.round(U / step) * step
        V_snap = np.round(V / step) * step
        W_snap = np.round(W / step) * step
        
        if _verify_fast(U_snap, V_snap, W_snap, T, tol=1e-9):
            U[:] = U_snap
            V[:] = V_snap
            W[:] = W_snap
            return True
            
    return False

def _build_sparsity_pattern(n, m, p, rank, seed):
    """
    Generate a sparse sparsity pattern for the rank-rank decomposition.
    Each rank term r gets a small set of nonzero positions in U, V, W.
    Returns (U_pattern, V_pattern, W_pattern) as binary matrices.
    """
    np.random.seed(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    U_pat = np.zeros((nm, rank), dtype=bool)
    V_pat = np.zeros((mp, rank), dtype=bool)
    W_pat = np.zeros((pn, rank), dtype=bool)
    
    for r in range(rank):
        # Each term should touch a small number of entries
        # For Strassen-like structure, each term touches ~2-3 entries per matrix
        n_u = np.random.randint(1, 4)
        n_v = np.random.randint(1, 4)
        n_w = np.random.randint(1, 4)
        
        u_idx = np.random.choice(nm, size=n_u, replace=False)
        v_idx = np.random.choice(mp, size=n_v, replace=False)
        w_idx = np.random.choice(pn, size=n_w, replace=False)
        
        U_pat[u_idx, r] = True
        V_pat[v_idx, r] = True
        W_pat[w_idx, r] = True
    
    return U_pat, V_pat, W_pat

def _solve_sparse_linear_system(T, U_pat, V_pat, W_pat, n, m, p):
    """
    Given sparsity patterns, solve for the values that make the decomposition work.
    T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    
    This is a nonlinear system, but if we fix the patterns, we can try to solve
    by treating it as a system of equations.
    
    Approach: For each (a,b,c) with T[a,b,c]=1, we need sum_r U[a,r]*V[b,r]*W[c,r] = 1
    For each (a,b,c) with T[a,b,c]=0, we need sum_r U[a,r]*V[b,r]*W[c,r] = 0
    
    This is still nonlinear. Instead, let's use a different approach:
    Fix W to simple values (1 or -1) where nonzero, then solve for U and V.
    """
    nm, mp, pn = n*m, m*p, p*n
    rank = U_pat.shape[1]
    
    # Initialize W with ±1 where nonzero
    W = np.zeros((pn, rank))
    for c in range(pn):
        for r in range(rank):
            if W_pat[c, r]:
                W[c, r] = np.random.choice([-1.0, 1.0])
    
    # Now we have: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # For fixed W, this is linear in U and V jointly but bilinear.
    # Let's fix V too and solve for U, then iterate.
    
    # Initialize V with ±1 where nonzero
    V = np.zeros((mp, rank))
    for b in range(mp):
        for r in range(rank):
            if V_pat[b, r]:
                V[b, r] = np.random.choice([-1.0, 1.0])
    
    # Solve for U: T[a,b,c] = sum_r U[a,r] * (V[b,r]*W[c,r])
    # For each a: T[a,:,:] = U[a,:] @ M_a where M_a[b,c] = sum_r V[b,r]*W[c,r]
    # But M_a doesn't depend on a, so M = V @ W^T (mp x pn)
    # Then T[a,:,:] flattened = U[a,:] @ M_flattened
    # So we solve: T_flat[a, :] = U[a, :] @ M_vec where M_vec is mp*pn vector
    
    M = V @ W.T  # mp x pn
    M_vec = M.flatten()  # mp*pn
    
    T_flat = T.reshape(nm, mp*pn)
    
    # U[a, :] @ M_vec = T_flat[a, :]
    # This is: sum_b,c U[a,b,c_index] * M_vec[b,c_index] = T_flat[a, :]
    # But U only has nonzeros at U_pat positions.
    
    # For each a, solve: U[a, :] restricted to nonzero pattern = T_flat[a, :] / M_vec
    # This is underdetermined. Let's use least squares with the pattern constraint.
    
    U = np.zeros((nm, rank))
    for a in range(nm):
        nz = np.where(U_pat[a, :])[0]
        if len(nz) == 0:
            continue
        # U[a, nz] @ M_vec[nz] = T_flat[a, :] -- but this doesn't work directly
        # Actually: sum_r U[a,r]*V[b,r]*W[c,r] for fixed a,b,c
        # Let me reformulate.
        
        # For fixed a: sum_{b,c} T[a,b,c] * (some function) ... 
        # Better: for each (b,c) pair, T[a,b,c] = sum_r U[a,r] * (V[b,r]*W[c,r])
        # So for fixed a, we have pn*mp equations: T[a,b,c] = sum_r U[a,r] * Q[b,c,r]
        # where Q[b,c,r] = V[b,r]*W[c,r]
        
        # Stack: for fixed a, the system is:
        # [Q[0,0,0], Q[0,0,1], ..., Q[0,0,r-1], Q[0,1,0], ..., Q[mp-1,pn-1,r-1]] @ U[a,:]^T = T[a,:,:].flatten()
        # But U[a,:] only has nonzeros at nz positions.
        
        Q = (V[:, :, None] * W[None, :, :]).reshape(mp*pn, rank)  # (mp*pn, rank)
        rhs = T[a, :, :].flatten()
        
        # Solve: Q @ U[a,:]^T = rhs, with U[a,r]=0 for r not in nz
        Q_nz = Q[:, nz]
        # Least squares
        try:
            sol, _, _, _ = np.linalg.lstsq(Q_nz, rhs, rcond=None)
            U[a, nz] = sol
        except np.linalg.LinAlgError:
            pass
    
    return U, V, W

def _verify_with_patterns(T, U, V, W, U_pat, V_pat, W_pat, n, m, p):
    """Verify that the solution respects the patterns and is correct."""
    nm, mp, pn = n*m, m*p, p*n
    rank = U.shape[1]
    
    # Check patterns are respected
    for r in range(rank):
        for a in range(nm):
            if not U_pat[a, r] and abs(U[a, r]) > 1e-10:
                return False
        for b in range(mp):
            if not V_pat[b, r] and abs(V[b, r]) > 1e-10:
                return False
        for c in range(pn):
            if not W_pat[c, r] and abs(W[c, r]) > 1e-10:
                return False
    
    return _verify_fast(U, V, W, T, tol=1e-8)

def _strassen_2x2_decomposition():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    where indices are for 2x2 matrices.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A11(B01-B10)
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = -1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_strassen_based_decomposition(n, m, p):
    """
    Build a Strassen-based decomposition for 2x5x6.
    Split middle dimension into 2+2+1, output columns into 2+2+2.
    For each (mid_pair, col_pair) use rank-7 Strassen.
    Leftover middle index uses direct terms.
    """
    n, m, p = 2, 5, 6
    nm, mp, pn = n*m, m*p, p*n
    
    # 2+2+1 split of middle, 2+2+2 split of columns
    mid_groups = [(0, 1), (2, 3)]
    leftover_mid = [4]
    col_pairs = [(0, 1), (2, 3), (4, 5)]
    
    U = []
    V = []
    W = []
    
    for j1, j2 in mid_groups:
        for k1, k2 in col_pairs:
            # Strassen 7 terms for A[:, j1:j2] @ B[j1:j2, k1:k2]
            # A entries: A[0,j1], A[0,j2], A[1,j1], A[1,j2] -> indices j1, j2, m+j1, m+j2
            # B entries: B[j1,k1], B[j1,k2], B[j2,k1], B[j2,k2] -> indices j1*p+k1, j1*p+k2, j2*p+k1, j2*p+k2
            # C entries: C[0,k1], C[0,k2], C[1,k1], C[1,k2] -> indices k1, k2, n+k1, n+k2
            
            # M1 = (A0j1+A1j2)(Bj1k1+Bj2k2)
            u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
            u[j1] = 1; u[m+j2] = 1
            v[j1*p+k1] = 1; v[j2*p+k2] = 1
            w[k1] = 1; w[n+k2] = -1
            U.append(u); V.append(v); W.append(w)
            
            # M2 = (A1j1+A1j2)Bj1k1
            u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
            u[m+j1] = 1; u[m+j2] = 1
            v[j1*p+k1] = 1
            w[n+k1] = 1
            U.append(u); V.append(v); W.append(w)
            
            # M3 = A0j1(Bj2k1-Bj2k2)
            u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
            u[j1] = 1
            v[j2*p+k1] = 1; v[j2*p+k2] = -1
            w[k2] = 1
            U.append(u); V.append(v); W.append(w)
            
            # M4 = A1j2(Bj1k2-Bj2k1)
            u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
            u[m+j2] = 1
            v[j1*p+k2] = 1; v[j2*p+k1] = -1
            w[k1] = -1; w[n+k1] = 1
            U.append(u); V.append(v); W.append(w)
            
            # M5 = (A0j1+A0j2)Bj2k2
            u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
            u[j1] = 1; u[j2] = 1
            v[j2*p+k2] = 1
            w[k1] = -1; w[k2] = 1
            U.append(u); V.append(v); W.append(w)
            
            # M6 = (A1j1-A0j1)(Bj1k1+Bj1k2)
            u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
            u[m+j1] = 1; u[j1] = -1
            v[j1*p+k1] = 1; v[j1*p+k2] = 1
            w[n+k2] = -1
            U.append(u); V.append(v); W.append(w)
            
            # M7 = (A0j2-A1j2)(Bj2k1+Bj2k2)
            u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
            u[j2] = 1; u[m+j2] = -1
            v[j2*p+k1] = 1; v[j2*p+k2] = 1
            w[k1] = 1; w[n+k2] = 1
            U.append(u); V.append(v); W.append(w)
    
    # Leftover middle index j=4: direct terms
    for j in leftover_mid:
        for k in range(p):
            for i in range(n):
                u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                U.append(u); V.append(v); W.append(w)
    
    U = np.array(U).T  # (nm, rank)
    V = np.array(V).T  # (mp, rank)
    W = np.array(W).T  # (pn, rank)
    
    return U, V, W

def _als_polish(U, V, W, T, n, m, p, iterations=20):
    """
    Polish the solution using Alternating Least Squares.
    """
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    for _ in range(iterations):
        T_mat = T.reshape(nm, mp*pn)
        V_mat = np.zeros((mp*pn, r))
        for b in range(mp):
            for c in range(pn):
                V_mat[b*pn+c, :] = V[b, :] * W[c, :]
        try:
            VtV = V_mat.T @ V_mat
            VtV += 1e-6 * np.eye(r)
            U = T_mat @ V_mat @ np.linalg.inv(VtV)
        except np.linalg.LinAlgError:
            pass
            
        T_mat_v = T.reshape(mp, nm*pn)
        U_mat = np.zeros((nm*pn, r))
        for a in range(nm):
            for c in range(pn):
                U_mat[a*pn+c, :] = U[a, :] * W[c, :]
        try:
            UtU = U_mat.T @ U_mat
            UtU += 1e-6 * np.eye(r)
            V = T_mat_v @ U_mat @ np.linalg.inv(UtU)
        except np.linalg.LinAlgError:
            pass
            
        T_mat_w = T.reshape(pn, nm*mp)
        UW_mat = np.zeros((nm*mp, r))
        for a in range(nm):
            for b in range(mp):
                UW_mat[a*mp+b, :] = U[a, :] * V[b, :]
        try:
            UWtUW = UW_mat.T @ UW_mat
            UWtUW += 1e-6 * np.eye(r)
            W = T_mat_w @ UW_mat @ np.linalg.inv(UWtUW)
        except np.linalg.LinAlgError:
            pass

    return U, V, W

def _gradient_descent_with_restart(U, V, W, T, n, m, p, rank, deadline, max_steps=500):
    """
    Run gradient descent with adaptive learning rate and early stopping.
    Returns (found, U, V, W) where found is True if a valid decomposition was found.
    """
    nm, mp, pn = n*m, m*p, p*n
    
    v = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
    m_ = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
    beta1, beta2 = 0.9, 0.999
    lr = 0.02
    t = 0
    best_loss = np.inf
    found = False
    
    for step in range(max_steps):
        if time.time() > deadline - 5:
            break
            
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T - recon
        loss = np.sum(R**2)
        
        if loss < best_loss:
            best_loss = loss
            if loss < 1e-10:
                found = True
                break
                
        gU = -np.einsum('abc,br,cr->ar', R, V, W)
        gV = -np.einsum('abc,ar,cr->br', R, U, W)
        gW = -np.einsum('abc,ar,br->cr', R, U, V)
        
        grads = [gU, gV, gW]
        params = [U, V, W]
        
        t += 1
        for i in range(3):
            m_[i] = beta1 * m_[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] ** 2)
            m_hat = m_[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)
            params[i] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            
        U = params[0]
        V = params[1]
        W = params[2]
        
        if step % 100 == 0 and step > 0:
            if best_loss > 1e-6:
                lr *= 0.7
            else:
                lr *= 0.5
        
        if step % 20 == 0 and loss < 1e-8:
            if _verify_fast(U, V, W, T, tol=1e-9):
                found = True
                break
                
    return found, U, V, W

def _strassen_init_2x5x6(rank, seed):
    """
    Initialize using Strassen-inspired structure for 2x5x6.
    """
    n, m, p = 2, 5, 6
    nm, mp, pn = n*m, m*p, p*n
    np.random.seed(seed)
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col_pairs = [(0, 1), (2, 3), (4, 5)]
    mid_groups = [(0, 1), (2, 3)]
    
    col_idx = 0
    for mid_g in mid_groups:
        for col_p in col_pairs:
            j1, j2 = mid_g
            k1, k2 = col_p
            
            if col_idx >= rank:
                break
            
            # M1
            U[j1, col_idx] = 1; U[m+j2, col_idx] = 1
            V[j1*p+k1, col_idx] = 1; V[j2*p+k2, col_idx] = 1
            W[k1, col_idx] = 1; W[n+k2, col_idx] = -1
            col_idx += 1
            if col_idx >= rank: break
            # M2
            U[m+j1, col_idx] = 1; U[m+j2, col_idx] = 1
            V[j1*p+k1, col_idx] = 1
            W[n+k1, col_idx] = 1
            col_idx += 1
            if col_idx >= rank: break
            # M3
            U[j1, col_idx] = 1
            V[j2*p+k1, col_idx] = 1; V[j2*p+k2, col_idx] = -1
            W[k2, col_idx] = 1
            col_idx += 1
            if col_idx >= rank: break
            # M4
            U[m+j2, col_idx] = 1
            V[j1*p+k2, col_idx] = 1; V[j2*p+k1, col_idx] = -1
            W[k1, col_idx] = -1; W[n+k1, col_idx] = 1
            col_idx += 1
            if col_idx >= rank: break
            # M5
            U[j1, col_idx] = 1; U[j2, col_idx] = 1
            V[j2*p+k2, col_idx] = 1
            W[k1, col_idx] = -1; W[k2, col_idx] = 1
            col_idx += 1
            if col_idx >= rank: break
            # M6
            U[m+j1, col_idx] = 1; U[j1, col_idx] = -1
            V[j1*p+k1, col_idx] = 1; V[j1*p+k2, col_idx] = 1
            W[n+k2, col_idx] = -1
            col_idx += 1
            if col_idx >= rank: break
            # M7
            U[j2, col_idx] = 1; U[m+j2, col_idx] = -1
            V[j2*p+k1, col_idx] = 1; V[j2*p+k2, col_idx] = 1
            W[k1, col_idx] = 1; W[n+k2, col_idx] = 1
            col_idx += 1
    
    for k in range(p):
        if col_idx >= rank: break
        U[4, col_idx] = 1; V[4*p+k, col_idx] = 1; W[k, col_idx] = 1
        col_idx += 1
        if col_idx >= rank: break
        U[m+4, col_idx] = 1; V[4*p+k, col_idx] = 1; W[n+k, col_idx] = 1
        col_idx += 1
    
    for r in range(col_idx, rank):
        U[:, r] = np.random.randn(nm) * 0.1
        V[:, r] = np.random.randn(mp) * 0.1
        W[:, r] = np.random.randn(pn) * 0.1
    
    return U, V, W

def _structured_init(n, m, p, rank, seed):
    """
    Generate structured initializations.
    """
    nm, mp, pn = n*m, m*p, p*n
    np.random.seed(seed)
    
    if seed % 2 == 0:
        return _strassen_init_2x5x6(rank, seed)
    
    U = np.random.randn(nm, rank) * 0.1
    V = np.random.randn(mp, rank) * 0.1
    W = np.random.randn(pn, rank) * 0.1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    T = _construct_target_tensor(n, m, p)
    
    # Schoolbook fallback (Rank 60)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Step 1: Try the explicit Strassen-based construction (rank 42)
    # 2 mid-groups * 3 col-pairs * 7 Strassen terms = 42
    # + 1 leftover mid * 6 cols * 2 rows = 12
    # Total = 42 + 12 = 54
    U_str, V_str, W_str = _build_strassen_based_decomposition(n, m, p)
    if _verify_fast(U_str, V_str, W_str, T, tol=1e-8):
        best_rank = U_str.shape[1]
        best_U, best_V, best_W = U_str.copy(), V_str.copy(), W_str.copy()
        _try_rational_snapping(best_U, best_V, best_W, T)
    
    # Step 2: Try to improve with gradient descent from Strassen init
    if time.time() < deadline - 30:
        for rank in [best_rank - 1, best_rank - 2, best_rank - 3]:
            if rank < 10 or time.time() > deadline - 20:
                break
            
            found = False
            for restart in range(5):
                if time.time() > deadline - 10:
                    break
                
                U, V, W = _structured_init(n, m, p, rank, seed=restart * 17 + 3)
                found, U, V, W = _gradient_descent_with_restart(
                    U, V, W, T, n, m, p, rank, deadline, max_steps=200)
                
                if found:
                    U, V, W = _als_polish(U, V, W, T, n, m, p, iterations=10)
                    if _verify_fast(U, V, W, T, tol=1e-8):
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W
                        _try_rational_snapping(best_U, best_V, best_W, T)
                        break
            
            if found:
                break
    
    U_final = best_U
    V_final = best_V
    W_final = best_W
    
    U_list = [[float(x) for x in row] for row in U_final]
    V_list = [[float(x) for x in row] for row in V_final]
    W_list = [[float(x) for x in row] for row in W_final]
    
    return {"U": U_list, "V": V_list, "W": W_list}
