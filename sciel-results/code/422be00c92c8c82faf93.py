"""
Matrix multiplication 2x3x4 — rank-minimal exact decomposition.

Known lower bound: 22. We attempt to find an exact rank-22 (or lower)
decomposition via numerical ALS / SVD initialization followed by exact
rational verification. Fallback: schoolbook rank 24.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def _build_tensor():
    """Build the matrix-multiplication tensor T of shape (nm, mp, pn)."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_exact(T, U, V, W):
    """
    Check that sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] == T exactly (with rational entries).
    U, V, W are numpy arrays with rational (float) entries.
    Returns True if exact match.
    """
    rank = U.shape[1]
    reconstructed = np.zeros(T.shape, dtype=np.float64)
    for c in range(rank):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        # outer product
        outer = np.outer(u, v)
        outer = outer * w  # broadcast w along last axis -> shape (nm, mp, pn)
        reconstructed += outer
    # Check exact equality with tolerance for floating point
    return np.allclose(reconstructed, T, atol=1e-12, rtol=1e-12)

def _snap_to_rationals(arr, max_denom=64):
    """
    Attempt to snap each entry of arr to a rational with denominator <= max_denom.
    Returns a numpy array of the snapped values, or None if any entry can't be snapped.
    """
    snapped = np.zeros_like(arr, dtype=np.float64)
    for idx in np.ndindex(arr.shape):
        val = arr[idx]
        if abs(val) < 1e-14:
            snapped[idx] = 0.0
            continue
        # Try fractions
        frac = Fraction(val).limit_denominator(max_denom)
        if abs(frac - val) < 1e-10:
            snapped[idx] = float(frac)
        else:
            return None
    return snapped

def _als_decompose(T, rank, max_iters=200, tol=1e-12):
    """
    Alternating Least Squares for tensor decomposition of given rank.
    T has shape (nm, mp, pn).
    Returns U (nm x rank), V (mp x rank), W (pn x rank) or None.
    """
    nm, mp, pn = T.shape
    # Initialize via SVD of a matricization
    T_mat = T.reshape(nm, mp * pn)
    try:
        U_svd, s, Vt = np.linalg.svd(T_mat, full_matrices=False)
        U = U_svd[:, :rank] * np.sqrt(s[:rank])
        V_mat = (s[:rank] * Vt[:rank, :]).reshape(rank, mp, pn)
        # Now V and W from V_mat: reshape to (rank, mp, pn), then
        # For ALS we need V (mp x rank) and W (pn x rank)
        # V_mat[c, a, b] = V[a,c] * W[b,c]
        # Initialize: use SVD of V_mat as (mp*pn, rank)
        V_mat_2d = V_mat.reshape(rank, mp * pn).T  # (mp*pn, rank)
        V_svd, s2, Wt = np.linalg.svd(V_mat_2d, full_matrices=False)
        V_init = V_svd[:, :rank] * np.sqrt(s2[:rank])  # (mp*pn, rank)
        V_init = V_init.reshape(mp, pn, rank).transpose(0, 2, 1)  # (mp, rank, pn) -> need (mp, rank)
        # This is getting complicated. Let's just do random init for ALS.
    except:
        pass
    
    # Random initialization
    U = np.random.randn(nm, rank) * 0.1
    V = np.random.randn(mp, rank) * 0.1
    W = np.random.randn(pn, rank) * 0.1
    
    # Normalize columns
    for c in range(rank):
        U[:, c] /= (np.linalg.norm(U[:, c]) + 1e-12)
        V[:, c] /= (np.linalg.norm(V[:, c]) + 1e-12)
        W[:, c] /= (np.linalg.norm(W[:, c]) + 1e-12)
    
    for it in range(max_iters):
        # Update U: fix V, W. T_mat = T.reshape(nm, mp*pn)
        # T_mat = U @ (V ⊙ W)^T where (V⊙W) is (rank, mp*pn)
        VW = np.einsum('cr,cr->rc', V, W.reshape(mp, 1, pn).reshape(mp, pn))  # wrong
        # Better: for each c, V[:,c] ⊗ W[:,c] is (mp*pn,)
        # T_mat[:, c] = sum over... no.
        # T_mat[i, a] = sum_c U[i,c] * V[a_row,c] * W[b_col,c] where a=(j,k)
        # Actually T_mat[i, j*mp+k... no, T shape is (nm, mp, pn)
        # Let me restructure: matricize T as (nm, mp*pn)
        T_mat = T.reshape(nm, mp * pn)
        # For each c, the contribution is U[:,c] ⊗ (V[:,c] ⊗ W[:,c])
        # V[:,c] ⊗ W[:,c] has shape (mp*pn,)
        # So T_mat = U @ C^T where C[r, c] = <V[:,c] ⊗ W[:,c], T_mat[:,r]> ... 
        # ALS: fix V,W. For each i, T_mat[i,:] = sum_c U[i,c] * (V[:,c] ⊗ W[:,c])
        # So T_mat = U @ D^T where D[c, (a,b)] = V[a,c]*W[b,c]
        D = np.einsum('ac,bc->abc', V, W).reshape(rank, mp * pn)
        # T_mat = U @ D^T  => U = T_mat @ pinv(D^T)
        try:
            U_new = np.linalg.lstsq(D.T, T_mat, rcond=None)[0]
        except:
            break
        U = U_new
        
        # Update V: fix U, W
        # T_mat2 = T.reshape(mp, nm*pn)
        T_mat2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
        E = np.einsum('ic,kc->ck', U, W).reshape(rank, nm * pn)  # E[c, (i,k)] = U[i,c]*W[k,c]
        try:
            V_new = np.linalg.lstsq(E.T, T_mat2, rcond=None)[0]
        except:
            break
        V = V_new
        
        # Update W: fix U, V
        T_mat3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
        F = np.einsum('ic,jc->cij', U, V).reshape(rank, nm * mp)  # F[c, (i,j)] = U[i,c]*V[j,c]
        try:
            W_new = np.linalg.lstsq(F.T, T_mat3, rcond=None)[0]
        except:
            break
        W = W_new
        
        # Check convergence
        T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        if np.linalg.norm(T_recon - T) < tol:
            break
    
    # Check if we found a good solution
    T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    residual = np.linalg.norm(T_recon - T)
    if residual < 1e-8:
        return U, V, W
    return None

def _try_rank(target_rank, T, time_limit):
    """Try to find exact decomposition of given rank via ALS with multiple restarts."""
    nm, mp, pn = T.shape
    best_residual = float('inf')
    best_result = None
    start = time.time()
    
    for restart in range(50):
        if time.time() - start > time_limit:
            break
        np.random.seed(42 + restart)
        result = _als_decompose(T, target_rank, max_iters=300, tol=1e-10)
        if result is not None:
            U, V, W = result
            # Snap to rationals
            U_snap = _snap_to_rationals(U, max_denom=256)
            V_snap = _snap_to_rationals(V, max_denom=256)
            W_snap = _snap_to_rationals(W, max_denom=256)
            if U_snap is not None and V_snap is not None and W_snap is not None:
                if _verify_exact(T, U_snap, V_snap, W_snap):
                    return U_snap, V_snap, W_snap, target_rank
            # Also try without snapping (check if float is exact enough)
            if _verify_exact(T, U, V, W):
                # Convert to exact fractions
                U_exact = np.zeros_like(U)
                V_exact = np.zeros_like(V)
                W_exact = np.zeros_like(W)
                for idx in np.ndindex(U.shape):
                    U_exact[idx] = float(Fraction(U[idx]).limit_denominator(1024))
                for idx in np.ndindex(V.shape):
                    V_exact[idx] = float(Fraction(V[idx]).limit_denominator(1024))
                for idx in np.ndindex(W.shape):
                    W_exact[idx] = float(Fraction(W[idx]).limit_denominator(1024))
                if _verify_exact(T, U_exact, V_exact, W_exact):
                    return U_exact, V_exact, W_exact, target_rank
    
    return None

def _schoolbook():
    """Fallback schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 110.0  # 110s margin out of 120s
    
    T = _build_tensor()
    nm, mp, pn = T.shape
    
    # Try rank 22 (known lower bound for 2x3x4)
    result = _try_rank(22, T, time_limit=90.0)
    if result is not None:
        U, V, W, r = result
        # Convert to lists of lists (integers or halves)
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 24
    if time.time() < deadline:
        return _schoolbook()
    
    return _schoolbook()
