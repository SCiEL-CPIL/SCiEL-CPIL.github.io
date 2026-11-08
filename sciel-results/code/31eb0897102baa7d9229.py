"""Matrix multiplication 2x4 by 4x7 rank-minimal decomposition search."""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reconstruct tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    T_true = _build_tensor(n, m, p)
    return np.allclose(T, T_true, atol=tol)

def _check_half_integer(arr, tol=1e-9):
    """Check if all entries are integer or half-integer."""
    arr = np.array(arr)
    vals = arr.ravel()
    return np.all(np.abs(2 * vals - np.round(2 * vals)) <= tol)

def _to_list(arr):
    """Convert numpy array to list of lists with rounded half-integer values."""
    arr = np.array(arr)
    rounded = np.round(arr * 2) / 2.0
    return rounded.tolist()

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition (rank = n*m*p)."""
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
    return U, V, W

def _rank1_check(C, tol=1e-10):
    """Check if matrix C is rank 1. Returns (is_rank1, u, v) if so."""
    svals = np.linalg.svd(C, compute_uv=False)
    if len(svals) == 0:
        return True, None, None
    if svals[0] < 1e-12:
        return True, None, None
    if len(svals) > 1 and svals[1] > tol * max(svals[0], 1e-12):
        return False, None, None
    U_sv, S_sv, Vt_sv = np.linalg.svd(C)
    u = U_sv[:, 0] * np.sqrt(S_sv[0])
    v = Vt_sv[0, :] * np.sqrt(S_sv[0])
    return True, u, v

def _try_w_based_solve(W_cand, T, nm, mp, pn, deadline):
    """
    Given a candidate W matrix, try to solve for U and V.
    Returns (U, V) if successful, else None.
    """
    r = W_cand.shape[1]
    
    # Check if W has full column rank
    if np.linalg.matrix_rank(W_cand) < r:
        return None
    
    try:
        WtW = W_cand.T @ W_cand
        WtW_inv = np.linalg.inv(WtW)
    except np.linalg.LinAlgError:
        return None
    
    # Contract T with W: TW[i,j,r] = sum_k T[i,j,k] * W[k,r]
    TW = np.einsum('ijk,kr->ijr', T, W_cand)
    
    # For each r, X_r = sum_s WtW_inv[r,s] * TW[:,:,s] should be rank-1
    U_sol = np.zeros((nm, r))
    V_sol = np.zeros((mp, r))
    
    for ri in range(r):
        if time.time() > deadline:
            return None
        X_ri = np.einsum('is,ijr->ij', WtW_inv[ri, :], TW)
        
        ok, u, v = _rank1_check(X_ri, tol=1e-8)
        if not ok:
            return None
        if u is not None:
            U_sol[:, ri] = u
            V_sol[:, ri] = v
    
    # Verify the decomposition
    T_recon = np.zeros((nm, mp, pn))
    for ri in range(r):
        T_recon += np.einsum('i,j,k->ijk', U_sol[:, ri], V_sol[:, ri], W_cand[:, ri])
    
    if not np.allclose(T_recon, T, atol=1e-8):
        return None
    
    # Check half-integer constraint
    if _check_half_integer(U_sol) and _check_half_integer(V_sol) and _check_half_integer(W_cand):
        return U_sol, V_sol
    
    # Try rounding
    Ur = np.round(U_sol * 2) / 2
    Vr = np.round(V_sol * 2) / 2
    Wr = np.round(W_cand * 2) / 2
    
    T_recon2 = np.zeros((nm, mp, pn))
    for ri in range(r):
        T_recon2 += np.einsum('i,j,k->ijk', Ur[:, ri], Vr[:, ri], Wr[:, ri])
    
    if np.allclose(T_recon2, T, atol=1e-8):
        return Ur, Vr, Wr
    
    return None

def _try_structured_decomposition(n, m, p, time_limit=120.0):
    """
    Try structured decompositions based on algebraic patterns.
    For 2x4x7, we exploit the structure:
    - n=2: output has 2 rows
    - m=4: intermediate dimension
    - p=7: input has 7 columns
    
    We try patterns where W has block structure based on k (0..6),
    and U, V have corresponding structures.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    T = _build_tensor(n, m, p)
    pn = p * n  # 14
    nm = n * m  # 8
    mp = m * p  # 28
    
    best_rank = n * m * p
    best_result = None
    
    # Lower bound from matricization
    T_mat1 = T.reshape(nm, mp * pn)
    T_mat2 = T.reshape(nm * mp, pn)
    T_mat3 = T.reshape(nm * pn, mp)
    rank_lb = max(np.linalg.matrix_rank(T_mat1), 
                  np.linalg.matrix_rank(T_mat2),
                  np.linalg.matrix_rank(T_mat3))
    
    # Try ranks from lower bound to schoolbook
    for r in range(max(rank_lb, 20), min(n * m * p + 1, 50)):
        if time.time() > deadline:
            break
        
        if r >= best_rank:
            break
        
        # Try different structured W patterns
        # Pattern 1: Block diagonal with 7 blocks (one per k)
        for block_size in range(1, min(4, r // 7 + 1)):
            if time.time() > deadline:
                break
            
            W_cand = np.zeros((pn, r))
            col_offset = 0
            for k in range(p):
                for i in range(n):
                    row = k * n + i
                    for b in range(block_size):
                        if col_offset + b < r:
                            W_cand[row, col_offset + b] = 1.0
                col_offset += block_size
            
            if np.linalg.matrix_rank(W_cand) >= min(r, pn):
                result = _try_w_based_solve(W_cand, T, nm, mp, pn, deadline)
                if result is not None:
                    U_sol, V_sol = result
                    best_rank = r
                    best_result = (U_sol.copy(), V_sol.copy(), W_cand.copy())
                    break
        
        if time.time() > deadline:
            break
        
        if best_rank == r:
            continue
        
        # Pattern 2: Sparse random with structured density
        np.random.seed(42 + r)
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            W_cand = np.zeros((pn, r))
            # Each row gets 1-2 nonzeros
            for row in range(pn):
                nnz = min(r, np.random.choice([1, 2], p=[0.5, 0.5]))
                cols = np.random.choice(r, size=nnz, replace=False)
                W_cand[row, cols] = np.random.choice([-1, 1], size=nnz)
            
            if np.linalg.matrix_rank(W_cand) >= min(r, pn):
                result = _try_w_based_solve(W_cand, T, nm, mp, pn, deadline)
                if result is not None:
                    U_sol, V_sol = result
                    best_rank = r
                    best_result = (U_sol.copy(), V_sol.copy(), W_cand.copy())
                    break
        
        if best_rank == r:
            continue
    
    if best_result is not None:
        U, V, W = best_result
        return _to_list(U), _to_list(V), _to_list(W)
    return None

def _try_near_schoolbook(n, m, p, time_limit=120.0):
    """
    Try to reduce rank from schoolbook by finding linear dependencies.
    Start from schoolbook and try to merge terms.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    T = _build_tensor(n, m, p)
    pn = p * n
    nm = n * m
    mp = m * p
    
    best_rank = n * m * p
    best_result = None
    
    # Get schoolbook decomposition
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    U_arr = np.array(U_sb, dtype=np.float64)
    V_arr = np.array(V_sb, dtype=np.float64)
    W_arr = np.array(W_sb, dtype=np.float64)
    rank = n * m * p
    
    # Try to find a smaller subset of terms that still works
    # This is a combinatorial problem, so we try random subsets
    np.random.seed(123)
    
    for target_r in range(max(rank - 10, 40), rank + 1):
        if time.time() > deadline:
            break
        
        if target_r >= best_rank:
            break
        
        # Try to find target_r terms from the schoolbook that span the tensor
        # Use random selection with replacement
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            # Select target_r random columns from schoolbook
            indices = np.random.choice(rank, size=target_r, replace=True)
            
            U_cand = U_arr[:, indices]
            V_cand = V_arr[:, indices]
            W_cand = W_arr[:, indices]
            
            # Check if this gives a valid decomposition
            T_recon = np.zeros((nm, mp, pn))
            for ri in range(target_r):
                T_recon += np.einsum('i,j,k->ijk', U_cand[:, ri], V_cand[:, ri], W_cand[:, ri])
            
            if np.allclose(T_recon, T, atol=1e-8):
                if _check_half_integer(U_cand) and _check_half_integer(V_cand) and _check_half_integer(W_cand):
                    best_rank = target_r
                    best_result = (U_cand.copy(), V_cand.copy(), W_cand.copy())
                    break
        
        if best_rank == target_r:
            continue
    
    if best_result is not None:
        U, V, W = best_result
        return _to_list(U), _to_list(V), _to_list(W)
    return None

# EVOLVE-BLOCK-END

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Try structured decomposition
    try:
        remaining = deadline - time.time()
        if remaining > 5.0:
            result = _try_structured_decomposition(n, m, p, time_limit=min(remaining, 70.0))
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    except:
        pass
    
    # Try near-schoolbook reduction
    try:
        remaining = deadline - time.time()
        if remaining > 5.0:
            result = _try_near_schoolbook(n, m, p, time_limit=min(remaining, 50.0))
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    except:
        pass
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
