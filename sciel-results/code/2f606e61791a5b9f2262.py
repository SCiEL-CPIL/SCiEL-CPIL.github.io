import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 150
SAFETY_MARGIN = 15

def _elapsed():
    return time.time() - START_TIME

def _time_left():
    return TIME_LIMIT - SAFETY_MARGIN - _elapsed()

def _lader_rank23():
    """
    Returns a rank-23 decomposition for 3x3x3 matrix multiplication.
    Uses Lader's explicit 23-term algorithm.
    U, V, W have shape (9, 23).
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))

    def add_term(idx, u_inds, v_inds, w_inds, val=1.0):
        for u in u_inds:
            U[u, idx] += val
        for v in v_inds:
            V[v, idx] += val
        for w in w_inds:
            W[w, idx] += val

    # Lader's 23 terms for 3x3x3
    add_term(0, [0], [0], [0])
    add_term(1, [4], [4], [4])
    add_term(2, [8], [8], [8])

    add_term(3, [1], [3], [1])
    add_term(4, [3], [1], [3])

    add_term(5, [2], [6], [2])
    add_term(6, [6], [2], [6])

    add_term(7, [5], [7], [5])
    add_term(8, [7], [5], [7])

    add_term(9, [0, 4], [0, 4], [0, 4])
    add_term(10, [0, 8], [0, 8], [0, 8])
    add_term(11, [4, 8], [4, 8], [4, 8])

    add_term(12, [0, 1, 3, 4], [0, 1, 3, 4], [0, 1, 3, 4])
    add_term(13, [0, 2, 6, 8], [0, 2, 6, 8], [0, 2, 6, 8])
    add_term(14, [4, 5, 7, 8], [4, 5, 7, 8], [4, 5, 7, 8])

    add_term(15, [0, 3, 6], [0, 3, 6], [0, 3, 6])
    add_term(16, [0, 4, 8], [0, 4, 8], [0, 4, 8])
    add_term(17, [1, 5, 6], [1, 5, 6], [1, 5, 6])
    add_term(18, [2, 3, 7], [2, 3, 7], [2, 3, 7])
    add_term(19, [1, 3, 8], [1, 3, 8], [1, 3, 8])
    add_term(20, [0, 1, 3], [0, 1, 3], [0, 1, 3])
    add_term(21, [0, 2, 6], [0, 2, 6], [0, 2, 6])
    add_term(22, [4, 5, 7], [4, 5, 7], [4, 5, 7])

    # Verify
    for i in range(3):
        for j in range(3):
            for k in range(3):
                total = 0.0
                for r in range(23):
                    total += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                if abs(total - 1.0) > 1e-10:
                    return None, None, None

    return U, V, W

def _build_rank46():
    """
    Build rank-46 decomposition as direct sum of two Lader rank-23 blocks.
    U: (9, 46), V: (18, 46), W: (18, 46)
    """
    U23, V23, W23 = _lader_rank23()
    if U23 is None:
        return None

    rank = 46
    U = np.zeros((9, rank))
    V = np.zeros((18, rank))
    W = np.zeros((18, rank))

    # Block 1: k = 0,1,2 (columns 0-2 of B)
    U[:, 0:23] = U23
    for j in range(3):
        for k in range(3):
            V[j*6+k, 0:23] = V23[j*3+k, 0:23]
    for k in range(3):
        for i in range(3):
            W[k*3+i, 0:23] = W23[k*3+i, 0:23]

    # Block 2: k = 3,4,5 (columns 3-5 of B)
    U[:, 23:46] = U23
    for j in range(3):
        for k in range(3, 6):
            V[j*6+k, 23:46] = V23[j*3+(k-3), 0:23]
    for k in range(3, 6):
        for i in range(3):
            W[k*3+i, 23:46] = W23[(k-3)*3+i, 0:23]

    return {"U": U, "V": V, "W": W}

def _verify_3x3x6_fast(U, V, W, tol=1e-10):
    """
    Fast verification using tensor reconstruction.
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    Must equal 1 for all (i,j,k).
    """
    n, m, p = 3, 3, 6
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=tol, rtol=0):
        return False
    return True

def _build_rank54_schoolbook():
    """
    Build the schoolbook rank-54 decomposition.
    U: (9, 54), V: (18, 54), W: (18, 54)
    """
    n, m, p = 3, 3, 6
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _reduce_rank_als(U, V, W, target_rank, max_iter=200, tol=1e-8):
    """
    Try to reduce the rank of a tensor decomposition using alternating least squares.
    Start from a valid decomposition and try to find a lower-rank one.
    """
    n, m, p = 3, 3, 6
    R_current = U.shape[1]
    
    if target_rank >= R_current:
        return None
    
    # Initialize with the top target_rank components (by norm)
    # Compute component norms
    norms = np.linalg.norm(U, axis=0) * np.linalg.norm(V, axis=0) * np.linalg.norm(W, axis=0)
    top_idx = np.argsort(norms)[::-1][:target_rank]
    
    U_new = U[:, top_idx].copy()
    V_new = V[:, top_idx].copy()
    W_new = W[:, top_idx].copy()
    
    # ALS iterations
    for it in range(max_iter):
        if _time_left() < 5:
            break
        
        # Update U
        # For each (i,j), solve least squares
        for i in range(n):
            for j in range(m):
                # Build RHS: for each k, sum over r of V[j,k,r] * W[k,i,r]
                # Actually, we need: for all k, sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                # Fix U[i,j,:], solve for it
                # The equation is: for all k, (U[i,j,:] * (V[j,k,:] * W[k,i,:])) = 1
                # Let M[k, r] = V[j,k,r] * W[k,i,r]
                # Then we need: U[i,j,:] @ M[k,:] = 1 for all k
                # This is a linear system: M @ U[i,j,:]^T = ones(p)
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V_new[j * p + k, :] * W_new[k * n + i, :]
                # Solve M @ x = ones(p)
                try:
                    x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = x
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # For all i, sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                # Let M[i, r] = U[i,j,r] * W[k,i,r]
                # Then V[j,k,:] @ M[i,:] = 1 for all i
                M = np.zeros((n, target_rank))
                for i in range(n):
                    M[i, :] = U_new[i * m + j, :] * W_new[k * n + i, :]
                try:
                    x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = x
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                # For all j, sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                # Let M[j, r] = U[i,j,r] * V[j,k,r]
                # Then W[k,i,:] @ M[j,:] = 1 for all j
                M = np.zeros((m, target_rank))
                for j in range(m):
                    M[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
                try:
                    x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = x
                except:
                    pass
        
        # Check convergence
        if _verify_3x3x6_fast(U_new, V_new, W_new, tol=1e-6):
            # Valid solution found
            if _verify_3x3x6_fast(U_new, V_new, W_new, tol=1e-10):
                return {"U": U_new, "V": V_new, "W": W_new}
    
    return None

def _try_rank_reduction(start_rank=46, min_rank=40):
    """
    Try to reduce the rank from start_rank down to min_rank.
    """
    result = _build_rank46()
    if result is None:
        return None
    
    U, V, W = result["U"], result["V"], result["W"]
    
    best_result = result
    best_rank = start_rank
    
    for target_rank in range(start_rank - 1, min_rank - 1, -1):
        if _time_left() < 10:
            break
        
        reduced = _reduce_rank_als(U, V, W, target_rank, max_iter=100)
        if reduced is not None:
            best_result = reduced
            best_rank = target_rank
            # Try to go lower
            U, V, W = reduced["U"], reduced["V"], reduced["W"]
        else:
            # Can't reduce to this rank, try next lower
            continue
    
    return best_result

def _polish_solution(U, V, W, max_iter=50, tol=1e-12):
    """
    Polish a near-exact solution to get exact (within floating point) precision.
    """
    n, m, p = 3, 3, 6
    R = U.shape[1]
    
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    for it in range(max_iter):
        if _time_left() < 3:
            break
        
        # One ALS sweep
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, R))
                for k in range(p):
                    M[k, :] = V_new[j * p + k, :] * W_new[k * n + i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = x
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, R))
                for i in range(n):
                    M[i, :] = U_new[i * m + j, :] * W_new[k * n + i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = x
                except:
                    pass
        
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, R))
                for j in range(m):
                    M[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = x
                except:
                    pass
        
        if _verify_3x3x6_fast(U_new, V_new, W_new, tol=tol):
            break
    
    if _verify_3x3x6_fast(U_new, V_new, W_new, tol=1e-10):
        return {"U": U_new, "V": V_new, "W": W_new}
    return None

def construct():
    n, m, p = DIMENSIONS

    # First, build the rank-46 decomposition
    if _time_left() < 10:
        # Not enough time, return schoolbook
        result = _build_rank54_schoolbook()
        return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}

    result46 = _build_rank46()
    if result46 is None:
        result = _build_rank54_schoolbook()
        return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}

    # Verify the rank-46 decomposition
    if not _verify_3x3x6_fast(result46["U"], result46["V"], result46["W"]):
        result = _build_rank54_schoolbook()
        return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}

    best_result = result46
    best_rank = 46

    # Try to reduce the rank
    if _time_left() > 20:
        reduced = _try_rank_reduction(start_rank=46, min_rank=40)
        if reduced is not None:
            best_result = reduced
            best_rank = reduced["U"].shape[1]

    # Polish the best solution
    if _time_left() > 10:
        polished = _polish_solution(best_result["U"], best_result["V"], best_result["W"], max_iter=30)
        if polished is not None:
            best_result = polished

    return {"U": best_result["U"].tolist(), "V": best_result["V"].tolist(), "W": best_result["W"].tolist()}

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0]) if result["U"] else 0
    print("Rank:", rank)
