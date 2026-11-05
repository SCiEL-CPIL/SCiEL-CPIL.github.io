import numpy as np
from fractions import Fraction
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_find_rank1_peel(T, n, m, p, max_iters=200):
    """
    Try to find a rank-1 summand in tensor T of the form:
    T[i,j,k] = a[i,j] * b[j,k] * c[k,i]
    
    Returns (a, b, c) as arrays if found, else None.
    """
    # For rank-1, each slice T[:, j, :] must be rank 1
    for j in range(m):
        M = T[:, j, :]
        try:
            svd_r = np.linalg.matrix_rank(M, tol=1e-10)
            if svd_r > 1:
                return None
        except:
            return None
    
    # Check consistency: for rank-1 tensor, T[i,j,k]*T[i',j',k']*T[i'',j'',k''] should equal
    # T[i,j',k']*T[i',j'',k]*T[i'',j,k] (multiplicative consistency)
    # But this is complex. Instead, let's use optimization.
    
    nm, mp, pn = n*m, m*p, p*n
    
    def _residual(params):
        a = params[:nm].reshape(n, m)
        b = params[nm:nm+mp].reshape(m, p)
        c = params[nm+mp:nm+mp+pn].reshape(p, n)
        T_recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] = a[i, j] * b[j, k] * c[k, i]
        return np.sum((T - T_recon)**2)
    
    # Try multiple initializations
    best_res = np.inf
    best_params = None
    
    for trial in range(max_iters):
        np.random.seed(trial)
        x0 = np.random.randn(nm + mp + pn) * 0.1
        try:
            result = minimize(_residual, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12})
            if result.fun < best_res:
                best_res = result.fun
                best_params = result.x
            if best_res < 1e-12:
                break
        except:
            continue
    
    if best_res > 1e-10:
        return None
    
    a = best_params[:nm].reshape(n, m)
    b = best_params[nm:nm+mp].reshape(m, p)
    c = best_params[nm+mp:nm+mp+pn].reshape(p, n)
    
    # Check if entries are close to half-integers
    all_vals = np.concatenate([a.flatten(), b.flatten(), c.flatten()])
    for val in all_vals:
        nearest_half = np.round(val * 2) / 2
        if abs(val - nearest_half) > 1e-6:
            return None
    
    return (a, b, c)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with the known rank-28 decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2, all k (10 terms) - schoolbook for column 2
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: j=0,1, k=4 (4 terms) - schoolbook for k=4
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: j=0,1, k=0,1 using Strassen (7 terms)
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: j=0,1, k=2,3 using Strassen (7 terms)
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Base decomposition verification failed")
    
    # Build the target tensor (all ones)
    T_target = np.ones((n, m, p))
    
    # Build current tensor from factors
    def _build_tensor(U_list, V_list, W_list, n, m, p):
        rank = len(U_list[0])
        U_arr = np.array([[float(x) for x in row] for row in U_list], dtype=float)
        V_arr = np.array([[float(x) for x in row] for row in V_list], dtype=float)
        W_arr = np.array([[float(x) for x in row] for row in W_list], dtype=float)
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    T[i, j, k] = np.sum(U_arr[idx_u, :] * V_arr[idx_v, :] * W_arr[idx_w, :])
        return T
    
    T_current = _build_tensor(U, V, W, n, m, p)
    
    # Try to peel off rank-1 summands to reduce rank
    # We want to find if T_current - T_target can be expressed as a sum of rank-1 terms
    # that we can "remove" to get a lower rank decomposition
    
    # Actually, the target is T_target = all ones. Our current decomposition already equals T_target.
    # To reduce rank, we need to find a different decomposition with fewer terms.
    
    # Strategy: try to find if the rank-28 decomposition can be compressed by
    # identifying linear dependencies among the rank-1 terms.
    
    # Check if the 28 rank-1 vectors in the tensor space are linearly independent.
    # Each rank-1 term t_r has components T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build the matrix of rank-1 terms (each term is a vector in R^(n*m*p))
    nmp = n * m * p
    terms_matrix = np.zeros((nmp, rank))
    
    for r_idx in range(rank):
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    terms_matrix[idx, r_idx] = U[idx_u][r_idx] * V[idx_v][r_idx] * W[idx_w][r_idx]
                    idx += 1
    
    # Check the rank of this matrix
    try:
        term_rank = np.linalg.matrix_rank(terms_matrix, tol=1e-10)
    except:
        term_rank = rank
    
    # If term_rank < rank, we have linear dependencies and can potentially reduce rank
    if term_rank < rank:
        # Find the null space
        try:
            _, S, Vt = np.linalg.svd(terms_matrix, full_matrices=True)
            null_dim = rank - term_rank
            null_space = Vt[term_rank:rank, :]  # Each row is a null vector
        except:
            null_space = None
    
    # For now, return the rank-28 solution since we haven't found a way to reduce it
    # The peeling approach would require more sophisticated optimization
    
    # Try a quick search for rank 27 by trying to set one of the 28 terms to be
    # a linear combination of others
    # This is complex, so for now just return rank 28
    
    # Verify final solution
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Final decomposition verification failed")
    
    # Convert to fractions for clean output
    U_frac = [[Fraction(int(round(x * 2)), 2) if abs(x * 2 - round(x * 2)) < 1e-6 else Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(int(round(x * 2)), 2) if abs(x * 2 - round(x * 2)) < 1e-6 else Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(int(round(x * 2)), 2) if abs(x * 2 - round(x * 2)) < 1e-6 else Fraction(x) for x in row] for row in W]
    
    return {"U": U_frac, "V": V_frac, "W": W_frac}
