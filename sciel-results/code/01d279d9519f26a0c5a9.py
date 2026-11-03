"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _schoolbook():
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

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Uf = [[Fraction(u).limit_denominator(10**6) for u in row] for row in U]
    Vf = [[Fraction(v).limit_denominator(10**6) for v in row] for row in V]
    Wf = [[Fraction(w).limit_denominator(10**6) for w in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = Fraction(0)
                for c in range(rank):
                    actual += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                if actual != expected:
                    return False
    return True

def _snap_to_half_integer(x, tol=1e-6):
    k = round(2 * x)
    if abs(2 * x - k) < tol:
        return k / 2.0
    return x

def _is_half_integer_matrix(M, tol=1e-6):
    for row in M:
        for x in row:
            if abs(2*x - round(2*x)) > tol:
                return False
    return True

def _als_decompose(n, m, p, rank, max_iter=50, tol=1e-10, deadline=None, seed=0):
    np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        if err < tol:
            break
    
    return U, V, W

def _build_matricization(n, m, p):
    """Build the tensor matricization: T[i*n+j, k] = delta for (i,j,k) where tensor=1"""
    # The matrix multiplication tensor T has T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # Matricization: reshape to (n*m, p) where row index = i*m+j, col index = k
    T = np.zeros((n*m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, k] = 1.0
    return T

def _iht_decompose(n, m, p, rank, max_iter=100, tol=1e-8, deadline=None, seed=0):
    """
    Iterative Hard Thresholding to find rank-r decomposition.
    We want U (n*m x r), V (m*p x r), W (p*n x r) such that
    for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    Strategy: Use a matricization approach. Flatten the tensor into a matrix and use
    IHT to recover a low-rank approximation, then map back to the 3-way factors.
    
    Alternative: Directly optimize the 3 factors using a projected gradient approach
    where we project onto the set of half-integer matrices.
    """
    np.random.seed(seed)
    
    # Initialize factors
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    step_size = 1.0 / (2 * rank)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient with respect to each factor
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Current value
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    # Gradient: d/dc = 1 - val
                    diff = 1.0 - val
                    # dU[i*m+j,c] = diff * V[j*p+k,c] * W[k*n+i,c]
                    grad_U[i*m+j, :] += diff * V[j*p+k, :] * W[k*n+i, :]
                    # dV[j*p+k,c] = diff * U[i*m+j,c] * W[k*n+i,c]
                    grad_V[j*p+k, :] += diff * U[i*m+j, :] * W[k*n+i, :]
                    # dW[k*n+i,c] = diff * U[i*m+j,c] * V[j*p+k,c]
                    grad_W[k*n+i, :] += diff * U[i*m+j, :] * V[j*p+k, :]
        
        # Gradient descent step
        U = U - step_size * grad_U
        V = V - step_size * grad_V
        W = W - step_size * grad_W
        
        # Project onto half-integer lattice (snap and verify)
        U_snapped = np.round(2 * U) / 2.0
        V_snapped = np.round(2 * V) / 2.0
        W_snapped = np.round(2 * W) / 2.0
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_snapped[i*m+j, :], V_snapped[j*p+k, :] * W_snapped[k*n+i, :])
                    err += (val - 1.0)**2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_snapped.copy(), V_snapped.copy(), W_snapped.copy()
        
        if err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def _try_rank_iht(rank, deadline, n, m, p):
    """Try IHT-based decomposition for given rank."""
    for seed in range(30):
        if time.time() > deadline - 5:
            return None
        try:
            U, V, W, err = _iht_decompose(n, m, p, rank, max_iter=50, deadline=deadline-5, seed=seed)
            
            U_list = [[U[i][j] for j in range(rank)] for i in range(n*m)]
            V_list = [[V[i][j] for j in range(rank)] for i in range(m*p)]
            W_list = [[W[i][j] for j in range(rank)] for i in range(p*n)]
            
            if _is_half_integer_matrix(U_list) and _is_half_integer_matrix(V_list) and _is_half_integer_matrix(W_list):
                if _verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        except:
            continue
    return None

def _try_rank_als(rank, deadline, n, m, p):
    """Try ALS-based decomposition for given rank."""
    for seed in range(20):
        if time.time() > deadline - 5:
            return None
        try:
            U, V, W = _als_decompose(n, m, p, rank, max_iter=20, deadline=deadline-5, seed=seed)
            U_snapped = [[_snap_to_half_integer(x) for x in row] for row in U]
            V_snapped = [[_snap_to_half_integer(x) for x in row] for row in V]
            W_snapped = [[_snap_to_half_integer(x) for x in row] for row in W]
            
            if _is_half_integer_matrix(U_snapped) and _is_half_integer_matrix(V_snapped) and _is_half_integer_matrix(W_snapped):
                if _verify(U_snapped, V_snapped, W_snapped):
                    return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
        except:
            continue
    return None

def _try_rank(rank, deadline, n, m, p):
    """Try both IHT and ALS methods for given rank."""
    # Try IHT first
    result = _try_rank_iht(rank, deadline, n, m, p)
    if result:
        return result
    # Try ALS
    result = _try_rank_als(rank, deadline, n, m, p)
    if result:
        return result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    
    # Try ranks from lower bound up
    for rank in range(16, 49):
        if time.time() > deadline - 10:
            break
        result = _try_rank(rank, deadline, n, m, p)
        if result:
            return result
            
    return _schoolbook()
# EVOLVE-BLOCK-END
