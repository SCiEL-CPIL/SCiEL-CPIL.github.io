"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _to_rational_array(mat, tol=1e-8):
    """Convert a numeric matrix to a matrix of Fractions (exact rationals)."""
    rows, cols = mat.shape
    result = [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            val = mat[i, j]
            if abs(val) < tol:
                result[i][j] = Fraction(0)
                continue
            found = False
            for den in range(1, 201):
                num = round(val * den)
                if abs(val - num / den) < tol:
                    result[i][j] = Fraction(num, den)
                    found = True
                    break
            if not found:
                result[i][j] = Fraction(val).limit_denominator(1000000)
    return result

def _verify_exact(U, V, W, n, m, p, rank, tol=1e-6):
    """Verify decomposition exactly by checking all valid and invalid triplets."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check all valid (i,j,k) triplets sum to 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check all invalid triplets sum to 0 (exhaustive for small dims)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i == i2 and j == j2 and k == k2:
                                continue
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j, c] * V[j2*p+k, c] * W[k*n+i2, c]
                            if abs(s) > tol:
                                return False
    return True

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[(i*m+j), (j'*p+k), (k'*n+i')]."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _als_solve_U(T, V, W, rank, n, m, p):
    """Solve for U given V, W."""
    # B[bc, r] = V[b, r] * W[c, r], shape (m*p * p*n, rank)
    B = np.zeros((m * p * p * n, rank))
    for b in range(m * p):
        for c in range(p * n):
            B[b * p * n + c, :] = V[b, :] * W[c, :]
    T_flat = T.reshape(n * m, m * p * p * n)
    try:
        U_sol = T_flat @ np.linalg.pinv(B.T)
    except:
        U_sol = np.random.randn(n * m, rank)
    return U_sol

def _als_solve_V(T, U, W, rank, n, m, p):
    """Solve for V given U, W."""
    # D[ac, r] = U[a, r] * W[c, r], shape (n*m * p*n, rank)
    D = np.zeros((n * m * p * n, rank))
    for a in range(n * m):
        for c in range(p * n):
            D[a * p * n + c, :] = U[a, :] * W[c, :]
    # T_reshaped[(a,c), b] = T[a, b, c]
    T_reshaped = np.zeros((n * m * p * n, m * p))
    for a in range(n * m):
        for c in range(p * n):
            T_reshaped[a * p * n + c, :] = T[a, :, c]
    try:
        V_T = np.linalg.pinv(D) @ T_reshaped
        V_sol = V_T.T
    except:
        V_sol = np.random.randn(m * p, rank)
    return V_sol

def _als_solve_W(T, U, V, rank, n, m, p):
    """Solve for W given U, V."""
    # D[ab, r] = U[a, r] * V[b, r], shape (n*m * m*p, rank)
    D = np.zeros((n * m * m * p, rank))
    for a in range(n * m):
        for b in range(m * p):
            D[a * m * p + b, :] = U[a, :] * V[b, :]
    # T_reshaped[(a,b), c] = T[a, b, c]
    T_reshaped = np.zeros((n * m * m * p, p * n))
    for a in range(n * m):
        for b in range(m * p):
            T_reshaped[a * m * p + b, :] = T[a, b, :]
    try:
        W_T = np.linalg.pinv(D) @ T_reshaped
        W_sol = W_T.T
    except:
        W_sol = np.random.randn(p * n, rank)
    return W_sol

def _try_als(T, n, m, p, target_rank, max_iters=80, n_trials=20, deadline=None):
    """Try to find a rank-target_rank decomposition via ALS."""
    start_time = time.time()
    for trial in range(n_trials):
        if deadline and time.time() > deadline - 10:
            break
        
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        for it in range(max_iters):
            if deadline and time.time() > deadline - 5:
                break
            U = _als_solve_U(T, V, W, target_rank, n, m, p)
            V = _als_solve_V(T, U, W, target_rank, n, m, p)
            W = _als_solve_W(T, U, V, target_rank, n, m, p)
        
        if deadline and time.time() > deadline - 3:
            break
        
        valid = _verify_exact(U, V, W, n, m, p, target_rank, tol=1e-4)
        if valid:
            U_rat = _to_rational_array(U)
            V_rat = _to_rational_array(V)
            W_rat = _to_rational_array(W)
            valid_rat = _verify_exact(U_rat, V_rat, W_rat, n, m, p, target_rank, tol=1e-6)
            if valid_rat:
                return {"U": U_rat, "V": V_rat, "W": W_rat}
    return None

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    
    # Build the schoolbook solution as fallback
    def make_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = schoolbook_rank
    
    # Build the full tensor T of shape (n*m, m*p, p*n)
    T = _build_tensor(n, m, p)
    
    import random
    random.seed(12345)
    np.random.seed(12345)
    
    lower_bound = max(n*m, m*p, p*n)  # 12
    
    # Try ranks from lower bound up to schoolbook, prioritizing higher ranks first
    # (more likely to find a solution) then work down
    for target_rank in range(schoolbook_rank - 1, lower_bound - 1, -1):
        if time.time() > DEADLINE - 30:
            break
        
        result = _try_als(T, n, m, p, target_rank, max_iters=60, n_trials=15, deadline=DEADLINE)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
