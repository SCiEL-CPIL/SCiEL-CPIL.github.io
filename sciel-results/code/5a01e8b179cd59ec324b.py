import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(s - 1) > tol:
                    return False
    return True

def _construct_rank_55(n, m, p):
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(5):
        idx_e = 0 * p + 2*j
        idx_f = 0 * p + 2*j + 1
        idx_g = 1 * p + 2*j
        idx_h = 1 * p + 2*j + 1
        idx_c00 = 2*j * n + 0
        idx_c01 = (2*j+1) * n + 0
        idx_c10 = 2*j * n + 1
        idx_c11 = (2*j+1) * n + 1
        idx_a = 0
        idx_b = 1
        idx_c = 2
        idx_d = 3
        
        U[idx_a][col] += 1
        U[idx_d][col] += 1
        V[idx_e][col] += 1
        V[idx_h][col] += 1
        W[idx_c00][col] += 1
        W[idx_c11][col] += 1
        col += 1
        
        U[idx_c][col] += 1
        U[idx_d][col] += 1
        V[idx_e][col] += 1
        W[idx_c10][col] += 1
        col += 1
        
        U[idx_a][col] += 1
        V[idx_f][col] += 1
        V[idx_h][col] -= 1
        W[idx_c01][col] += 1
        col += 1
        
        U[idx_d][col] += 1
        V[idx_g][col] += 1
        V[idx_f][col] -= 1
        W[idx_c00][col] += 1
        W[idx_c10][col] += 1
        col += 1
        
        U[idx_a][col] += 1
        U[idx_b][col] += 1
        V[idx_h][col] += 1
        W[idx_c00][col] -= 1
        W[idx_c01][col] += 1
        col += 1
        
        U[idx_c][col] += 1
        U[idx_d][col] += 1
        V[idx_f][col] += 1
        V[idx_h][col] -= 1
        W[idx_c11][col] -= 1
        col += 1
        
        U[idx_a][col] += 1
        U[idx_b][col] += 1
        V[idx_g][col] += 1
        V[idx_e][col] -= 1
        W[idx_c00][col] += 1
        col += 1

    for k in range(10):
        U[4][col] += 1
        V[2 * p + k][col] += 1
        W[k * n + 0][col] += 1
        col += 1
        
    for k in range(10):
        U[5][col] += 1
        V[2 * p + k][col] += 1
        W[k * n + 1][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}

def _try_rank_54(n, m, p, deadline):
    """Try to find a rank-54 decomposition using alternating minimization."""
    rank = 54
    np.random.seed(123)
    
    # Initialize from rank-55 solution with one column zeroed out
    base = _construct_rank_55(n, m, p)
    U = np.array(base["U"], dtype=float)
    V = np.array(base["V"], dtype=float)
    W = np.array(base["W"], dtype=float)
    
    # Zero out the last column
    U[:, -1] = 0
    V[:, -1] = 0
    W[:, -1] = 0
    
    # Truncate to rank 54
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]
    
    # Add some random perturbation
    U += 0.01 * np.random.randn(*U.shape)
    V += 0.01 * np.random.randn(*V.shape)
    W += 0.01 * np.random.randn(*W.shape)
    
    for iteration in range(500):
        if time.time() > deadline - 10:
            break
        
        # Solve for W
        for k in range(p):
            for i in range(n):
                B = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        B[j, c] = U[i*m+j, c] * V[j*p+k, c]
                rhs = np.ones(m)
                w, _, _, _ = np.linalg.lstsq(B, rhs, rcond=None)
                W[k*n+i, :] = w
        
        # Solve for U
        for j in range(m):
            for i in range(n):
                B = np.zeros((p, rank))
                for k in range(p):
                    for c in range(rank):
                        B[k, c] = V[j*p+k, c] * W[k*n+i, c]
                rhs = np.ones(p)
                u, _, _, _ = np.linalg.lstsq(B, rhs, rcond=None)
                U[i*m+j, :] = u
        
        # Solve for V
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    for c in range(rank):
                        B[i, c] = U[i*m+j, c] * W[k*n+i, c]
                rhs = np.ones(n)
                v, _, _, _ = np.linalg.lstsq(B, rhs, rcond=None)
                V[j*p+k, :] = v
        
        # Check convergence
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    max_err = max(max_err, abs(s - 1))
        
        if max_err < 1e-12:
            # Found a solution! Convert to exact fractions
            U_frac = [[Fraction(U[i, c]).limit_denominator(10**6) for c in range(rank)] for i in range(n*m)]
            V_frac = [[Fraction(V[i, c]).limit_denominator(10**6) for c in range(rank)] for i in range(m*p)]
            W_frac = [[Fraction(W[i, c]).limit_denominator(10**6) for c in range(rank)] for i in range(p*n)]
            
            if _verify(U_frac, V_frac, W_frac, n, m, p):
                return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try rank 54 first
    result = _try_rank_54(n, m, p, deadline)
    if result is not None:
        return result
    
    # Fall back to rank 55
    result = _construct_rank_55(n, m, p)
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    else:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
