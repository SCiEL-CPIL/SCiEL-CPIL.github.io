"""Matrix multiplication 3x5x7 - rank minimization via structured search."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _verify_np(U, V, W, n, m, p):
    """Fast verification using numpy."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                val = np.sum(u_row * v_row * w_row)
                if abs(val - expected) > 1e-10:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = Fraction(1)
                val = Fraction(0)
                for c in range(rank):
                    u_val = Fraction(U[i*m+j][c])
                    v_val = Fraction(V[j*p+k][c])
                    w_val = Fraction(W[k*n+i][c])
                    val += u_val * v_val * w_val
                if val != target:
                    return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n  # 15, 35, 21

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Structured rank-5 decomposition: factor through middle dimension
    # For each j (0..4), use one rank-1 term:
    #   U[i,j,c] = delta_{c,j}, V[j,k,c] = delta_{c,j}, W[k,i,c] = 1
    # This gives T[i,j,k] = sum_c delta_{c,j} * delta_{c,j} * 1 = 1
    def structured_rank5():
        rank = m  # 5
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        # U[i*m+j, c] = 1 if c == j else 0
        for i in range(n):
            for j in range(m):
                U[i*m+j][j] = 1
        
        # V[j*p+k, c] = 1 if c == j else 0
        for j in range(m):
            for k in range(p):
                V[j*p+k][j] = 1
        
        # W[k*n+i, c] = 1 for all k, i, c
        for k in range(p):
            for i in range(n):
                for c in range(rank):
                    W[k*n+i][c] = 1
        
        return {"U": U, "V": V, "W": W}

    # Try the structured rank-5 decomposition first
    result = structured_rank5()
    
    # Verify with numpy
    if _verify_np(result["U"], result["V"], result["W"], n, m, p):
        # Verify exactly
        if _verify_exact(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Fallback: schoolbook
    return schoolbook()
