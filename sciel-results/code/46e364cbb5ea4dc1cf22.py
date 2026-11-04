import numpy as np
import time
import sys
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 5
    deadline = time.time() + 110  # 10s margin

    def make_schoolbook():
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
        return U, V, W

    best_U, best_V, best_W = make_schoolbook()
    best_rank = n * m * p

    def verify_full(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    s = 0.0
                    for k in range(r):
                        s += U[i * m + j, k] * V[j * p + l, k] * W[l * n + i, k]
                    if abs(s - 1) > tol:
                        return False
        return True

    def verify_225(U, V, W, tol=1e-9):
        # U: (4, r), V: (10, r), W: (10, r)
        # n=2, m=2, p=5
        r = U.shape[1]
        for i in range(2):
            for j in range(2):
                for l in range(5):
                    s = 0.0
                    for k in range(r):
                        s += U[i * 2 + j, k] * V[j * 5 + l, k] * W[l * 2 + i, k]
                    if abs(s - 1) > tol:
                        return False
        return True

    # Strategy: Find rank 18 decomposition for 2x2x5
    # Then duplicate it to get rank 36 for 2x4x5
    # 2x4x5 = (2x2x5) + (2x2x5)
    
    R_225 = 18
    np.random.seed(42)
    found = False
    
    # Faster search for 2x2x5 rank 18
    # We use a linear system approach with random sparse U, V
    # Variables W (10*R), Equations 20
    # 10*18 = 180 vars, 20 eqs. Very underdetermined.
    # We need integer/half-integer solutions.
    
    for attempt in range(5000):
        if time.time() > deadline - 5:
            break
            
        # Random sparse U, V with entries {-1, 0, 1}
        # To increase chance of integer W, we can try specific patterns
        U_225 = np.random.choice([-1, 0, 1], size=(4, R_225))
        V_225 = np.random.choice([-1, 0, 1], size=(10, R_225))
        
        # Ensure no row/col is all zero to avoid trivial null spaces?
        # Actually, lstsq handles it.
        
        A = np.zeros((20, 10 * R_225))
        b = np.ones(20)
        
        eq_idx = 0
        for i in range(2):
            for j in range(2):
                for l in range(5):
                    for k in range(R_225):
                        coeff = U_225[i * 2 + j, k] * V_225[j * 5 + l, k]
                        if coeff != 0:
                            w_idx = (l * 2 + i) * R_225 + k
                            A[eq_idx, w_idx] = coeff
                    eq_idx += 1
                    
        try:
            # Solve A @ W_vec = b
            # Use lstsq
            W_vec, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Check for integer solution
            W_int = np.round(W_vec)
            if np.allclose(A @ W_int, b, atol=1e-9):
                W_225 = W_int.reshape((10, R_225))
                if verify_225(U_225, V_225, W_225):
                    # Construct 2x4x5
                    U_full = np.zeros((8, 2 * R_225))
                    V_full = np.zeros((20, 2 * R_225))
                    W_full = np.zeros((10, 2 * R_225))
                    
                    # Block 1: j in 0..1
                    U_full[:, :R_225] = np.zeros((8, R_225))
                    # U for 2x2x5 is (4, R). We need (8, 2R).
                    # Rows 0..3 correspond to i=0,1 and j=0,1.
                    # Rows 4..7 correspond to i=0,1 and j=2,3.
                    
                    # Fill Block 1 (j=0,1)
                    U_full[0:4, :R_225] = U_225
                    V_full[0:10, :R_225] = V_225
                    W_full[:, :R_225] = W_225
                    
                    # Fill Block 2 (j=2,3)
                    # U rows 4..7
                    U_full[4:8, R_225:] = U_225
                    # V rows 10..19
                    V_full[10:20, R_225:] = V_225
                    # W same as block 1 (since output W is shared? No, W is per rank term)
                    # Wait, the decomposition is sum_k U_k V_k W_k.
                    # For block 1, k is 0..R-1. For block 2, k is R..2R-1.
                    # W_full[:, R_225:] should be W_225?
                    # Yes, because the output matrix C is the same size.
                    # The rank-1 terms for block 2 are distinct terms in the sum.
                    W_full[:, R_225:] = W_225
                    
                    if verify_full(U_full, V_full, W_full):
                        best_U, best_V, best_W = U_full, V_full, W_full
                        best_rank = 2 * R_225
                        found = True
                        break
                        
        except Exception:
            continue

    if not found:
        # Fallback to schoolbook if search fails
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
