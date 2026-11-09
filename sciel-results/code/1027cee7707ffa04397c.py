import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _validate(U, V, W):
    """Check if U, V, W form a valid exact decomposition."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    recon = np.einsum('ac,bc,dc->abd', U_arr, V_arr, W_arr)
    
    if not np.allclose(recon, T, atol=1e-9):
        return False
    
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                v = float(val)
                if abs(v * 2 - round(v * 2)) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try to find a rank-21 solution
    target_rank = 21
    U = np.zeros((n * m, target_rank), dtype=float)
    V = np.zeros((m * p, target_rank), dtype=float)
    W = np.zeros((p * n, target_rank), dtype=float)
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    best_err = np.inf
    best_solution = None
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Random initialization
        for c in range(target_rank):
            for a in range(n * m):
                U[a, c] = random.choice([0, 0.5, 1, -0.5, -1])
            for b in range(m * p):
                V[b, c] = random.choice([0, 0.5, 1, -0.5, -1])
            for d in range(p * n):
                W[d, c] = random.choice([0, 0.5, 1, -0.5, -1])
        
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        err = np.sum((recon - T) ** 2)
        
        if err < best_err:
            best_err = err
            best_solution = (U.copy(), V.copy(), W.copy())
        
        # Local search
        for step in range(1000):
            if time.time() > deadline:
                break
                
            # Pick random entry
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                idx = random.randint(0, n * m - 1)
                col = random.randint(0, target_rank - 1)
                old_val = U[idx, col]
                new_val = random.choice([0, 0.5, 1, -0.5, -1])
                if new_val == old_val: continue
                U[idx, col] = new_val
            elif mat_idx == 1:
                idx = random.randint(0, m * p - 1)
                col = random.randint(0, target_rank - 1)
                old_val = V[idx, col]
                new_val = random.choice([0, 0.5, 1, -0.5, -1])
                if new_val == old_val: continue
                V[idx, col] = new_val
            else:
                idx = random.randint(0, p * n - 1)
                col = random.randint(0, target_rank - 1)
                old_val = W[idx, col]
                new_val = random.choice([0, 0.5, 1, -0.5, -1])
                if new_val == old_val: continue
                W[idx, col] = new_val
                
            recon = np.einsum('ac,bc,dc->abd', U, V, W)
            new_err = np.sum((recon - T) ** 2)
            
            if new_err < err:
                err = new_err
                if err < best_err:
                    best_err = err
                    best_solution = (U.copy(), V.copy(), W.copy())
                    if err < 1e-9:
                        # Found a solution!
                        U_frac = [[Fraction(U[a, c]).limit_denominator(2) for c in range(target_rank)] for a in range(n * m)]
                        V_frac = [[Fraction(V[b, c]).limit_denominator(2) for c in range(target_rank)] for b in range(m * p)]
                        W_frac = [[Fraction(W[d, c]).limit_denominator(2) for c in range(target_rank)] for d in range(p * n)]
                        if _validate(U_frac, V_frac, W_frac):
                            return {"U": U_frac, "V": V_frac, "W": W_frac}
            else:
                # Revert
                if mat_idx == 0:
                    U[idx, col] = old_val
                elif mat_idx == 1:
                    V[idx, col] = old_val
                else:
                    W[idx, col] = old_val
    
    # If no rank-21 solution found, return schoolbook (rank 42)
    return _schoolbook()
