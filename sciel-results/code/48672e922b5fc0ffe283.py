"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_half_integer(x, tol=1e-9):
    """Check if x is close to a half-integer."""
    half = round(x * 2) / 2.0
    return abs(x - half) < tol

def _snap_to_half_integer(x, tol=1e-9):
    """Snap a value to the nearest half-integer if close, else return None."""
    half = round(x * 2) / 2.0
    if abs(x - half) < tol:
        return half
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    np.random.seed(42)
    
    # Strategy: For fixed U and W, solve linear system for V
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # For fixed U,W: sum_c (U[i*m+j,c]*W[k*n+i,c]) * V[j*p+k,c] = 1
    # This is linear in V variables
    
    # Use half-integer random values to increase chance of rational solution
    half_int_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    for target_rank in range(20, 61):
        if time.time() > deadline - 20:
            break
        
        max_attempts = 200 if target_rank < 30 else 100
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            # Random U and W from half-integers
            U = np.random.choice(half_int_values, size=(n*m, target_rank))
            W = np.random.choice(half_int_values, size=(p*n, target_rank))
            
            # Set up linear system for V
            # Unknowns: V[j*p+k, c] for j in range(m), k in range(p), c in range(target_rank)
            # Number of unknowns: m*p*target_rank
            # Number of equations: n*m*p
            
            num_unknowns = m * p * target_rank
            num_eqs = n * m * p
            
            A_mat = np.zeros((num_eqs, num_unknowns))
            b_vec = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation: sum_c U[i*m+j,c] * W[k*n+i,c] * V[j*p+k,c] = 1
                        for c in range(target_rank):
                            col_idx = (j * p + k) * target_rank + c
                            A_mat[eq_idx, col_idx] = U[i*m+j, c] * W[k*n+i, c]
                        eq_idx += 1
            
            # Solve least squares
            try:
                V_flat, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            V = V_flat.reshape(m*p, target_rank)
            
            # Verify exactness of the real-valued solution
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                # Try to snap to half-integers
                U_snap = np.zeros_like(U)
                V_snap = np.zeros_like(V)
                W_snap = np.zeros_like(W)
                valid = True
                
                for i in range(U.shape[0]):
                    for j in range(U.shape[1]):
                        val = _snap_to_half_integer(U[i,j])
                        if val is None:
                            valid = False
                            break
                        U_snap[i,j] = val
                    if not valid: break
                
                if valid:
                    for i in range(V.shape[0]):
                        for j in range(V.shape[1]):
                            val = _snap_to_half_integer(V[i,j])
                            if val is None:
                                valid = False
                                break
                            V_snap[i,j] = val
                        if not valid: break
                
                if valid:
                    for i in range(W.shape[0]):
                        for j in range(W.shape[1]):
                            val = _snap_to_half_integer(W[i,j])
                            if val is None:
                                valid = False
                                break
                            W_snap[i,j] = val
                        if not valid: break
                
                if valid and _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
