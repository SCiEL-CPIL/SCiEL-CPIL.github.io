"""Rewritten program: rank-minimal exact tensor decomposition for 2x5 * 5x6."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

_DEADLINE = 150.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k])  # Wait, this isn't right
                # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_half_integer_matrix(M, tol=1e-9):
    """Check if all entries are half-integers."""
    for row in M:
        for v in row:
            if abs(v * 2.0 - round(v * 2.0)) > tol:
                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Schoolbook decomposition
    U = [[0.0] * schoolbook_rank for _ in range(n * m)]
    V = [[0.0] * schoolbook_rank for _ in range(m * p)]
    W = [[0.0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    
    # Try to find a lower-rank decomposition using random search with local refinement
    # The problem is small enough that we can try several approaches
    
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    best_rank = schoolbook_rank
    
    # Lower bound: max(n*m, m*p, p*n) = 30
    min_rank = max(n*m, m*p, p*n)
    
    # Try ranks from min_rank to schoolbook_rank-1
    for target_rank in range(min_rank, schoolbook_rank):
        if _time_left() < 15:
            break
        
        # Try a few random initializations with local search
        for attempt in range(10):
            if _time_left() < 10:
                break
            
            random.seed(12345 + target_rank * 1000 + attempt)
            np.random.seed(12345 + target_rank * 1000 + attempt)
            
            # Initialize with small random values
            U_r = np.random.uniform(-0.5, 0.5, (n*m, target_rank))
            V_r = np.random.uniform(-0.5, 0.5, (m*p, target_rank))
            W_r = np.random.uniform(-0.5, 0.5, (p*n, target_rank))
            
            # Simple iterative refinement: try to reduce the error
            # Error = sum_{i,j,k} (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
            # Use a simple gradient-free approach: perturb and accept if error decreases
            
            def compute_error(Ur, Vr, Wr):
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += Ur[i*m+j, c] * Vr[j*p+k, c] * Wr[k*n+i, c]
                            err += (s - 1.0) ** 2
                return err
            
            current_error = compute_error(U_r, V_r, W_r)
            
            # Try a few iterations of random perturbation
            for it in range(100):
                if _time_left() < 5:
                    break
                
                # Pick a random matrix and entry to perturb
                mat_choice = random.randint(0, 2)
                row_idx = random.randint(0, (n*m if mat_choice == 0 else m*p if mat_choice == 1 else p*n) - 1)
                col_idx = random.randint(0, target_rank - 1)
                
                old_val = [U_r[row_idx, col_idx], V_r[row_idx, col_idx], W_r[row_idx, col_idx]][mat_choice]
                perturbation = random.uniform(-0.1, 0.1)
                
                if mat_choice == 0:
                    U_r[row_idx, col_idx] += perturbation
                elif mat_choice == 1:
                    V_r[row_idx, col_idx] += perturbation
                else:
                    W_r[row_idx, col_idx] += perturbation
                
                new_error = compute_error(U_r, V_r, W_r)
                
                if new_error < current_error:
                    current_error = new_error
                else:
                    # Revert
                    if mat_choice == 0:
                        U_r[row_idx, col_idx] = old_val
                    elif mat_choice == 1:
                        V_r[row_idx, col_idx] = old_val
                    else:
                        W_r[row_idx, col_idx] = old_val
            
            # Check if we found a valid solution
            if current_error < 1e-6:
                # Snap to half-integers and verify
                U_snap = np.round(U_r * 2.0) / 2.0
                V_snap = np.round(V_r * 2.0) / 2.0
                W_snap = np.round(W_r * 2.0) / 2.0
                
                if _verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                    best_U = U_snap.tolist()
                    best_V = V_snap.tolist()
                    best_W = W_snap.tolist()
                    best_rank = target_rank
                    break
        
        if best_rank == target_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
