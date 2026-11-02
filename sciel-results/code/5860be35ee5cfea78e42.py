"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
import random

def _verify_exact(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The tensor T[a, b, c] = sum_r U[a,r]*V[b,r]*W[c,r] should equal 1 iff
    there exists a valid multiplication path, i.e., the standard matmul tensor.
    
    For matmul of (n x m) * (m x p):
    T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k).
    All other entries must be 0.
    """
    rank = len(U[0]) if U else 0
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
    
    # Check non-zero entries match expected pattern
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i*m+j, j*p+k, k*n+i] - 1.0) > 1e-12:
                    return False
    # Check all other entries are zero
    # Create mask of expected non-zero positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 0.0
    if np.max(np.abs(T)) > 1e-12:
        return False
    return True

def _to_fraction_list(arr):
    """Convert a numpy array to a list of lists of Fraction (or int if whole)."""
    result = []
    for row in arr:
        r = []
        for val in row:
            if abs(val - round(val)) < 1e-10:
                r.append(int(round(val)))
            else:
                # Try to find a simple rational
                frac = Fraction(val).limit_denominator(10000)
                r.append(frac)
        result.append(r)
    return result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Schoolbook rank as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # The tensor T has exactly n*m*p = 48 non-zero entries
    # We try to find a decomposition of rank < 48
    
    # Lower bound: max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # Try ranks from 47 down to 16
    
    best_result = {
        "U": _to_fraction_list(U_sb),
        "V": _to_fraction_list(V_sb),
        "W": _to_fraction_list(W_sb)
    }
    
    # For each candidate rank, do randomized search
    for rank in range(rank_sb - 1, 15, -1):
        if time.time() > deadline:
            break
        
        max_iters = 50 if rank > 30 else 200
        found = False
        
        for it in range(max_iters):
            if time.time() > deadline:
                found = True  # stop searching, use best
                break
            
            np.random.seed(None)
            
            # Random initialization with small values
            scale = 1.0
            U = np.random.choice([-scale, 0, scale], size=(n*m, rank))
            V = np.random.choice([-scale, 0, scale], size=(m*p, rank))
            W = np.random.choice([-scale, 0, scale], size=(p*n, rank))
            
            # Compute initial tensor
            T = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                T += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
            
            # Target tensor
            T_target = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i*m+j, j*p+k, k*n+i] = 1.0
            
            # Error
            err = np.sum((T - T_target)**2)
            
            if err < 1e-10:
                # Found exact solution!
                U_final = U.copy()
                V_final = V.copy()
                W_final = W.copy()
                # Round to nearest half-integer
                U_final = np.round(U_final * 2) / 2
                V_final = np.round(V_final * 2) / 2
                W_final = np.round(W_final * 2) / 2
                if _verify_exact(U_final, V_final, W_final, n, m, p):
                    best_result = {
                        "U": _to_fraction_list(U_final),
                        "V": _to_fraction_list(V_final),
                        "W": _to_fraction_list(W_final)
                    }
                    found = True
                    break
                continue
            
            # Local search / iterative refinement
            for step in range(200):
                if time.time() > deadline:
                    found = True
                    break
                
                # Pick random coordinate to perturb
                which = np.random.randint(3)
                if which == 0:
                    mat = U
                    rows = n*m
                elif which == 1:
                    mat = V
                    rows = m*p
                else:
                    mat = W
                    rows = p*n
                
                row_idx = np.random.randint(rows)
                col_idx = np.random.randint(rank)
                
                old_val = mat[row_idx, col_idx]
                new_val = old_val + np.random.choice([-1.0, -0.5, 0.5, 1.0])
                
                # Compute delta in T
                # Delta T[a,b,c] = (new_val - old_val) * product of other two factors at col_idx
                delta = new_val - old_val
                if which == 0:
                    # U changed: delta * V[b, col] * W[c, col]
                    delta_T = np.outer(np.ones(n*m)*delta, np.outer(V[:, col_idx], W[:, col_idx]))
                elif which == 1:
                    delta_T = np.outer(np.outer(U[:, col_idx], delta*np.ones(m*p)), W[:, col_idx])
                else:
                    delta_T = np.outer(np.outer(U[:, col_idx], V[:, col_idx]), delta*np.ones(p*n))
                
                new_T = T + delta_T
                new_err = np.sum((new_T - T_target)**2)
                
                if new_err < err:
                    mat[row_idx, col_idx] = new_val
                    T = new_T
                    err = new_err
                else:
                    # Occasionally accept to escape local minima
                    if np.random.random() < 0.05 and new_err < err * 1.5:
                        mat[row_idx, col_idx] = new_val
                        T = new_T
                        err = new_err
                
                if err < 1e-10:
                    U_final = U.copy()
                    V_final = V.copy()
                    W_final = W.copy()
                    U_final = np.round(U_final * 2) / 2
                    V_final = np.round(V_final * 2) / 2
                    W_final = np.round(W_final * 2) / 2
                    if _verify_exact(U_final, V_final, W_final, n, m, p):
                        best_result = {
                            "U": _to_fraction_list(U_final),
                            "V": _to_fraction_list(V_final),
                            "W": _to_fraction_list(W_final)
                        }
                        found = True
                        break
            if found:
                break
        
        if found:
            break
    
    # If we found a better rank, return it; otherwise return schoolbook
    return best_result
# EVOLVE-BLOCK-END
