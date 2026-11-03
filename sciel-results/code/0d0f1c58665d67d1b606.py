import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin
    
    # Try to find a decomposition for M_{2,3,6} directly with low rank
    # Lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # Schoolbook rank: n*m*p = 36
    
    # Strategy:
    # 1. Try structured decompositions for various ranks
    # 2. Use systematic search over small integer/half-integer values
    # 3. Fallback to schoolbook
    
    best_rank = n * m * p  # schoolbook
    best_result = _schoolbook(n, m, p)
    
    # Try ranks from lower bound up to schoolbook
    lower_bound = max(n*m, m*p, p*n)
    
    for r in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition of rank r
        result = _search_rank(n, m, p, r, deadline)
        if result is not None:
            best_rank = r
            best_result = result
            break  # We found the minimal rank, stop
    
    return best_result

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition (rank = n*m*p)"""
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

def _search_rank(n, m, p, r, deadline):
    """
    Search for a rank-r decomposition of the matrix multiplication tensor.
    
    The tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    Strategy:
    - Fix U and W, then solve for V using linear algebra
    - For each (j,k), we have a linear system: sum_c U[i*m+j,c]*W[k*n+i,c]*V[j*p+k,c] = 1 for all i
    - This gives n equations in r unknowns (V[j*p+k, 0..r-1])
    - We need the solution to be integer or half-integer
    """
    
    # Try structured patterns first
    result = _try_structured_patterns(n, m, p, r)
    if result is not None:
        return result
    
    # Random search with structured initialization
    val_range = 3  # values in {-3, -2, -1, 0, 1, 2, 3}
    
    for seed in range(200):
        if time.time() > deadline - 5:
            break
        
        random.seed(seed)
        
        # Generate random U and W with small integer entries
        U = np.zeros((n * m, r), dtype=float)
        W = np.zeros((p * n, r), dtype=float)
        
        # Use a mix of 0, 1, -1, 2, -2 for better conditioning
        for idx in range(n * m):
            for c in range(r):
                U[idx, c] = random.choice([-2, -1, 0, 0, 0, 1, 1, 2])
        
        for idx in range(p * n):
            for c in range(r):
                W[idx, c] = random.choice([-2, -1, 0, 0, 0, 1, 1, 2])
        
        # Solve for V
        V = np.zeros((m * p, r), dtype=float)
        valid = True
        
        for j in range(m):
            for k in range(p):
                # Build the linear system for V[j*p+k, :]
                # For each i: sum_c U[i*m+j, c] * W[k*n+i, c] * V[j*p+k, c] = 1
                A_sys = np.zeros((n, r), dtype=float)
                for i in range(n):
                    for c in range(r):
                        A_sys[i, c] = U[i * m + j, c] * W[k * n + i, c]
                
                # Solve A_sys @ V_col = ones(n)
                try:
                    if np.linalg.matrix_rank(A_sys) < n:
                        valid = False
                        break
                    V_col, residuals, rank, sv = np.linalg.lstsq(A_sys, np.ones(n), rcond=None)
                except:
                    valid = False
                    break
                
                # Check if V_col has integer or half-integer entries
                V_col_check = V_col * 2
                if not np.allclose(V_col_check, np.round(V_col_check), atol=1e-8):
                    valid = False
                    break
                
                V[j * p + k, :] = V_col
            
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify the solution
        if _verify_decomposition(U, V, W, n, m, p, r):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None

def _verify_decomposition(U, V, W, n, m, p, r):
    """Verify that the decomposition is exact"""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(r):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if not np.isclose(val, 1, atol=1e-8):
                    return False
    return True

def _try_structured_patterns(n, m, p, r):
    """Try known structured decompositions"""
    
    # Pattern 1: For small cases, try specific known patterns
    
    # For M_{2,3,6}, try to use the fact that 6 = 3*2
    # Decompose into 3 copies of M_{2,3,2}
    if n == 2 and m == 3 and p == 6:
        # Try to find rank for M_{2,3,2} first
        for sub_rank in range(6, r // 3 + 1):
            if sub_rank * 3 > r:
                break
            
            result = _try_sub_decomposition(2, 3, 2, sub_rank)
            if result is not None:
                # Combine 3 copies
                U_sub, V_sub, W_sub = result
                full_rank = sub_rank * 3
                if full_rank > r:
                    continue
                
                U_full = [[0] * full_rank for _ in range(n * m)]
                V_full = [[0] * full_rank for _ in range(m * p)]
                W_full = [[0] * full_rank for _ in range(p * n)]
                
                for rep in range(3):
                    offset = rep * sub_rank
                    for i in range(2):
                        for j in range(3):
                            u_idx = i * 3 + j
                            for c in range(sub_rank):
                                U_full[i * 3 + j][offset + c] = U_sub[u_idx][c]
                    
                    for j in range(3):
                        for k_sub in range(2):
                            v_idx = j * 2 + k_sub
                            k_full = rep * 2 + k_sub
                            for c in range(sub_rank):
                                V_full[j * 6 + k_full][offset + c] = V_sub[v_idx][c]
                    
                    for k_sub in range(2):
                        for i in range(2):
                            w_idx = k_sub * 2 + i
                            k_full = rep * 2 + k_sub
                            for c in range(sub_rank):
                                W_full[k_full * 2 + i][offset + c] = W_sub[w_idx][c]
                
                if _verify_decomposition(np.array(U_full), np.array(V_full), np.array(W_full), n, m, p, full_rank):
                    return {"U": U_full, "V": V_full, "W": W_full}
    
    return None

def _try_sub_decomposition(n, m, p, r):
    """Try to find a decomposition for a smaller sub-problem"""
    
    # For M_{2,3,2}, try specific patterns
    if n == 2 and m == 3 and p == 2:
        # Try rank 6 (schoolbook for sub-problem)
        if r >= 6:
            U = np.zeros((6, r), dtype=float)
            V = np.zeros((6, r), dtype=float)
            W = np.zeros((4, r), dtype=float)
            
            col = 0
            for i in range(2):
                for j in range(3):
                    for k in range(2):
                        U[i * 3 + j, col] = 1
                        V[j * 2 + k, col] = 1
                        W[k * 2 + i, col] = 1
                        col += 1
                        if col >= r:
                            break
                    if col >= r:
                        break
                if col >= r:
                    break
            
            if col == 6 and r == 6:
                return (U.tolist(), V.tolist(), W.tolist())
        
        # Try rank 7, 8, etc. with random search
        for seed in range(50):
            random.seed(seed)
            U = np.array([[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(6)], dtype=float)
            W = np.array([[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(4)], dtype=float)
            
            V = np.zeros((6, r), dtype=float)
            valid = True
            
            for j in range(3):
                for k in range(2):
                    A_sys = np.zeros((2, r), dtype=float)
                    for i in range(2):
                        for c in range(r):
                            A_sys[i, c] = U[i * 3 + j, c] * W[k * 2 + i, c]
                    
                    try:
                        V_col = np.linalg.lstsq(A_sys, np.ones(2), rcond=None)[0]
                    except:
                        valid = False
                        break
                    
                    V_col_check = V_col * 2
                    if not np.allclose(V_col_check, np.round(V_col_check), atol=1e-8):
                        valid = False
                        break
                    
                    V[j * 2 + k, :] = V_col
                
                if not valid:
                    break
            
            if valid:
                if _verify_decomposition(U, V, W, 2, 3, 2, r):
                    return (U.tolist(), V.tolist(), W.tolist())
    
    return None
