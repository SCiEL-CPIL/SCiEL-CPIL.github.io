import time
import sys
import random
import numpy as np
from itertools import combinations

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    """
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 110.0  # 40s margin before 150s kill
    
    # Known rank 44 solution - clean construction
    U44, V44, W44 = _get_rank_44_solution()
    
    best_rank = 44
    best_U, best_V, best_W = U44, V44, W44
    
    # Try to find rank 43 first (most likely improvement)
    if time.time() < deadline - 30:
        found = _search_for_rank(43, deadline - 20)
        if found:
            best_rank = 43
            best_U, best_V, best_W = found
    
    # If we found 43 and have time, try 42
    if best_rank == 43 and time.time() < deadline - 10:
        found = _search_for_rank(42, deadline - 5)
        if found:
            best_rank = 42
            best_U, best_V, best_W = found
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _get_rank_44_solution():
    """
    Clean rank-44 construction using Strassen for 2x2*2x8 and schoolbook for 2x1*1x8.
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    # Strassen 2x2x2 coefficients
    S_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    
    S_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    
    S_W = [
        [1, 0, 0, 1],   # M1
        [0, 1, 0, -1],  # M2
        [0, 0, 1, 1],   # M3
        [1, 1, 0, 0],   # M4
        [-1, 0, 1, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8) - split B1 into four 2x2 blocks
    for block_idx in range(4):
        k_start = block_idx * 2
        
        for r in range(7):
            # Fill U for A1 (rows 0,1 of A, cols 0,1 of A)
            for li in range(2):
                for la in range(2):
                    val = S_U[r][li * 2 + la]
                    if val != 0:
                        u_idx = li * m + la
                        U[u_idx][col] += val
            
            # Fill V for B1 block (rows 0,1 of B, cols k_start..k_start+1)
            for la in range(2):
                for lk in range(2):
                    val = S_V[r][la * 2 + lk]
                    if val != 0:
                        v_idx = la * p + (k_start + lk)
                        V[v_idx][col] += val
            
            # Fill W for C block (cols k_start..k_start+1 of C, rows 0..1 of C)
            for lk in range(2):
                for li in range(2):
                    val = S_W[r][lk * 2 + li]
                    if val != 0:
                        w_idx = (k_start + lk) * n + li
                        W[w_idx][col] += val
            
            col += 1
            
    # Part 2: A2 (2x1) * B2 (1x8) - outer product
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            
            col += 1
    
    # Validate
    valid = True
    for i in range(n):
        for a in range(m):
            for k in range(p):
                acc = 0
                u_row = U[i * m + a]
                v_row = V[a * p + k]
                w_row = W[k * n + i]
                for c in range(rank):
                    if u_row[c] != 0 and v_row[c] != 0 and w_row[c] != 0:
                        acc += u_row[c] * v_row[c] * w_row[c]
                if abs(acc - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return U, V, W
    else:
        # Fallback: schoolbook
        return _get_schoolbook_solution()

def _get_schoolbook_solution():
    """Fallback schoolbook solution (rank 48)."""
    n, m, p = 2, 3, 8
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for a in range(m):
            for k in range(p):
                U[i * m + a][col] = 1
                V[a * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _search_for_rank(target_rank, deadline):
    """
    Try to find a decomposition of the given rank by starting from the rank-44 solution
    and attempting to eliminate terms.
    """
    n, m, p = 2, 3, 8
    U44, V44, W44 = _get_rank_44_solution()
    r44 = 44
    num_eliminate = r44 - target_rank
    
    if num_eliminate <= 0:
        return None
    
    # Precompute as numpy arrays
    U_arr = np.array(U44, dtype=float)
    V_arr = np.array(V44, dtype=float)
    W_arr = np.array(W44, dtype=float)
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        for j in range(m*p):
            if i // m == j % p:
                k = (j // p) * n + (i % m)
                T[i, j, k] = 1
    
    # For each choice of terms to eliminate, try to find a valid decomposition
    for elim_set in combinations(range(r44), num_eliminate):
        if time.time() > deadline:
            return None
        
        keep_set = [t for t in range(r44) if t not in elim_set]
        
        # For each eliminated term, find coefficients to express it in terms of kept terms
        new_U = U_arr.copy()
        new_V = V_arr.copy()
        new_W = W_arr.copy()
        
        valid = True
        for t_e in elim_set:
            # Build linear system for this eliminated term
            num_keep = len(keep_set)
            
            # Collect equations where the eliminated term has non-zero contribution
            A_rows = []
            b_vals = []
            
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        lhs = U_arr[i, t_e] * V_arr[j, t_e] * W_arr[k, t_e]
                        if abs(lhs) < 1e-12:
                            continue
                        row = []
                        for t_k in keep_set:
                            row.append(U_arr[i, t_k] * V_arr[j, t_k] * W_arr[k, t_k])
                        A_rows.append(row)
                        b_vals.append(lhs)
            
            if not A_rows:
                valid = False
                break
            
            A = np.array(A_rows)
            b = np.array(b_vals)
            
            # Solve least squares
            try:
                c, residuals, rank_val, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                valid = False
                break
            
            # Check if solution is good
            if residuals.size > 0:
                if residuals[0] > 1e-8 * (np.sum(b**2) + 1e-10):
                    valid = False
                    break
            
            # Add contributions to kept terms
            for idx, t_k in enumerate(keep_set):
                if abs(c[idx]) > 1e-10:
                    new_U[:, t_k] += c[idx] * new_U[:, t_e]
                    new_V[:, t_k] += c[idx] * new_V[:, t_e]
                    new_W[:, t_k] += c[idx] * new_W[:, t_e]
            
            # Zero out eliminated term
            new_U[:, t_e] = 0
            new_V[:, t_e] = 0
            new_W[:, t_e] = 0
        
        if not valid:
            continue
        
        # Verify the solution
        error = 0.0
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    computed = 0
                    for t in keep_set:
                        computed += new_U[i, t] * new_V[j, t] * new_W[k, t]
                    error += (computed - T[i, j, k]) ** 2
        
        if error < 1e-6:
            # Found valid solution!
            final_U = []
            final_V = []
            final_W = []
            for t in keep_set:
                final_U.append(new_U[:, t].tolist())
                final_V.append(new_V[:, t].tolist())
                final_W.append(new_W[:, t].tolist())
            
            final_U = np.array(final_U).T.tolist()
            final_V = np.array(final_V).T.tolist()
            final_W = np.array(final_W).T.tolist()
            
            return final_U, final_V, final_W
    
    return None
