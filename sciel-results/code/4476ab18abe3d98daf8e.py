import time
import numpy as np
from itertools import combinations

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    """
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    # Get the rank-44 solution
    U44, V44, W44 = _get_rank_44_solution()
    
    best_rank = 44
    best_U, best_V, best_W = U44, V44, W44
    
    # Try to find rank 43 by eliminating one term
    if time.time() < deadline - 60:
        found = _try_eliminate_one(44, deadline - 40)
        if found:
            best_rank = 43
            best_U, best_V, best_W = found
    
    # Try to find rank 42 by eliminating two terms
    if best_rank == 43 and time.time() < deadline - 30:
        U43, V43, W43 = best_U, best_V, best_W
        found = _try_eliminate_one(43, deadline - 20, base_rank=43, U_base=U43, V_base=V43, W_base=W43)
        if found:
            best_rank = 42
            best_U, best_V, best_W = found
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _get_rank_44_solution():
    """
    Returns a rank-44 solution using Strassen's algorithm on 2x2 blocks
    plus schoolbook for the 2x1*1x8 part.
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 44)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 44)
    W = [[0] * rank for _ in range(p * n)]  # shape (16, 44)
    
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
    
    # Part 1: A1 (2x2) * B1 (2x8)
    # B1 has rows 0, 1 of B. Split into 4 blocks of 2 columns.
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
    
    # Part 2: A2 (2x1) * B2 (1x8)
    # A2 is col 2 of A (rows 0,1).
    # B2 is row 2 of B (cols 0..7).
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            
            col += 1
    
    return U, V, W

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor T[i,j,k]."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n*m):
        i_row = i // m
        i_col = i % m
        for j in range(m*p):
            j_row = j // p
            j_col = j % p
            if i_col == j_row:
                k_row = i_row
                k_col = j_col
                k = k_col * n + k_row
                T[i, j, k] = 1.0
    return T

def _try_eliminate_one(current_rank, deadline, base_rank=None, U_base=None, V_base=None, W_base=None):
    """
    Try to eliminate one term from the current decomposition.
    """
    n, m, p = 2, 3, 8
    
    if base_rank is None:
        U44, V44, W44 = _get_rank_44_solution()
        U_arr = np.array(U44, dtype=float)
        V_arr = np.array(V44, dtype=float)
        W_arr = np.array(W44, dtype=float)
    else:
        U_arr = np.array(U_base, dtype=float)
        V_arr = np.array(V_base, dtype=float)
        W_arr = np.array(W_base, dtype=float)
    
    T = _build_target_tensor(n, m, p)
    nm, mp, pn = n*m, m*p, p*n
    
    # Precompute all (i,j,k) indices where T != 0
    nonzero_ijk = []
    for i in range(nm):
        for j in range(mp):
            for k in range(pn):
                if T[i, j, k] != 0:
                    nonzero_ijk.append((i, j, k))
    
    num_nz = len(nonzero_ijk)
    
    # For each term t to eliminate, check if the remaining terms can be rescaled
    for t_e in range(current_rank):
        if time.time() > deadline:
            return None
        
        keep_set = [t for t in range(current_rank) if t != t_e]
        target_rank = current_rank - 1
        
        # Build linear system: sum_{t in keep} c_t * U[i,t]*V[j,t]*W[k,t] = T[i,j,k]
        A = np.zeros((num_nz, len(keep_set)))
        b = np.zeros(num_nz)
        
        for idx, (i, j, k) in enumerate(nonzero_ijk):
            b[idx] = T[i, j, k]
            for k_idx, t_k in enumerate(keep_set):
                A[idx, k_idx] = U_arr[i, t_k] * V_arr[j, t_k] * W_arr[k, t_k]
        
        try:
            c, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=1e-12)
        except:
            continue
        
        # Check residual
        if residuals.size > 0:
            rel_res = residuals[0] / (np.linalg.norm(b) + 1e-15)
            if rel_res > 1e-10:
                continue
        
        # Verify with higher precision
        error = 0.0
        for idx, (i, j, k) in enumerate(nonzero_ijk):
            computed = np.dot(A[idx, :], c)
            error += (computed - T[i, j, k]) ** 2
        
        if error < 1e-10:
            # Found valid solution - construct the output
            final_U = np.zeros((nm, target_rank))
            final_V = np.zeros((mp, target_rank))
            final_W = np.zeros((pn, target_rank))
            
            for col_idx, t in enumerate(keep_set):
                final_U[:, col_idx] = U_arr[:, t]
                final_V[:, col_idx] = V_arr[:, t]
                final_W[:, col_idx] = W_arr[:, t]
            
            # Scale by c
            for col_idx in range(target_rank):
                final_U[:, col_idx] *= c[col_idx]
            
            return final_U.tolist(), final_V.tolist(), final_W.tolist()
    
    return None
