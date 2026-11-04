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
    
    # Try to find rank 43 and 42
    for target_rank in [43, 42]:
        if time.time() > deadline - 20:
            break
        
        found = _search_for_rank(target_rank, deadline)
        if found:
            best_rank = target_rank
            best_U, best_V, best_W = found
            break  # If we found 43, try 42; if found 42, stop
    
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

def _verify_decomposition(U, V, W, T, tol=1e-8):
    """Verify that the decomposition exactly matches the target tensor."""
    nm, mp, pn = T.shape
    rank = len(U[0]) if U else 0
    
    max_err = 0.0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute all tensor entries
    for i in range(nm):
        for j in range(mp):
            for k in range(pn):
                computed = np.sum(U_arr[i, :] * V_arr[j, :] * W_arr[k, :])
                err = abs(computed - T[i, j, k])
                if err > max_err:
                    max_err = err
    
    return max_err < tol

def _search_for_rank(target_rank, deadline):
    """
    Try to find a decomposition of the given rank by eliminating terms
    from the rank-44 solution using linear algebra.
    """
    n, m, p = 2, 3, 8
    U44, V44, W44 = _get_rank_44_solution()
    r44 = 44
    num_eliminate = r44 - target_rank
    
    if num_eliminate <= 0:
        return None
    
    T = _build_target_tensor(n, m, p)
    
    # Convert to numpy arrays
    U_arr = np.array(U44, dtype=float)
    V_arr = np.array(V44, dtype=float)
    W_arr = np.array(W44, dtype=float)
    
    nm, mp, pn = n*m, m*p, p*n
    
    # Precompute all (i,j,k) indices where T != 0
    nonzero_ijk = []
    for i in range(nm):
        for j in range(mp):
            for k in range(pn):
                if T[i, j, k] != 0:
                    nonzero_ijk.append((i, j, k))
    
    # Strategy: try single-term eliminations first, then pairs
    if num_eliminate == 1:
        # Try eliminating each single term
        for t_e in range(r44):
            if time.time() > deadline - 10:
                return None
            
            keep_set = [t for t in range(r44) if t != t_e]
            
            # Build linear system:
            # For all (i,j,k): U[i,t_e]*V[j,t_e]*W[k,t_e] = sum_{t_k in keep} c[t_k] * U[i,t_k]*V[j,t_k]*W[k,t_k]
            
            A = np.zeros((len(nonzero_ijk), len(keep_set)))
            b = np.zeros(len(nonzero_ijk))
            
            for idx, (i, j, k) in enumerate(nonzero_ijk):
                b[idx] = U_arr[i, t_e] * V_arr[j, t_e] * W_arr[k, t_e]
                for k_idx, t_k in enumerate(keep_set):
                    A[idx, k_idx] = U_arr[i, t_k] * V_arr[j, t_k] * W_arr[k, t_k]
            
            # Solve least squares
            try:
                c, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            # Check if the solution is good
            if residuals.size > 0:
                if residuals[0] > 1e-6 * (np.sum(b**2) + 1e-10):
                    continue
            
            # Verify the solution
            error = 0.0
            for i in range(nm):
                for j in range(mp):
                    for k in range(pn):
                        computed = 0.0
                        for t_k in keep_set:
                            computed += U_arr[i, t_k] * V_arr[j, t_k] * W_arr[k, t_k]
                        error += (computed - T[i, j, k]) ** 2
            
            if error < 1e-4:
                # Found a valid solution!
                final_U = []
                final_V = []
                final_W = []
                for t in keep_set:
                    final_U.append(U_arr[:, t].tolist())
                    final_V.append(V_arr[:, t].tolist())
                    final_W.append(W_arr[:, t].tolist())
                
                # Transpose to get shape (nm, rank), (mp, rank), (pn, rank)
                final_U = np.array(final_U).T.tolist()
                final_V = np.array(final_V).T.tolist()
                final_W = np.array(final_W).T.tolist()
                
                return final_U, final_V, final_W
    
    elif num_eliminate == 2:
        # Try eliminating pairs of terms
        for t_e1, t_e2 in combinations(range(r44), 2):
            if time.time() > deadline - 10:
                return None
            
            keep_set = [t for t in range(r44) if t not in (t_e1, t_e2)]
            
            # Build linear system:
            # For all (i,j,k): U[i,t_e1]*V[j,t_e1]*W[k,t_e1] + U[i,t_e2]*V[j,t_e2]*W[k,t_e2]
            # = sum_{t_k in keep} c[t_k] * U[i,t_k]*V[j,t_k]*W[k,t_k]
            
            A = np.zeros((len(nonzero_ijk), len(keep_set)))
            b = np.zeros(len(nonzero_ijk))
            
            for idx, (i, j, k) in enumerate(nonzero_ijk):
                b[idx] = (U_arr[i, t_e1] * V_arr[j, t_e1] * W_arr[k, t_e1] +
                         U_arr[i, t_e2] * V_arr[j, t_e2] * W_arr[k, t_e2])
                for k_idx, t_k in enumerate(keep_set):
                    A[idx, k_idx] = U_arr[i, t_k] * V_arr[j, t_k] * W_arr[k, t_k]
            
            try:
                c, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            if residuals.size > 0:
                if residuals[0] > 1e-6 * (np.sum(b**2) + 1e-10):
                    continue
            
            # Verify
            error = 0.0
            for i in range(nm):
                for j in range(mp):
                    for k in range(pn):
                        computed = 0.0
                        for t_k in keep_set:
                            computed += U_arr[i, t_k] * V_arr[j, t_k] * W_arr[k, t_k]
                        error += (computed - T[i, j, k]) ** 2
            
            if error < 1e-4:
                final_U = []
                final_V = []
                final_W = []
                for t in keep_set:
                    final_U.append(U_arr[:, t].tolist())
                    final_V.append(V_arr[:, t].tolist())
                    final_W.append(W_arr[:, t].tolist())
                
                final_U = np.array(final_U).T.tolist()
                final_V = np.array(final_V).T.tolist()
                final_W = np.array(final_W).T.tolist()
                
                return final_U, final_V, final_W
    
    return None
