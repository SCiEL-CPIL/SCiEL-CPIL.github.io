"""Combined program: Strassen blocks + improved rank-reduction for 2x2x15."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START

def _build_strassen_53(n, m, p):
    """Build rank-53: 7 Strassen blocks (rank 7 each) + 1 schoolbook (rank 4)."""
    blocks = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,)]
    
    all_U, all_V, all_W = [], [], []

    for block in blocks:
        if len(block) == 2:
            k0, k1 = block
            terms = _strassen_terms(k0, k1)
        else:
            k = block[0]
            terms = _schoolbook_terms(k)
        
        for (u_dict, v_dict, w_dict) in terms:
            u_entries = [(i*m+j, c) for (i,j), c in u_dict.items()]
            v_entries = [(j*p+k_idx, c) for (j,k_idx), c in v_dict.items()]
            w_entries = [(k_idx*n+i, c) for (k_idx,i), c in w_dict.items()]
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)

    rank = len(all_U)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for c in range(rank):
        for row, val in all_U[c]:
            U[row][c] = val
        for row, val in all_V[c]:
            V[row][c] = val
        for row, val in all_W[c]:
            W[row][c] = val

    return U, V, W


def _strassen_terms(k0, k1):
    """Strassen rank-7 terms for 2x2x2 block with k indices k0, k1."""
    return [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,k0):1, (1,k1):1}, {(k0,0):1, (k1,1):1}),
        # M2 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,k0):1}, {(k0,1):1, (k1,1):-1}),
        # M3 = A00(B01-B11)
        ({(0,0):1}, {(0,k1):1, (1,k1):-1}, {(k1,0):1, (k1,1):1}),
        # M4 = A11(B10-B00)
        ({(1,1):1}, {(1,k0):1, (0,k0):-1}, {(k0,0):1, (k0,1):1}),
        # M5 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,k1):1}, {(k0,0):-1, (k1,0):1}),
        # M6 = (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,k0):1, (0,k1):1}, {(k1,1):1}),
        # M7 = (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,k0):1, (1,k1):1}, {(k0,0):1}),
    ]


def _schoolbook_terms(k):
    """Schoolbook rank-4 terms for 2x2x1 block."""
    return [
        ({(0,0):1}, {(0,k):1}, {(k,0):1}),
        ({(0,1):1}, {(1,k):1}, {(k,0):1}),
        ({(1,0):1}, {(0,k):1}, {(k,1):1}),
        ({(1,1):1}, {(1,k):1}, {(k,1):1}),
    ]


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly matches matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                val = float(np.sum(U_arr[u_row, :] * V_arr[v_row, :] * W_arr[w_row, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_reduce_rank_advanced(U, V, W, n, m, p, deadline=None):
    """
    Advanced rank reduction: try to find columns that can be removed by
    checking if the tensor contribution is in the span of other contributions.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # For each column, check if its tensor contribution is a linear combination
    # of other columns' contributions
    for r in range(rank):
        if deadline and time.time() > deadline:
            break
        
        # Compute the tensor contribution of column r
        # T_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # We need to check if this is in the span of T_s for s != r
        
        # Build the system: for each (i,j,k), T_r[i,j,k] = sum_s c_s * T_s[i,j,k]
        # This is n*m*p equations in (rank-1) unknowns
        
        others = [s for s in range(rank) if s != r]
        if len(others) == 0:
            continue
        
        # Build matrix A where A[(i,j,k), s_idx] = T_s[i,j,k]
        # and b[(i,j,k)] = T_r[i,j,k]
        num_ijk = n * m * p
        A = np.zeros((num_ijk, len(others)))
        b = np.zeros(num_ijk)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = i * m + j
                    v_row = j * p + k
                    w_row = k * n + i
                    b[idx] = U_arr[u_row, r] * V_arr[v_row, r] * W_arr[w_row, r]
                    for s_idx, s in enumerate(others):
                        A[idx, s_idx] = U_arr[u_row, s] * V_arr[v_row, s] * W_arr[w_row, s]
                    idx += 1
        
        # Solve least squares
        try:
            c, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        # Check if the solution is exact
        residual = np.linalg.norm(A @ c - b)
        if residual < 1e-9:
            # Column r's contribution is a linear combination of others
            # Remove it and verify
            U_new = np.delete(U_arr, r, axis=1)
            V_new = np.delete(V_arr, r, axis=1)
            W_new = np.delete(W_arr, r, axis=1)
            
            if _verify(U_new, V_new, W_new, n, m, p):
                return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None


def _try_reduce_rank_iterative(U, V, W, n, m, p, max_iterations=3, deadline=None):
    """
    Iteratively try to reduce rank by finding removable columns.
    """
    U_cur, V_cur, W_cur = U, V, W
    for _ in range(max_iterations):
        if deadline and time.time() > deadline:
            break
        result = _try_reduce_rank_advanced(U_cur, V_cur, W_cur, n, m, p, deadline=deadline)
        if result is not None:
            U_cur, V_cur, W_cur = result
        else:
            break
    return U_cur, V_cur, W_cur


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Build rank-53 baseline
    U53, V53, W53 = _build_strassen_53(n, m, p)
    
    if not _verify(U53, V53, W53, n, m, p):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W = U53, V53, W53
    best_rank = 53
    
    # Try to reduce rank iteratively
    if time.time() < deadline - 5:
        reduced = _try_reduce_rank_iterative(U53, V53, W53, n, m, p, 
                                              max_iterations=3, deadline=deadline)
        if reduced is not None:
            rU, rV, rW = reduced
            if _verify(rU, rV, rW, n, m, p):
                new_rank = len(rU[0]) if isinstance(rU, list) else np.array(rU).shape[1]
                if new_rank < best_rank:
                    best_U, best_V, best_W = rU, rV, rW
                    best_rank = new_rank
    
    return {"U": best_U, "V": best_V, "W": best_W}


def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return U, V, W

# EVOLVE-BLOCK-END
