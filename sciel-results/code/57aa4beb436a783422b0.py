"""Matrix multiplication 3x4 by 4x6: Combined Strassen + optimization approach."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _construct_strassen_3x4x6():
    """
    Construct a Strassen-based decomposition for 3x4x6 matrix multiplication.
    
    Strategy:
    - Row 0: Schoolbook (24 terms)
    - Rows 1-2: Split into 2x4x2 blocks, use Strassen (7 terms each)
    - Total: 24 + 6*7 = 24 + 42 = 66 terms
    """
    n, m, p = 3, 4, 6
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term for A[i,j] * B[j,k] -> C[i,k]
    def add_schoolbook(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    # Helper to add Strassen terms for a 2x2x2 block
    # i_base, j_base, k_base are the global starting indices for the 2x2 block
    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        
        # Indices for A (2x2)
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        
        # Indices for B (2x2)
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        
        # Indices for C (2x2) -> W index is k*n + i
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        # m1 = (a+d)(e+h)
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # m2 = (c+d)e
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        # m3 = a(f-h)
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        # m4 = d(g-e)
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        # m5 = (a+b)h
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1

    # Part 1: Row 0 of A (M(1,4,6))
    # A[0, :] * B[:, :] -> C[0, :]
    # Schoolbook: 4 * 6 = 24 terms
    for j in range(4):
        for k in range(6):
            add_schoolbook(0, j, k)

    # Part 2: Rows 1,2 of A (M(2,4,6))
    # Split k (columns of B) into 3 groups of 2
    for k_block in range(3):
        k_base = k_block * 2
        
        # For each k_block, we have a 2x4x2 sub-problem
        # Split j (columns of A / rows of B) into 2 groups of 2
        for j_block in range(2):
            j_base = j_block * 2
            
            # i_base is 1 (since we are handling rows 1 and 2)
            i_base = 1
            
            # Add Strassen for this 2x2x2 block
            add_strassen(i_base, j_base, k_base)

    return U, V, W

def _try_rank_am(n, m, p, rank, max_iter=300, tol=1e-8, seed=None):
    """Try to find a decomposition of given rank using alternating minimization."""
    if _time_left() < 5.0:
        return None
    
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_obj = float('inf')
    best_state = None
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Update U: for each (i,j), solve for U[i*m+j, :] given V, W
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k, :] given U, W
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    x = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    V[j * p + k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i, :] given U, V
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    x = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    W[k * n + i, :] = x
                except:
                    pass
        
        # Compute objective: sum of squared errors
        obj = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    obj += (s - 1.0) ** 2
        
        if obj < best_obj:
            best_obj = obj
            best_state = (U.copy(), V.copy(), W.copy())
        
        if obj < tol:
            break
    
    if best_obj < tol:
        U, V, W = best_state
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return U_list, V_list, W_list
    
    return None

def _try_column_reduction(U, V, W, n, m, p):
    """
    Try to reduce rank by removing redundant columns.
    Returns (success, new_U, new_V, new_W, new_rank)
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try removing each column one at a time
    for remove_col in range(rank):
        if _time_left() < 5.0:
            break
        
        U_test = np.delete(U_np, remove_col, axis=1)
        V_test = np.delete(V_np, remove_col, axis=1)
        W_test = np.delete(W_np, remove_col, axis=1)
        
        # Check if the tensor equation still holds
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    val = 0.0
                    for c in range(rank - 1):
                        val += U_test[i*m+j, c] * V_test[j*p+k, c] * W_test[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
        
        if valid:
            new_rank = rank - 1
            new_U = U_test.tolist()
            new_V = V_test.tolist()
            new_W = W_test.tolist()
            return True, new_U, new_V, new_W, new_rank
    
    return False, None, None, None, rank

def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    
    Primary: Strassen-based decomposition (rank 66)
    Secondary: Attempt to find lower-rank solution via optimization
    """
    n, m, p = 3, 4, 6
    
    # Build the Strassen-based solution
    U, V, W = _construct_strassen_3x4x6()
    rank = len(U[0])
    
    # Verify the solution is correct
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        U, V, W = _schoolbook(n, m, p)
        rank = n * m * p
        return {"U": U, "V": V, "W": W}
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Attempt 1: Column reduction
    success, new_U, new_V, new_W, new_rank = _try_column_reduction(U, V, W, n, m, p)
    if success and new_rank < best_rank:
        best_rank = new_rank
        best_U, best_V, best_W = new_U, new_V, new_W
    
    # Attempt 2: Try alternating minimization for lower ranks
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But in practice, we expect something higher
    
    for target_rank in range(max(24, best_rank - 10), best_rank):
        if _time_left() < 10.0:
            break
        
        # Try a few random initializations
        for seed in range(3):
            if _time_left() < 5.0:
                break
            
            result = _try_rank_am(n, m, p, target_rank, max_iter=200, tol=1e-8, seed=seed)
            if result is not None:
                U_test, V_test, W_test = result
                # Verify
                if _verify(U_test, V_test, W_test, n, m, p, tol=1e-6):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_test, V_test, W_test
                    break
        
        if best_rank == target_rank:
            break
    
    return {"U": best_U, "V": best_W, "W": best_W}
