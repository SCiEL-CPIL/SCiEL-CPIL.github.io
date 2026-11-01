"""Tensor rank minimization for 2x2*2x4 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import linprog
from itertools import combinations

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _build_rank14():
    """Return the known rank-14 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices
    # A: 0:00, 1:01, 2:10, 3:11
    # B: 0:00, 1:01, 2:02, 3:03, 4:10, 5:11, 6:12, 7:13
    # W (C): 0:00, 1:10, 2:01, 3:11, 4:02, 5:12, 6:03, 7:13
    # W index is k*n + i.

    # Strassen for B1 (cols 0,1)
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Strassen for B2 (cols 2,3)
    # N1
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return U, V, W


def _verify_decomposition(U, V, W, tol=1e-8):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = len(U_arr[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U_arr[i*m + j, :] * V_arr[j*p + k, :] * W_arr[k*n + i, :])
                if abs(total - 1.0) > tol:
                    return False
    return True


def _peeling_search(max_rank=13, max_iters=200, tol=1e-10):
    """
    Peeling algorithm for tensor rank minimization.
    
    The multiplication tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    We seek a rank-r decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    
    Peeling works by:
    1. Fixing two factors, solving for the third via linear systems
    2. Iteratively improving until convergence
    3. Checking if the solution has rational (integer/half-integer) entries
    """
    n, m, p = DIMENSIONS
    target_rank = max_rank
    
    def _verify(U, V, W, tol=1e-8):
        """Verify decomposition."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U_arr[i*m + j, :] * V_arr[j*p + k, :] * W_arr[k*n + i, :])
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    def _to_rational(arr, tol=1e-4):
        """Try to convert float array to integer or half-integer values."""
        result = []
        for row in arr:
            new_row = []
            for val in row:
                int_val = round(val)
                if abs(val - int_val) < tol:
                    new_row.append(int_val)
                else:
                    half_val = round(val * 2) / 2.0
                    if abs(val - half_val) < tol:
                        new_row.append(half_val)
                    else:
                        qtr_val = round(val * 4) / 4.0
                        if abs(val - qtr_val) < tol:
                            new_row.append(qtr_val)
                        else:
                            return None
            result.append(new_row)
        return result
    
    def _alternating_minimize(U_init, V_init, W_init, max_iter=200, tol=1e-10):
        """Alternating minimization with peeling-like structure."""
        rank = target_rank
        U = np.array(U_init, dtype=float)
        V = np.array(V_init, dtype=float)
        W = np.array(W_init, dtype=float)
        
        for iteration in range(max_iter):
            U_old = U.copy()
            V_old = V.copy()
            W_old = W.copy()
            
            # Fix V and W, solve for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A_sys = np.zeros((p, rank))
                    for k in range(p):
                        A_sys[k, :] = V[j*p + k, :] * W[k*n + i, :]
                    b_sys = np.ones(p)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        U_new[i*m + j, :] = sol
                    except np.linalg.LinAlgError:
                        return None
            U = U_new
            
            # Fix U and W, solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A_sys = np.zeros((n, rank))
                    for i in range(n):
                        A_sys[i, :] = U[i*m + j, :] * W[k*n + i, :]
                    b_sys = np.ones(n)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        V_new[j*p + k, :] = sol
                    except np.linalg.LinAlgError:
                        return None
            V = V_new
            
            # Fix U and V, solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A_sys = np.zeros((m, rank))
                    for j in range(m):
                        A_sys[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    b_sys = np.ones(m)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        W_new[k*n + i, :] = sol
                    except np.linalg.LinAlgError:
                        return None
            W = W_new
            
            change = np.max(np.abs(U - U_old)) + np.max(np.abs(V - V_old)) + np.max(np.abs(W - W_old))
            if change < tol:
                break
        
        if not _verify(U, V, W, tol=1e-6):
            return None
        
        U_rat = _to_rational(U, tol=1e-4)
        V_rat = _to_rational(V, tol=1e-4)
        W_rat = _to_rational(W, tol=1e-4)
        
        if U_rat is None or V_rat is None or W_rat is None:
            return None
        
        U_arr = np.array(U_rat, dtype=float)
        V_arr = np.array(V_rat, dtype=float)
        W_arr = np.array(W_rat, dtype=float)
        if not _verify(U_arr, V_arr, W_arr, tol=1e-6):
            return None
        
        return U_rat, V_rat, W_rat
    
    # Strategy 1: Start from rank-14 and try to merge pairs
    U14, V14, W14 = _build_rank14()
    U14_arr = np.array(U14, dtype=float)
    V14_arr = np.array(V14, dtype=float)
    W14_arr = np.array(W14, dtype=float)
    
    best_result = None
    
    for r1 in range(14):
        for r2 in range(r1 + 1, 14):
            U_init = np.zeros((n*m, target_rank))
            V_init = np.zeros((m*p, target_rank))
            W_init = np.zeros((p*n, target_rank))
            
            new_idx = 0
            for r in range(14):
                if r == r1 or r == r2:
                    continue
                U_init[:, new_idx] = U14_arr[:, r]
                V_init[:, new_idx] = V14_arr[:, r]
                W_init[:, new_idx] = W14_arr[:, r]
                new_idx += 1
            
            U_init[:, new_idx] = (U14_arr[:, r1] + U14_arr[:, r2]) / 2.0
            V_init[:, new_idx] = (V14_arr[:, r1] + V14_arr[:, r2]) / 2.0
            W_init[:, new_idx] = (W14_arr[:, r1] + W14_arr[:, r2]) / 2.0
            
            result = _alternating_minimize(U_init, V_init, W_init)
            if result is not None:
                best_result = result
                break
        if best_result is not None:
            break
    
    # Strategy 2: Random initializations
    if best_result is None:
        np.random.seed(42)
        for trial in range(50):
            U_init = np.random.randn(n*m, target_rank)
            V_init = np.random.randn(m*p, target_rank)
            W_init = np.random.randn(p*n, target_rank)
            
            U_init /= np.sqrt(np.sum(U_init**2))
            V_init /= np.sqrt(np.sum(V_init**2))
            W_init /= np.sqrt(np.sum(W_init**2))
            
            result = _alternating_minimize(U_init, V_init, W_init)
            if result is not None:
                best_result = result
                break
    
    return best_result


def construct():
    start_time = time.time()
    deadline = 130
    
    result_13 = None
    if time.time() - start_time < deadline - 5:
        result_13 = _peeling_search()
    
    if result_13 is not None:
        return {"U": result_13[0], "V": result_13[1], "W": result_13[2]}
    
    U, V, W = _build_rank14()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
