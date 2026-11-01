import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 3]
_START_TIME = time.time()
TIME_LIMIT = 120  # seconds

def _time_remaining():
    return TIME_LIMIT - 10 - (time.time() - _START_TIME)

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W

def _build_strassen_2x2x2():
    """
    Build the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) as lists.
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # U: A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V: B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # W: C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] += 1
    U[3][0] += 1
    V[0][0] += 1
    V[3][0] += 1
    W[0][0] += 1
    W[3][0] += 1
    
    # M2: (A10+A11)B00
    U[2][1] += 1
    U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1
    W[3][1] -= 1
    
    # M3: A00(B01-B11)
    U[0][2] += 1
    V[1][2] += 1
    V[3][2] -= 1
    W[1][2] += 1
    W[3][2] += 1
    
    # M4: A11(B10-B00)
    U[3][3] += 1
    V[2][3] += 1
    V[0][3] -= 1
    W[0][3] += 1
    W[2][3] += 1
    
    # M5: (A00+A01)B11
    U[0][4] += 1
    U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1
    W[1][4] += 1
    
    # M6: (A10-A00)(B00+B01)
    U[2][5] += 1
    U[0][5] -= 1
    V[0][5] += 1
    V[1][5] += 1
    W[3][5] += 1
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] += 1
    U[3][6] -= 1
    V[2][6] += 1
    V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def _build_decomposed_2x3x3():
    """
    Decompose 2x3x3 into:
    - Part 1: 2x2x2 (columns j=0,1; rows k=0,1) -> Strassen rank 7
    - Part 2: 2x2x1 (columns j=0,1; row k=2) -> schoolbook rank 4
    - Part 3: 2x1x3 (column j=2; rows k=0,1,2) -> schoolbook rank 6
    Total rank: 17
    """
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x2x2 block
    U1, V1, W1 = _build_strassen_2x2x2()
    for r in range(7):
        for a_idx in range(4):
            i_local = a_idx // 2
            j_local = a_idx % 2
            a_global = i_local * 3 + j_local
            U[a_global][r] = U1[a_idx][r]
        for b_idx in range(4):
            j_local = b_idx // 2
            k_local = b_idx % 2
            b_global = j_local * 3 + k_local
            V[b_global][r] = V1[b_idx][r]
        for c_idx in range(4):
            k_local = c_idx // 2
            i_local = c_idx % 2
            c_global = k_local * 2 + i_local
            W[c_global][r] = W1[c_idx][r]
    
    # Part 2: 2x2x1 block (j in {0,1}, k=2)
    col_offset = 7
    for i in range(2):
        for j in range(2):
            col = col_offset + i * 2 + j
            a_global = i * 3 + j
            b_global = j * 3 + 2
            c_global = 4 + i
            U[a_global][col] = 1
            V[b_global][col] = 1
            W[c_global][col] = 1
    
    # Part 3: 2x1x3 block (j=2, k in {0,1,2})
    col_offset = 11
    for i in range(2):
        for k in range(3):
            col = col_offset + i * 3 + k
            a_global = i * 3 + 2
            b_global = 6 + k
            c_global = k * 2 + i
            U[a_global][col] = 1
            V[b_global][col] = 1
            W[c_global][col] = 1
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(val - 1) > tol:
                    return False
    return True

def _try_rank_reduction_by_linear_combination(n, m, p, initial_U, initial_V, initial_W):
    """
    Try to reduce rank by finding linear combinations of terms that can be merged.
    This is a heuristic approach: try to find if any two terms can be replaced
    by a single term that produces the same contribution to all tensor entries.
    """
    U_arr = np.array(initial_U, dtype=float).copy()
    V_arr = np.array(initial_V, dtype=float).copy()
    W_arr = np.array(initial_W, dtype=float).copy()
    rank = U_arr.shape[1]
    
    # For each pair of columns, check if they can be merged
    # Two rank-1 terms u1⊗v1⊗w1 and u2⊗v2⊗w2 can be merged if there exist
    # u, v, w such that u⊗v⊗w = u1⊗v1⊗w1 + u2⊗v2⊗w2
    # This is a very restrictive condition.
    
    # Instead, let's try a different approach: look for columns that are
    # linear combinations of other columns in a way that allows cancellation.
    
    # This is complex, so let's try a simpler heuristic:
    # Try to set one column to zero and see if we can adjust others to compensate.
    
    for col_to_remove in range(rank):
        if _time_remaining() < 5:
            break
        
        U_test = U_arr.copy()
        V_test = V_arr.copy()
        W_test = W_arr.copy()
        
        U_test[:, col_to_remove] = 0
        V_test[:, col_to_remove] = 0
        W_test[:, col_to_remove] = 0
        
        # Check if this still satisfies the tensor equations
        # (it won't in general, but let's verify)
        if _verify_fast(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p):
            # We found a lower-rank decomposition!
            # Remove the zero column
            U_new = np.delete(U_test, col_to_remove, axis=1)
            V_new = np.delete(V_test, col_to_remove, axis=1)
            W_new = np.delete(W_test, col_to_remove, axis=1)
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None

def _try_polynomial_optimization(n, m, p, target_rank):
    """
    Try to find a rank-target_rank decomposition by solving a system of equations.
    We parameterize the factor matrices with random initial values and use
    gradient descent to minimize the residual.
    """
    if _time_remaining() < 30:
        return None
    
    U_init = np.random.randn(n * m, target_rank) * 0.1
    V_init = np.random.randn(m * p, target_rank) * 0.1
    W_init = np.random.randn(p * n, target_rank) * 0.1
    
    # Use a simple iterative approach
    lr = 0.01
    for iteration in range(1000):
        if _time_remaining() < 10:
            break
        
        # Compute residual
        residual = 0.0
        grad_U = np.zeros_like(U_init)
        grad_V = np.zeros_like(V_init)
        grad_W = np.zeros_like(W_init)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_init[a], V_init[b] * W_init[c])
                    err = val - 1.0
                    residual += err * err
                    
                    # Gradient w.r.t. U[a, :]
                    grad_U[a] += err * V_init[b] * W_init[c]
                    # Gradient w.r.t. V[b, :]
                    grad_V[b] += err * U_init[a] * W_init[c]
                    # Gradient w.r.t. W[c, :]
                    grad_W[c] += err * U_init[a] * V_init[b]
        
        # Update
        U_init -= lr * grad_U
        V_init -= lr * grad_V
        W_init -= lr * grad_W
        
        if residual < 1e-12:
            break
    
    if residual < 1e-9:
        return U_init.tolist(), V_init.tolist(), W_init.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, try the known rank-17 decomposition
    try:
        U, V, W = _build_decomposed_2x3x3()
        if _verify_fast(U, V, W, n, m, p):
            best_U, best_V, best_W = U, V, W
            best_rank = len(U[0])
            
            # Try to reduce rank
            result = _try_rank_reduction_by_linear_combination(n, m, p, U, V, W)
            if result is not None:
                U_red, V_red, W_red = result
                if _verify_fast(U_red, V_red, W_red, n, m, p):
                    best_U, best_V, best_W = U_red, V_red, W_red
                    best_rank = len(U_red[0])
            
            # Try random optimization for rank 16
            if best_rank > 16:
                result = _try_polynomial_optimization(n, m, p, 16)
                if result is not None:
                    U_opt, V_opt, W_opt = result
                    if _verify_fast(U_opt, V_opt, W_opt, n, m, p):
                        best_U, best_V, best_W = U_opt, V_opt, W_opt
                        best_rank = 16
            
            return {"U": best_U, "V": best_V, "W": best_W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
