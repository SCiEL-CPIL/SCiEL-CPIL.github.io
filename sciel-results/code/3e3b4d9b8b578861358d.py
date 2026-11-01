"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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


def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Tensor T[i,j,k] = 1 if i,j,k correspond to C[i,j] = A[i,:]*B[:,j]
    # A is n x m, B is m x p, C is n x p
    # T index: (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # For each output C[i,k], sum over j: A[i,j] * B[j,k]
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal delta(i,k) contribution
    # Actually, the decomposition is: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But that's not right. Let me think again.
    
    # Standard tensor decomposition for matrix multiplication:
    # T = sum_r U_r ⊗ V_r ⊗ W_r
    # where (U_r)_{i,j} * (V_r)_{j,k} * (W_r)_{k,i} = delta for the multiplication tensor
    
    # For C = A * B: C[i,k] = sum_j A[i,j] B[j,k]
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k (since every triple contributes)
    # Wait no. The tensor is T[i,j,k] = 1 if this is a valid multiplication step.
    # For matrix multiplication, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] B[j,k]
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(len(U[0])):
                    total += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _try_rank13():
    """Attempt to find a rank-13 decomposition via alternating minimization."""
    n, m, p = DIMENSIONS
    target_rank = 13
    
    # Precompute index mappings
    # U index: i*m + j for i in [0,n), j in [0,m)
    # V index: j*p + k for j in [0,m), k in [0,p)
    # W index: k*n + i for k in [0,p), i in [0,n)
    
    def _alternating_minimize(U_init, V_init, W_init, max_iter=500, tol=1e-10):
        """Alternating minimization: fix two factors, solve for the third."""
        n, m, p = DIMENSIONS
        rank = target_rank
        
        U = np.array(U_init, dtype=float)
        V = np.array(V_init, dtype=float)
        W = np.array(W_init, dtype=float)
        
        for iteration in range(max_iter):
            # Fix V and W, solve for U
            # For each (i,j): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # This gives a linear system in U[i*m+j, :]
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Build system: for each k, sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                    A_sys = np.zeros((p, rank))
                    for k in range(p):
                        A_sys[k, :] = V[j*p + k, :] * W[k*n + i, :]
                    b_sys = np.ones(p)
                    # Least squares solution
                    try:
                        sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
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
                        sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
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
                        sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        W_new[k*n + i, :] = sol
                    except np.linalg.LinAlgError:
                        return None
            W = W_new
            
            # Check convergence
            if np.max(np.abs(U - U_new)) < tol and \
               np.max(np.abs(V - V_new)) < tol and \
               np.max(np.abs(W - W_new)) < tol:
                break
        
        # Verify the solution
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                    if abs(total - 1.0) > 1e-8:
                        return None
        
        # Convert to integer/half-integer if possible
        def _to_rational(arr):
            """Try to convert float array to integer or half-integer values."""
            result = []
            for row in arr:
                new_row = []
                for val in row:
                    # Try integer
                    int_val = round(val)
                    if abs(val - int_val) < 1e-6:
                        new_row.append(int_val)
                    else:
                        # Try half-integer
                        half_val = round(val * 2) / 2.0
                        if abs(val - half_val) < 1e-6:
                            new_row.append(half_val)
                        else:
                            return None
                result.append(new_row)
            return result
        
        U_rat = _to_rational(U)
        V_rat = _to_rational(V)
        W_rat = _to_rational(W)
        
        if U_rat is None or V_rat is None or W_rat is None:
            return None
        
        return U_rat, V_rat, W_rat
    
    # Initialize with random values
    np.random.seed(42)
    best_result = None
    
    for trial in range(20):
        U_init = np.random.randn(n*m, target_rank)
        V_init = np.random.randn(m*p, target_rank)
        W_init = np.random.randn(p*n, target_rank)
        
        # Scale to reasonable magnitudes
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
    deadline = 140  # 120s limit + safety margin
    
    # Try to find rank-13 decomposition with time guard
    result_13 = None
    if time.time() - start_time < deadline - 5:
        result_13 = _try_rank13()
    
    if result_13 is not None:
        return {"U": result_13[0], "V": result_13[1], "W": result_13[2]}
    
    # Fall back to known rank-14 decomposition
    U, V, W = _build_rank14()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
