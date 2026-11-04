"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def find_rank7_mv():
        """Find a rank-7 decomposition for 3x3x1 matrix-vector multiplication."""
        # Target: C_i = sum_j A_ij v_j
        # Tensor T[i, j, 0] = delta_ij
        # We want U (3x7), V (3x7), W (3x7) such that sum_k U[i,k] V[j,k] W[0,k] = delta_ij
        
        def objective(x):
            U = x[:21].reshape(3, 7)
            V = x[21:42].reshape(3, 7)
            W = x[42:63].reshape(3, 7)
            
            # Compute T[i, j, 0] = sum_k U[i,k] V[j,k] W[0,k]
            # Note: W is 3x7, but the index for v is 0? 
            # In the tensor T[i, j, k], k is the index for v.
            # For 3x3x1, k is always 0.
            # So we use W[0, k] ? No, W has shape (1, 7) effectively?
            # The interface expects W of shape (p*n, rank) = (1*3, 7) = (3, 7).
            # The index for W is k*n + i. Here k=0, so index is i.
            # So W[i, k] corresponds to the coefficient for C_i.
            
            residual = np.zeros(9)
            idx = 0
            for i in range(3):
                for j in range(3):
                    val = np.sum(U[i, :] * V[j, :] * W[i, :])
                    target = 1.0 if i == j else 0.0
                    residual[idx] = val - target
                    idx += 1
            return residual

        best_err = float('inf')
        best_x = None
        
        for restart in range(20):
            np.random.seed(42 + restart)
            x0 = np.random.randn(63)
            try:
                result = least_squares(objective, x0, method='lm', max_nfev=5000)
                if result.cost < best_err:
                    best_err = result.cost
                    best_x = result.x
                    if best_err < 1e-10:
                        break
            except Exception:
                continue
        
        if best_err < 1e-6:
            U = best_x[:21].reshape(3, 7)
            V = best_x[21:42].reshape(3, 7)
            W = best_x[42:63].reshape(3, 7)
            # Round to nearest 0.5 or integer if possible
            # Check if rounding works
            for scale in [1, 2, 4, 8]:
                U_r = np.round(U * scale) / scale
                V_r = np.round(V * scale) / scale
                W_r = np.round(W * scale) / scale
                res = np.array([np.sum(U_r[i, :] * V_r[j, :] * W_r[i, :]) - (1.0 if i==j else 0.0) for i in range(3) for j in range(3)])
                if np.allclose(res, 0, atol=1e-6):
                    return U_r, V_r, W_r
            return U, V, W
        return None

    mv_decomp = find_rank7_mv()
    
    if mv_decomp is not None:
        U_mv, V_mv, W_mv = mv_decomp
        # Construct rank-42 decomposition for 3x3x6
        rank = 42
        U = np.zeros((9, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        
        for k in range(6):
            # Copy the 7 terms for column k
            start = k * 7
            end = start + 7
            
            # U: rows are i*m+j = i*3+j. 
            # In 3x3x1, U[i, t] is for A[i, j]? No.
            # In the 3x3x1 decomposition:
            # U[i, t] corresponds to A[i, j]? No, U is for A, V is for v, W is for C.
            # Wait, the tensor is T[i, j, k] where i is row of A, j is col of A/row of v, k is col of C (here k=0).
            # So U[i, t] is coefficient for A[i, j]? No.
            # The decomposition is T[i, j, k] = sum_t U[i, t] V[j, t] W[k, t].
            # Here i is row of A, j is col of A, k is index of v.
            # So U is 3x7, V is 3x7, W is 1x7 (or 3x7 if we consider W[k*n+i]?).
            # In the full 3x3x6 problem:
            # i (0..2), j (0..2), k (0..5).
            # U[i*m+j, t] -> U[i*3+j, t].
            # V[j*p+k, t] -> V[j*6+k, t].
            # W[k*n+i, t] -> W[k*3+i, t].
            
            # For the mv decomposition:
            # U_mv[i, t] -> A[i, j] coefficient? No, U_mv is 3x7.
            # In the mv problem, the indices are i (row of A), j (col of A), k (index of v, always 0).
            # So U_mv[i, t] corresponds to U[i*3+j, t] with j fixed? No.
            # The term is U[i, t] * V[j, t] * W[0, t].
            # In the full problem, for a specific column k:
            # We want to compute C[i, k] = sum_j A[i, j] B[j, k].
            # This is the same as mv multiplication for column k.
            # So we map:
            # U_full[i*3+j, t] = U_mv[i, t] ? No, U_mv depends on i and t, but not j?
            # Wait, in the mv decomposition, U is 3x7. It depends on i (row of A).
            # But A has 9 elements.
            # The decomposition T[i, j, 0] = sum_t U[i, t] V[j, t] W[0, t].
            # Here U[i, t] is associated with the row i of A?
            # No, the term is product of A[i, j], v[j], and output C[i].
            # So U[i, t] should be the coefficient for A[i, j]?
            # If U depends on i and t, and V depends on j and t, then U[i, t] V[j, t] is a term for A[i, j].
            # So U_full[i*3+j, t] should be related to U_mv[i, t] and V_mv[j, t]?
            # No, the factors are U, V, W.
            # U_full[i*3+j, t] = U_mv[i, t] ?
            # V_full[j*6+k, t] = V_mv[j, t] ?
            # W_full[k*3+i, t] = W_mv[i, t] ? (Since W_mv[i, t] was for C[i])
            
            # Let's check the indices.
            # In mv: T[i, j, 0] = sum_t U_mv[i, t] V_mv[j, t] W_mv[0, t].
            # Wait, W_mv was 3x7 in my solver, with W_mv[i, t] corresponding to C[i].
            # So W_mv[0, t] in the formula should be W_mv[i, t] for the i-th output.
            # So the term is U_mv[i, t] * V_mv[j, t] * W_mv[i, t].
            
            # In full problem for column k:
            # T[i, j, k] = sum_t U_full[i*3+j, t] V_full[j*6+k, t] W_full[k*3+i, t].
            # We want this to equal sum_t U_mv[i, t] V_mv[j, t] W_mv[i, t].
            
            # So we can set:
            # U_full[i*3+j, t] = U_mv[i, t]
            # V_full[j*6+k, t] = V_mv[j, t]
            # W_full[k*3+i, t] = W_mv[i, t]
            
            # Let's verify.
            # Product: U_mv[i, t] * V_mv[j, t] * W_mv[i, t].
            # This matches the mv decomposition.
            
            for i in range(3):
                for j in range(3):
                    for t in range(7):
                        idx = start + t
                        U[i*3+j, idx] = U_mv[i, t]
                        V[j*6+k, idx] = V_mv[j, t]
                        W[k*3+i, idx] = W_mv[i, t]
                        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
