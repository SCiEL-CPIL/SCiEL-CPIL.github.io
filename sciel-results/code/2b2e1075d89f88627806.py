"""Tensor rank minimization for 2x5 * 5x6 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _make_schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check that the decomposition exactly reproduces the matrix multiplication tensor."""
    # The tensor T[i,j,k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Actually, the standard bilinear algorithm form is:
    # C[i,k] = sum_c M_c[i] * N_c[k]
    # where M_c = U[:,c] * A and N_c = B * V[:,c]
    # The constraint is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Let's check this directly
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                max_err = max(max_err, abs(val - 1.0))
    return max_err < tol

def _round_to_half_int(arr, tol=1e-6):
    """Round array entries to nearest half-integer, check residual."""
    rounded = np.round(arr * 2.0) / 2.0
    residual = np.abs(arr - rounded).max()
    return rounded, residual

def _try_optimize(n, m, p, rank, max_time=60.0):
    """Try to find a rank-rank decomposition via iterative refinement."""
    start_time = time.time()
    
    # Initialize with random matrices
    np.random.seed(42)
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Build the tensor constraint matrix
    # We need: for all (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is a system of n*m*p equations in rank unknowns (3*rank variables total)
    # Use alternating least squares
    
    for it in range(5000):
        if time.time() - start_time > max_time:
            break
        
        # Solve for U given V, W
        # For each i,j: sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c] summed over k... no
        # Actually, the constraint is:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        # This is nonlinear. Let's use a different approach.
        
        # Alternative: use the fact that for fixed i, the equation is:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # Let's define for each (i,j,k): A_{ijk,c} = V[j*p+k, c] * W[k*n+i, c]
        # Then we need sum_c U[i*m+j, c] * A_{ijk,c} = 1
        # This is a linear system in U for fixed i: for each j,k
        # But U[i*m+j, c] appears in equations for fixed i, varying j,k
        # For fixed i: we have m*p equations, m*rank unknowns (U[i*m+0..m-1, :])
        # A_{ijk,c} depends on j,k,c
        
        # Let's set up the linear system for U given V,W:
        # For fixed i, the unknowns are U[i*m+j, c] for j in range(m), c in range(rank)
        # Equations: for j in range(m), k in range(p):
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # This is m*p equations in m*rank unknowns.
        # If m*rank > m*p, it's underdetermined; we want minimum norm solution.
        
        # Build the system
        A_mat = np.zeros((m * p, m * rank))
        b_vec = np.ones(m * p)
        for j in range(m):
            for k in range(p):
                eq_idx = j * p + k
                for c in range(rank):
                    A_mat[eq_idx, j * rank + c] = V[j * p + k, c] * W[k * n + i, c]
        
        # Solve minimum norm: U_sub = A^T (A A^T)^{-1} b
        try:
            AA_T = A_mat @ A_mat.T
            if np.linalg.matrix_rank(AA_T) < m * p:
                continue
            U_sub = A_mat.T @ np.linalg.solve(AA_T, b_vec)
        except:
            continue
        
        U[i * m:(i + 1) * m, :] = U_sub.reshape(m, rank)
        
        # Similarly solve for V given U, W
        # For fixed j: sum_c V[j*p+k, c] * U[i*m+j, c] * W[k*n+i, c] = 1 for all i,k
        # Unknowns: V[j*p+k, c] for k in range(p), c in range(rank) -> p*rank unknowns
        # Equations: n*p equations
        for j in range(m):
            A_mat_v = np.zeros((n * p, p * rank))
            b_v = np.ones(n * p)
            for i in range(n):
                for k in range(p):
                    eq_idx = i * p + k
                    for c in range(rank):
                        A_mat_v[eq_idx, k * rank + c] = U[i * m + j, c] * W[k * n + i, c]
            try:
                AA_T_v = A_mat_v @ A_mat_v.T
                if np.linalg.matrix_rank(AA_T_v) < n * p:
                    continue
                V_sub = A_mat_v.T @ np.linalg.solve(AA_T_v, b_v)
                V[j * p:(j + 1) * p, :] = V_sub.reshape(p, rank)
            except:
                continue
        
        # Solve for W given U, V
        # For fixed k: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1 for all i,j
        # Unknowns: W[k*n+i, c] for i in range(n), c in range(rank) -> n*rank unknowns
        # Equations: n*m equations
        for k in range(p):
            A_mat_w = np.zeros((n * m, n * rank))
            b_w = np.ones(n * m)
            for i in range(n):
                for j in range(m):
                    eq_idx = i * m + j
                    for c in range(rank):
                        A_mat_w[eq_idx, i * rank + c] = U[i * m + j, c] * V[j * p + k, c]
            try:
                AA_T_w = A_mat_w @ A_mat_w.T
                if np.linalg.matrix_rank(AA_T_w) < n * m:
                    continue
                W_sub = A_mat_w.T @ np.linalg.solve(AA_T_w, b_w)
                W[k * n:(k + 1) * n, :] = W_sub.reshape(n, rank)
            except:
                continue
        
        # Check convergence
        if it % 50 == 0:
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(val - 1.0))
            if max_err < 1e-10:
                break
    
    # Check if we have a valid solution
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        # Try to round to half-integers
        U_r, rU = _round_to_half_int(U)
        V_r, rV = _round_to_half_int(V)
        W_r, rW = _round_to_half_int(W)
        if rU < 1e-6 and rV < 1e-6 and rW < 1e-6:
            if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-8):
                return U_r, V_r, W_r
        # If rounding fails, check if the solution is close to half-integers
        # Try snapping to nearest half-integer and rechecking
        U_s = np.round(U * 2.0) / 2.0
        V_s = np.round(V * 2.0) / 2.0
        W_s = np.round(W * 2.0) / 2.0
        if _verify_decomposition(U_s, V_s, W_s, n, m, p, tol=1e-6):
            return U_s, V_s, W_s
        # Return the floating point solution if it's valid
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0  # Safety margin
    
    # Start with schoolbook rank 60
    U_sb, V_sb, W_sb = _make_schoolbook(n, m, p)
    
    # Try lower ranks
    for rank in range(30, 61):
        if time.time() > deadline - 30:
            break
        
        result = _try_optimize(n, m, p, rank, max_time=min(20.0, max(1.0, deadline - time.time() - 10)))
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[U[i, j] for j in range(U.shape[1])] for i in range(U.shape[0])]
            V_list = [[V[i, j] for j in range(V.shape[1])] for i in range(V.shape[0])]
            W_list = [[W[i, j] for j in range(W.shape[1])] for i in range(W.shape[0])]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U_list = [[U_sb[i, j] for j in range(U_sb.shape[1])] for i in range(U_sb.shape[0])]
    V_list = [[V_sb[i, j] for j in range(V_sb.shape[1])] for i in range(V_sb.shape[0])]
    W_list = [[W_sb[i, j] for j in range(W_sb.shape[1])] for i in range(W_sb.shape[0])]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
