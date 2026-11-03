"""
Exact low-rank tensor decomposition for 4x5 * 5x5 matrix multiplication.

The multiplication tensor T[i,j,k] = delta_{i,0} (for the first 5x5 block)
plus contributions from off-diagonal blocks.

We attempt to find a decomposition:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

Strategy:
1. Start with the schoolbook (rank 100) decomposition.
2. Use a structured approach: identify the 25 entries of the main 5x5 block
   and the 25 entries of the off-diagonal blocks (4x5 each for rows 1-3).
   Actually, the tensor is:
   - T[0, j, k] = 1 for j in 0..4, k in 0..4 (5x5 identity? No, it's the multiplication)
   
   Wait, let me reconsider. The matrix multiplication tensor for A (n x m) * B (m x p):
   C[i,k] = sum_j A[i,j] * B[j,k]
   
   The tensor T[i,j,k] = 1 for all i,j,k? No.
   The tensor T[i,j,k] = delta_{i,0}? No.
   
   The multiplication tensor has T[i,j,k] = 1 for all valid i,j,k? 
   No. The tensor T is defined such that:
   (A*B)[i,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
   
   So T[i,j,k] = 1 if and only if the (i,j) entry of A and (j,k) entry of B
   contribute to the (i,k) entry of C. This means T[i,j,k] = 1 for all i,j,k
   that form a valid "path" i -> j -> k.
   
   For n=4, m=5, p=5:
   - i ranges over 0..3 (rows of A and C)
   - j ranges over 0..4 (cols of A = rows of B)
   - k ranges over 0..4 (cols of B and C)
   
   T[i,j,k] = 1 for ALL i in 0..3, j in 0..4, k in 0..4.
   
   So the target tensor IS all ones with shape (4, 5, 5)!
   
   The schoolbook decomposition uses rank 100 = 4*5*5.
   
   Can we do better? The tensor is all-ones, which is a rank-1 tensor in the
   standard 3-way sense, BUT the factor structure is constrained:
   U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
   
   This is NOT a standard 3-way tensor decomposition. The indices are coupled.
   
   Let's think about this differently. We need:
   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
   
   This is equivalent to: for each r, define:
   u_r[i,j] = U[i*m+j, r]
   v_r[j,k] = V[j*p+k, r]  
   w_r[k,i] = W[k*n+i, r]
   
   Then sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1
   
   This is a "cyclic" tensor decomposition. The trace of the product
   u_r * v_r * w_r (as matrices) should sum to 1 for each (i,k)... no.
   
   Actually: sum_j u_r[i,j] * v_r[j,k] = (u_r * v_r)[i,k]
   Then multiply by w_r[k,i] and sum over k: sum_k (u_r * v_r)[i,k] * w_r[k,i]
   = trace(u_r * v_r * w_r)
   
   So we need: sum_r trace(u_r * v_r * w_r) = 1 where u_r is 4x5, v_r is 5x5, w_r is 5x4.
   
   But wait, that's just a scalar. The constraint is per (i,j,k), not just the trace.
   
   Let me re-examine: T[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i] summed over r = 1.
   
   This means for EACH (i,j,k), the sum must be 1. It's 100 independent equations.
   
   With rank R, we have 3*R*50 variables (U has 20*R, V has 25*R, W has 20*R entries... 
   wait: U is (n*m, R) = (20, R), V is (m*p, R) = (25, R), W is (p*n, R) = (20, R).
   Total variables: 65*R.
   
   100 equations, 65*R unknowns. For R=2, we have 130 unknowns for 100 equations.
   Might be feasible!
   
   Let's try to solve this system for small R.
"""
import time
import numpy as np
from itertools import product

N, M, P = 4, 5, 5
R_SCHOOLBOOK = N * M * P  # 100


def _build_target_tensor():
    """The target tensor T[i,j,k] = 1 for all i,j,k."""
    return np.ones((N, M, P))


def _verify(U, V, W, tol=1e-9):
    """Check the tensor identity exactly."""
    T = np.zeros((N, M, P))
    for r in range(U.shape[1]):
        u = U[:, r].reshape(N, M)
        v = V[:, r].reshape(M, P)
        w = W[:, r].reshape(P, N)
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return np.allclose(T, 1.0, atol=tol, rtol=1e-9)


def _schoolbook():
    """Standard schoolbook decomposition with rank N*M*P."""
    rank = N * M * P
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, c] = 1
                V[j * P + k, c] = 1
                W[k * N + i, c] = 1
                c += 1
    return U, V, W


def _build_linear_system_for_U(V, W, T_target):
    """
    Build the linear system A @ u_flat = t_flat for solving U given V, W.
    
    T[i,j,k] = sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r]
    
    For fixed (i,j), and all k:
    T[i,j,k] = sum_r U[i*M+j, r] * (V[j*P+k, r] * W[k*N+i, r])
    
    So for each (i,j), we have P equations in R unknowns (the R values of U[i*M+j, :]).
    Actually, all (i,j,k) together give N*M*P equations in N*M*R unknowns.
    """
    n, m, p, R = N, M, P, V.shape[1]
    num_eqs = n * m * p
    num_unk = n * m * R
    
    A = np.zeros((num_eqs, num_unk))
    b = T_target.flatten().copy()
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                for r in range(R):
                    unk_idx = (i * m + j) * R + r
                    A[eq_idx, unk_idx] = V[j * p + k, r] * W[k * n + i, r]
    
    return A, b


def _solve_for_U(V, W, T_target):
    """Solve for U given V, W using least squares."""
    A, b = _build_linear_system_for_U(V, W, T_target)
    # A is (100, 20*R), b is (100,)
    # Use least squares
    try:
        u_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        return u_flat.reshape(N * M, V.shape[1])
    except:
        return None


def _solve_for_V(U, W, T_target):
    """Solve for V given U, W."""
    n, m, p, R = N, M, P, U.shape[1]
    num_eqs = n * m * p
    num_unk = m * p * R
    
    A = np.zeros((num_eqs, num_unk))
    b = T_target.flatten().copy()
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                for r in range(R):
                    unk_idx = (j * p + k) * R + r
                    A[eq_idx, unk_idx] = U[i * m + j, r] * W[k * n + i, r]
    
    try:
        v_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        return v_flat.reshape(M * P, R)
    except:
        return None


def _solve_for_W(U, V, T_target):
    """Solve for W given U, V."""
    n, m, p, R = N, M, P, U.shape[1]
    num_eqs = n * m * p
    num_unk = p * n * R
    
    A = np.zeros((num_eqs, num_unk))
    b = T_target.flatten().copy()
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                for r in range(R):
                    unk_idx = (k * n + i) * R + r
                    A[eq_idx, unk_idx] = U[i * m + j, r] * V[j * p + k, r]
    
    try:
        w_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        return w_flat.reshape(P * N, R)
    except:
        return None


def _als_decomposition(rank, T_target, max_iter=100, seed=0, tol=1e-8):
    """
    Alternating Least Squares for the cyclic tensor decomposition.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(N * M, rank) * 0.1
    V = rng.randn(M * P, rank) * 0.1
    W = rng.randn(P * N, rank) * 0.1
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Update U
        U_new = _solve_for_U(V, W, T_target)
        if U_new is None:
            break
        
        # Update V
        V_new = _solve_for_V(U_new, W, T_target)
        if V_new is None:
            break
        
        # Update W
        W_new = _solve_for_W(U_new, V_new, T_target)
        if W_new is None:
            break
        
        # Check convergence
        err = np.linalg.norm(T_target - np.einsum('ir,jr,kr->ijk',
                    U_new.reshape(N, M, rank), 
                    V_new.reshape(M, P, rank), 
                    W_new.reshape(P, N, rank),
                    optimize=True) - 1.0)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
        
        if err < tol:
            break
        
        U, V, W = U_new, V_new, W_new
    
    return best_U, best_V, best_W, best_err


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    T_target = _build_target_tensor()
    
    # Start with schoolbook as safe fallback
    U_sb, V_sb, W_sb = _schoolbook()
    best_rank = R_SCHOOLBOOK
    best_result = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank decompositions using ALS
    # Try ranks from 25 (theoretical lower bound) up to 99
    # But be smart: try a few promising ranks
    
    ranks_to_try = [25, 30, 35, 40, 45, 50, 60, 70, 80, 90]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try multiple seeds for this rank
        for seed in range(10):
            if time.time() > deadline:
                break
            
            U, V, W, err = _als_decomposition(rank, T_target, max_iter=50, seed=seed, tol=1e-8)
            
            if err < 1e-6:
                # Verify exactly
                if _verify(U, V, W, tol=1e-9):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U, V, W)
                    break  # Found valid decomposition for this rank, move to next rank
    
    # Return the best result
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
