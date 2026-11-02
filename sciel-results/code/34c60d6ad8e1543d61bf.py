import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Indices:
    # U: (n*m, R) -> index i*m + j
    # V: (m*p, R) -> index j*p + k
    # W: (p*n, R) -> index k*n + i
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Target Tensor T[u, v, w]
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # T[u, v, w] = 1 if indices are consistent, else 0
    
    TARGET_TENSOR = np.zeros((size_u, size_v, size_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                TARGET_TENSOR[u, v, w] = 1.0

    def tensor_contraction(U, V, W):
        return np.einsum('ur,vr,wr->uvw', U, V, W)

    def verify_solution(U, V, W):
        """Verify that U, V, W correctly decompose the tensor with exact arithmetic."""
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    total = 0
                    for c in range(len(U[0])):
                        total += U_frac[u_idx][c] * V_frac[v_idx][c] * W_frac[w_idx][c]
                    if total != 1:
                        return False
        return True

    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((size_u, rank_school))
    V_school = np.zeros((size_v, rank_school))
    W_school = np.zeros((size_w, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_solution = None
    best_rank = rank_school

    # ALS Implementation
    def run_als(U, V, W, max_iter=100, tol=1e-10):
        """
        Run Alternating Least Squares to refine U, V, W.
        Returns the final U, V, W and the error.
        """
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Update U
            # For each u, solve min || T[u,:,:] - sum_r U[u,r] V[:,r] W[:,r] ||
            # This is linear in U[u, :]
            # Let A_u be (size_v * size_w) x R
            # A_u[ (v, w), r ] = V[v, r] * W[w, r]
            # b_u = T[u, :, :].flatten()
            # Solve A_u * U_u = b_u
            
            U_new = np.zeros_like(U)
            for u in range(size_u):
                # Construct A_u
                # V is (size_v, R), W is (size_w, R)
                # We need outer product of each row of V and W? No.
                # A_u[v, w, r] = V[v, r] * W[w, r]
                # Reshape to (size_v * size_w, R)
                # V[:, :, None] * W[None, :, :] -> (size_v, size_w, R)
                # Reshape to (size_v * size_w, R)
                A_u = (V[:, :, np.newaxis] * W[np.newaxis, :, :]).reshape(size_v * size_w, -1)
                b_u = TARGET_TENSOR[u].flatten()
                
                # Least squares
                try:
                    U_new[u], residuals, rank, sv = np.linalg.lstsq(A_u, b_u, rcond=None)
                except:
                    U_new[u] = U[u]
            
            # Update V
            V_new = np.zeros_like(V)
            for v in range(size_v):
                A_v = (U[:, :, np.newaxis] * W[np.newaxis, :, :]).reshape(size_u * size_w, -1)
                b_v = TARGET_TENSOR[:, v, :].flatten()
                try:
                    V_new[v], residuals, rank, sv = np.linalg.lstsq(A_v, b_v, rcond=None)
                except:
                    V_new[v] = V[v]
                    
            # Update W
            W_new = np.zeros_like(W)
            for w in range(size_w):
                A_w = (U[:, :, np.newaxis] * V[np.newaxis, :, :]).reshape(size_u * size_v, -1)
                b_w = TARGET_TENSOR[:, :, w].flatten()
                try:
                    W_new[w], residuals, rank, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
                except:
                    W_new[w] = W[w]
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            T_curr = tensor_contraction(U, V, W)
            err = np.linalg.norm(T_curr - TARGET_TENSOR)
            if err < tol:
                break
                
        T_curr = tensor_contraction(U, V, W)
        err = np.linalg.norm(T_curr - TARGET_TENSOR)
        return U, V, W, err

    def try_rank(R, seed):
        """
        Try to find a decomposition of rank R using ALS.
        """
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(size_u, R) * 0.1
        V = np.random.randn(size_v, R) * 0.1
        W = np.random.randn(size_w, R) * 0.1
        
        # Run ALS
        U, V, W, err = run_als(U, V, W, max_iter=200)
        
        if err < 1e-6:
            # Try to snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            # Verify snapped
            T_snap = tensor_contraction(U_snap, V_snap, W_snap)
            if np.allclose(T_snap, TARGET_TENSOR, atol=1e-9):
                if verify_solution(U_snap, V_snap, W_snap):
                    return U_snap, V_snap, W_snap
            
            # If snapping failed, maybe the values are close to integers?
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            T_int = tensor_contraction(U_int, V_int, W_int)
            if np.allclose(T_int, TARGET_TENSOR, atol=1e-9):
                if verify_solution(U_int, V_int, W_int):
                    return U_int, V_int, W_int

            # If snapping to 0.5 or 1 fails, check if the raw solution is "exact" enough
            # and just return it? No, task requires integer/half-integer.
            # We might need a more sophisticated rounding or solving a linear system for the rounded values.
            # But for now, if ALS converges to a low error, it's a strong candidate.
            
            # Let's try a "polish" with exact arithmetic if possible?
            # For now, return None if snapping fails, but keep the best rank found if it was valid.
            
        return None

    # Try ranks from 47 down to 40
    # We use multiple seeds for each rank to increase chance of success
    for R in range(47, 39, -1):
        if time.time() > deadline - 15:
            break
            
        found = False
        for seed in range(10):
            if time.time() > deadline - 15:
                break
                
            res = try_rank(R, 42 + R * 100 + seed)
            if res:
                U_sol, V_sol, W_sol = res
                best_rank = R
                best_solution = {
                    "U": U_sol.tolist(),
                    "V": V_sol.tolist(),
                    "W": W_sol.tolist()
                }
                found = True
                break
        
        if found:
            break

    if best_solution:
        return best_solution
    else:
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
