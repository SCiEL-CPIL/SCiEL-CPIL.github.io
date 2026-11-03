import time
import sys
import random
import numpy as np
import scipy.optimize
from fractions import Fraction

# Dimensions for the matrix multiplication: (n, m, p)
# A is n x m, B is m x p, C is n x p
# Here n=2, m=3, p=3
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    We first try to find a decomposition with rank 16 using numerical search.
    If not found within time budget, return the schoolbook rank-18 or known rank-17 solution.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 9
    dim_w = p * n  # 6
    
    # Target tensor
    T_target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_target[idx_u, idx_v, idx_w] = 1.0
    
    def verify(U, V, W):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = len(U[0])
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        T_calc = np.zeros((dim_u, dim_v, dim_w))
        for r in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U_np[:, r], V_np[:, r], W_np[:, r])
            
        if not np.allclose(T_calc, T_target, atol=1e-9):
            return False
        return True
    
    def rank17_decomposition():
        """
        Construct a rank-17 decomposition for 2x3x3 matrix multiplication.
        """
        rank = 17
        U = [[0.0] * rank for _ in range(dim_u)]
        V = [[0.0] * rank for _ in range(dim_v)]
        W = [[0.0] * rank for _ in range(dim_w)]
        
        # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        U[0][0] = 1; U[4][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        U[3][1] = 1; U[4][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        U[0][2] = 1
        V[1][2] = 1; V[4][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        U[4][3] = 1
        V[3][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        U[0][4] = 1; U[1][4] = 1
        V[4][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        U[3][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        U[1][6] = 1; U[4][6] = -1
        V[3][6] = 1; V[4][6] = 1
        W[0][6] = 1
        
        # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
        U[2][7] = 1
        V[6][7] = 1
        W[0][7] = 1
        
        U[2][8] = 1
        V[7][8] = 1
        W[2][8] = 1
        
        U[5][9] = 1
        V[6][9] = 1
        W[1][9] = 1
        
        U[5][10] = 1
        V[7][10] = 1
        W[3][10] = 1
        
        # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
        U[0][11] = 1
        V[2][11] = 1
        W[4][11] = 1
        
        U[1][12] = 1
        V[5][12] = 1
        W[4][12] = 1
        
        U[2][13] = 1
        V[8][13] = 1
        W[4][13] = 1
        
        U[3][14] = 1
        V[2][14] = 1
        W[5][14] = 1
        
        U[4][15] = 1
        V[5][15] = 1
        W[5][15] = 1
        
        U[5][16] = 1
        V[8][16] = 1
        W[5][16] = 1
        
        return {"U": U, "V": V, "W": W}
    
    def schoolbook():
        """Return the schoolbook rank-18 decomposition."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(dim_u)]
        V = [[0.0] * rank for _ in range(dim_v)]
        W = [[0.0] * rank for _ in range(dim_w)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u][col_idx] = 1.0
                    V[idx_v][col_idx] = 1.0
                    W[idx_w][col_idx] = 1.0
                    col_idx += 1
        
        return {"U": U, "V": V, "W": W}
    
    def try_rank16():
        """
        Attempt to find a rank-16 decomposition using numerical optimization.
        Strategy: Start from rank-17, zero out one column, perturb, and optimize.
        """
        rank = 16
        total_params = dim_u * rank + dim_v * rank + dim_w * rank
        
        T_flat = T_target.flatten()
        
        def compute_residual(x):
            U = x[:dim_u*rank].reshape(dim_u, rank)
            V = x[dim_u*rank:dim_u*rank + dim_v*rank].reshape(dim_v, rank)
            W = x[dim_u*rank + dim_v*rank:].reshape(dim_w, rank)
            T_calc = np.zeros((dim_u, dim_v, dim_w))
            for r in range(rank):
                T_calc += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            return (T_calc - T_target).flatten()
        
        # Get rank-17 solution as starting point
        r17 = rank17_decomposition()
        U17 = np.array(r17["U"])
        V17 = np.array(r17["V"])
        W17 = np.array(r17["W"])
        
        # Try zeroing different columns of the rank-17 and perturbing
        best_result = None
        best_resid = np.inf
        
        for col_to_zero in range(17):
            if time.time() > deadline - 15:
                break
                
            # Create rank-16 by removing column col_to_zero
            cols_keep = [c for c in range(17) if c != col_to_zero]
            U16 = U17[:, cols_keep].copy()
            V16 = V17[:, cols_keep].copy()
            W16 = W17[:, cols_keep].copy()
            
            # Flatten
            x0 = np.concatenate([U16.flatten(), V16.flatten(), W16.flatten()])
            
            # Compute initial residual
            resid0 = compute_residual(x0)
            resid0_norm = np.linalg.norm(resid0)
            
            # Try multiple perturbations
            for pert_seed in range(3):
                if time.time() > deadline - 10:
                    break
                    
                np.random.seed(col_to_zero * 100 + pert_seed)
                
                # Add small random perturbation
                x_perturbed = x0 + np.random.normal(0, 0.1, total_params)
                
                try:
                    # Use least squares with LM
                    result = scipy.optimize.least_squares(
                        compute_residual, x_perturbed, 
                        method='lm', max_nfev=500, ftol=1e-10, xtol=1e-10
                    )
                    x_sol = result.x
                    resid_final = np.linalg.norm(compute_residual(x_sol))
                    
                    if resid_final < best_resid:
                        best_resid = resid_final
                        U_sol = x_sol[:dim_u*rank].reshape(dim_u, rank)
                        V_sol = x_sol[dim_u*rank:dim_u*rank + dim_v*rank].reshape(dim_v, rank)
                        W_sol = x_sol[dim_u*rank + dim_v*rank:].reshape(dim_w, rank)
                        
                        # Round to nearest 0.5
                        U_rounded = np.round(U_sol * 2) / 2
                        V_rounded = np.round(V_sol * 2) / 2
                        W_rounded = np.round(W_sol * 2) / 2
                        
                        U_list = U_rounded.tolist()
                        V_list = V_rounded.tolist()
                        W_list = W_rounded.tolist()
                        
                        if verify(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
                        
                        # Also try rounding to integers
                        U_int = np.round(U_sol)
                        V_int = np.round(V_sol)
                        W_int = np.round(W_sol)
                        U_list_int = U_int.tolist()
                        V_list_int = V_int.tolist()
                        W_list_int = W_int.tolist()
                        
                        if verify(U_list_int, V_list_int, W_list_int):
                            return {"U": U_list_int, "V": V_list_int, "W": W_list_int}
                            
                except Exception as e:
                    pass
        
        # Also try random initialization
        for seed in range(5):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(1000 + seed)
            x0 = np.random.normal(0, 1, total_params)
            
            try:
                result = scipy.optimize.least_squares(
                    compute_residual, x0, 
                    method='lm', max_nfev=500, ftol=1e-10, xtol=1e-10
                )
                x_sol = result.x
                resid_final = np.linalg.norm(compute_residual(x_sol))
                
                if resid_final < best_resid:
                    best_resid = resid_final
                    U_sol = x_sol[:dim_u*rank].reshape(dim_u, rank)
                    V_sol = x_sol[dim_u*rank:dim_u*rank + dim_v*rank].reshape(dim_v, rank)
                    W_sol = x_sol[dim_u*rank + dim_v*rank:].reshape(dim_w, rank)
                    
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    
                    if verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception as e:
                pass
        
        return None
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # First, try the known rank-17 decomposition
    if time.time() < deadline:
        r17 = rank17_decomposition()
        if verify(r17["U"], r17["V"], r17["W"]):
            best_result = r17
            best_rank = 17
    
    # Attempt rank 16 search if time permits
    if best_rank > 16 and time.time() < deadline - 20:
        try:
            r16 = try_rank16()
            if r16:
                return r16
        except Exception:
            pass

    if best_rank < n * m * p:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
