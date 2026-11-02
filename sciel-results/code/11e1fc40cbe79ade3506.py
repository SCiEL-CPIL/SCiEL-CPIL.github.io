"""Rewritten program for 4x4 * 4x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 6]

import numpy as np
import time
import random

def _verify_rank(n, m, p, rank, U, V, W):
    """Verify that the given factors decompose the matmul tensor correctly."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m + j, c] * V[j*p + k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-6:
                    return False, (i, j, k, val)
    return True, None

def _compute_error(n, m, p, U, V, W):
    """Compute the total squared error of the decomposition."""
    rank = U.shape[1]
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m + j, c] * V[j*p + k, c] * W[k*n+i, c]
                err += (val - 1.0) ** 2
    return err

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 second deadline, with margin for safety
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 4*4*6 = 96
    
    # First, create the schoolbook solution as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U_school[i * m + j, c] = 1
                V_school[j * p + k, c] = 1
                W_school[k * n + i, c] = 1
    
    best_rank = schoolbook_rank
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    random.seed(42)
    np.random.seed(42)
    
    # Strategy: Use iterative alternating least squares / gradient-based optimization
    # For each target rank, try to find a valid decomposition
    
    # We'll try ranks from 90 down to 70
    # The key insight: we need to solve a system of nonlinear equations
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Use a smarter initialization and optimization approach
    
    def try_rank(target_rank, max_time=30):
        """Try to find a valid decomposition for the given rank."""
        local_start = time.time()
        local_deadline = local_start + max_time
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > local_deadline or time.time() > deadline - 5:
                break
            
            # Random initialization with structure
            U_r = np.random.randn(n * m, target_rank) * 0.3
            V_r = np.random.randn(m * p, target_rank) * 0.3
            W_r = np.random.randn(p * n, target_rank) * 0.3
            
            # Normalize each column
            for c in range(target_rank):
                norm_u = np.linalg.norm(U_r[:, c])
                norm_v = np.linalg.norm(V_r[:, c])
                norm_w = np.linalg.norm(W_r[:, c])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    scale = 1.0 / (norm_u * norm_v * norm_w) ** (1.0/3.0)
                    U_r[:, c] *= scale
                    V_r[:, c] *= scale
                    W_r[:, c] *= scale
            
            # Alternating optimization
            # Fix U and V, solve for W (linear least squares)
            # Fix U and W, solve for V
            # Fix V and W, solve for U
            
            for iteration in range(100):
                if time.time() > local_deadline or time.time() > deadline - 5:
                    break
                
                # Check current error
                err = _compute_error(n, m, p, U_r, V_r, W_r)
                if err < 1e-8:
                    return True, U_r, V_r, W_r
                
                # Optimize W given U, V
                # For each (i, k): sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # This is: for each (i, k), A_{j,c} = U[i*m+j, c] * V[j*p+k, c]
                # We need: sum_c W[k*n+i, c] * (sum_j A_{j,c} * 1) ... no
                # Actually: for fixed (i,k), the equation is:
                # sum_j [sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]] = 1
                # = sum_c W[k*n+i, c] * [sum_j U[i*m+j, c] * V[j*p+k, c]] = 1
                # Let B_{i,k,c} = sum_j U[i*m+j, c] * V[j*p+k, c]
                # Then: sum_c B_{i,k,c} * W[k*n+i, c] = 1 for all (i,k)
                
                # Build the system for W
                # Number of equations: n*p = 24
                # Unknowns: p*n * target_rank = 24 * target_rank
                # This is underdetermined, so we find min-norm solution
                
                B = np.zeros((n * p, target_rank))
                for i in range(n):
                    for k in range(p):
                        for c in range(target_rank):
                            B[i*p + k, c] = 0
                            for j in range(m):
                                B[i*p + k, c] += U_r[i*m + j, c] * V_r[j*p + k, c]
                
                # Solve B @ W_flat = ones(n*p)
                # W_flat has shape (n*p, target_rank) -> reshape to (p*n, target_rank)
                # Actually W has shape (p*n, target_rank), and W[k*n+i, c]
                # So row index is k*n + i, not i*p + k
                # Let's rebuild properly
                
                B2 = np.zeros((n * p, target_rank))
                for i in range(n):
                    for k in range(p):
                        for c in range(target_rank):
                            s = 0
                            for j in range(m):
                                s += U_r[i*m + j, c] * V_r[j*p + k, c]
                            B2[i*p + k, c] = s
                
                # W row index: k*n + i
                # So we need to map (i,k) -> row k*n+i
                # B2 row index: i*p + k
                # We need to permute B2 rows to match W's row ordering
                
                W_flat_target = np.ones(n * p)
                # Solve B2 @ W_perm = W_flat_target
                # where W_perm[i*p+k, c] = W[k*n+i, c]
                
                if target_rank <= n * p:
                    W_perm, _, _, _ = np.linalg.lstsq(B2, W_flat_target, rcond=None)
                else:
                    W_perm, _, _, _ = np.linalg.lstsq(B2, W_flat_target, rcond=None)
                
                # Unpermute: W[k*n+i, c] = W_perm[i*p+k, c]
                for i in range(n):
                    for k in range(p):
                        W_r[k*n + i, :] = W_perm[i*p + k, :]
                
                # Optimize U given V, W
                # For each (i, j): sum_k sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # = sum_c U[i*m+j, c] * [sum_k V[j*p+k, c] * W[k*n+i, c]] = 1
                # Let C_{i,j,c} = sum_k V[j*p+k, c] * W[k*n+i, c]
                # Then: sum_c C_{i,j,c} * U[i*m+j, c] = 1 for all (i,j)
                
                C = np.zeros((n * m, target_rank))
                for i in range(n):
                    for j in range(m):
                        for c in range(target_rank):
                            s = 0
                            for k in range(p):
                                s += V_r[j*p + k, c] * W_r[k*n + i, c]
                            C[i*m + j, c] = s
                
                U_target = np.ones(n * m)
                if target_rank <= n * m:
                    U_new, _, _, _ = np.linalg.lstsq(C, U_target, rcond=None)
                else:
                    U_new, _, _, _ = np.linalg.lstsq(C, U_target, rcond=None)
                U_r = U_new.reshape(n * m, target_rank)
                
                # Optimize V given U, W
                # For each (j, k): sum_i sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # = sum_c V[j*p+k, c] * [sum_i U[i*m+j, c] * W[k*n+i, c]] = 1
                # Let D_{j,k,c} = sum_i U[i*m+j, c] * W[k*n+i, c]
                # Then: sum_c D_{j,k,c} * V[j*p+k, c] = 1 for all (j,k)
                
                D = np.zeros((m * p, target_rank))
                for j in range(m):
                    for k in range(p):
                        for c in range(target_rank):
                            s = 0
                            for i in range(n):
                                s += U_r[i*m + j, c] * W_r[k*n + i, c]
                            D[j*p + k, c] = s
                
                V_target = np.ones(m * p)
                if target_rank <= m * p:
                    V_new, _, _, _ = np.linalg.lstsq(D, V_target, rcond=None)
                else:
                    V_new, _, _, _ = np.linalg.lstsq(D, V_target, rcond=None)
                V_r = V_new.reshape(m * p, target_rank)
            
            # Final check
            err = _compute_error(n, m, p, U_r, V_r, W_r)
            if err < 1e-8:
                return True, U_r, V_r, W_r
        
        return False, None, None, None
    
    # Try ranks from 90 down to 70
    for target_rank in range(90, 65, -1):
        if time.time() > deadline - 10:
            break
        
        success, U_try, V_try, W_try = try_rank(target_rank, max_time=25)
        
        if success:
            # Verify
            ok, _ = _verify_rank(n, m, p, target_rank, U_try, V_try, W_try)
            if ok:
                best_rank = target_rank
                best_U = U_try
                best_V = V_try
                best_W = W_try
                print(f"Found rank {target_rank} decomposition")
                # Continue trying for lower ranks
            else:
                print(f"Rank {target_rank} failed verification")
        else:
            print(f"Rank {target_rank} not found")
            # If we've tried several ranks without success, stop
            if target_rank < 80:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
