"""
Tensor decomposition for 2x4 by 4x4 matrix multiplication.
Uses a structured rank-28 decomposition by exploiting block structure
and Strassen's algorithm for 2x2x2 sub-problems.

The current best known rank for [2,4,4] matrix multiplication is 28,
achieved by decomposing into four 2x2x2 sub-problems each using Strassen (rank 7).
We keep this exact construction as the guaranteed fallback.

We also attempt a brief local search for a rank-27 decomposition by
relaxing one Strassen term and trying to absorb it into the others via
gradient descent on the tensor equation. If no valid rank-27 is found
within the time budget, we return the rank-28 solution.
"""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

def _build_rank28():
    """Build the exact rank-28 Strassen-based decomposition."""
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    
    U = np.zeros((n * m, rank))  # 8 x 28
    V = np.zeros((m * p, rank))  # 16 x 28
    W = np.zeros((p * n, rank))  # 8 x 28
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    blocks = [
        {'A_off': 0, 'B_row_off': 0, 'B_col_off': 0, 'C_col_off': 0},
        {'A_off': 2, 'B_row_off': 2, 'B_col_off': 0, 'C_col_off': 0},
        {'A_off': 0, 'B_row_off': 0, 'B_col_off': 2, 'C_col_off': 2},
        {'A_off': 2, 'B_row_off': 2, 'B_col_off': 2, 'C_col_off': 2},
    ]
    
    current_rank = 0
    for block in blocks:
        A_off = block['A_off']
        B_row_off = block['B_row_off']
        B_col_off = block['B_col_off']
        C_col_off = block['C_col_off']
        
        for term in strassen_terms:
            coeff_A, coeff_B, coeff_C = term
            
            for loc_idx, val in coeff_A.items():
                li, lj = divmod(loc_idx, 2)
                u_idx = li * m + (A_off + lj)
                U[u_idx, current_rank] += val
            
            for loc_idx, val in coeff_B.items():
                li, lj = divmod(loc_idx, 2)
                v_idx = (B_row_off + li) * p + (B_col_off + lj)
                V[v_idx, current_rank] += val
            
            for loc_idx, val in coeff_C.items():
                li, lj = divmod(loc_idx, 2)
                w_idx = (C_col_off + lj) * n + li
                W[w_idx, current_rank] += val
            
            current_rank += 1
    
    return U.tolist(), V.tolist(), W.tolist()


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if i is row of A,
    j is row of B, k is col of B. Shape (n*m, m*p, p*n) with trace convention W[k*n+i]."""
    # T is indexed as (i*m+j, j*p+k, k*n+i) -> 1
    # Actually the tensor M_{i,j,k} for matmul: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor entries are T_{i,j,k} where the decomposition is
    # T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # So T_{i,j,k} = 1 for all valid (i,j,k)
    tensor = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                tensor[a_idx, b_idx, c_idx] = 1.0
    return tensor


def _residual(U, V, W):
    """Compute residual norm: ||T - sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]||_F"""
    # Efficient: contract U and V for each r, then dot with W
    # Or use einsum
    # R = T - sum_r U[:,r] V[:,r]^T W[:,r] ... 
    # Let's compute the reconstructed tensor
    # recon[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # Use einsum: 'ar,br,cr->abc'
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return recon


def _try_rank27(tensor, U28, V28, W28, time_budget=80.0):
    """
    Try to find a rank-27 decomposition by starting from rank-28 and
    perturbing + optimizing to drive one rank-1 term to zero.
    Returns (U, V, W) as lists if successful, else None.
    """
    n, m, p = DIMENSIONS
    target_rank = 27
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_residual = np.inf
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(50):
        if time.time() - start_time > time_budget:
            break
        
        # Start from rank-28, drop one term (randomly chosen)
        drop_idx = rng.integers(0, 28)
        
        # Take all terms except drop_idx
        keep = [i for i in range(28) if i != drop_idx]
        
        U = U28[:, keep].copy().astype(np.float64)
        V = V28[:, keep].copy().astype(np.float64)
        W = W28[:, keep].copy().astype(np.float64)
        
        # Add small random perturbation
        U += rng.normal(0, 0.01, U.shape)
        V += rng.normal(0, 0.01, V.shape)
        W += rng.normal(0, 0.01, W.shape)
        
        # Gradient descent to minimize ||T - recon||^2
        lr = 0.005
        for step in range(2000):
            if time.time() - start_time > time_budget:
                break
            
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = tensor - recon
            
            # Gradient of ||residual||^2 w.r.t. U:
            # d/dU[a,r] = -2 * sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
            # = -2 * residual @ (V[:,r] ⊗ W[:,r]) summed over b,c
            # More efficiently: grad_U[a,r] = -2 * sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
            # = -2 * einsum('abc,br,cr->ar', residual, V, W)
            
            grad_U = -2.0 * np.einsum('abc,br,cr->ar', residual, V, W)
            grad_V = -2.0 * np.einsum('abc,ar,cr->br', residual, U, W)
            grad_W = -2.0 * np.einsum('abc,ar,br->cr', residual, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check residual
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            res_norm = np.linalg.norm(tensor - recon)
            
            if res_norm < best_residual:
                best_residual = res_norm
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if res_norm < 1e-10:
                break
        
        if best_residual < 1e-10:
            break
    
    if best_residual < 1e-10 and best_U is not None:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x4 matrix.
    
    Guaranteed: rank 28 via Strassen.
    Attempted: rank 27 via local search (best effort within time budget).
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Build the guaranteed rank-28 solution
    U28_list, V28_list, W28_list = _build_rank28()
    U28 = np.array(U28_list, dtype=np.float64)
    V28 = np.array(V28_list, dtype=np.float64)
    W28 = np.array(W28_list, dtype=np.float64)
    
    # Verify rank-28 is correct
    tensor = _build_tensor(2, 4, 4)
    recon28 = np.einsum('ar,br,cr->abc', U28, V28, W28)
    assert np.linalg.norm(tensor - recon28) < 1e-10, "Rank-28 verification failed"
    
    # Try to find rank-27
    if time.time() < deadline - 10.0:
        result27 = _try_rank27(tensor, U28, V28, W28, time_budget=120.0)
        if result27 is not None:
            U27, V27, W27 = result27
            # Final verification
            U27_arr = np.array(U27, dtype=np.float64)
            V27_arr = np.array(V27, dtype=np.float64)
            W27_arr = np.array(W27, dtype=np.float64)
            recon27 = np.einsum('ar,br,cr->abc', U27_arr, V27_arr, W27_arr)
            if np.linalg.norm(tensor - recon27) < 1e-10:
                return {"U": U27, "V": V27, "W": W27}
    
    # Fallback to rank-28
    return {"U": U28_list, "V": V28_list, "W": W28_list}
