"""
5x5 matrix multiplication - rank-minimal exact tensor decomposition.
Implements a randomized algebraic rank-reduction heuristic with
integer-preserving Gram-Schmidt orthogonalization to find lower-rank
decompositions below the schoolbook rank of 125.
"""
import numpy as np
from itertools import product
import time

DIMENSIONS = [5, 5, 5]

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition C[i,k] = sum_c W[k*n+i,c] * (U^T A)_c * (V^T B)_c
    equals (AB)[i,k] for all i,k.
    
    This is done by checking the tensor identity:
    For all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    And for all other (i,j,k) combinations not in the tensor, the sum should be 0.
    Actually, the tensor T[i,j,k] for matrix mult is:
    T[i,j,k] = 1 if this (i,j,k) triple contributes, which is always for valid indices.
    But the bilinear form structure means we need:
    sum_c U[a,c] * V[b,c] * W[c_idx,c] where a=i*m+j, b=j'*p+k, c_idx=k'*n+i'
    This gets complex. Let's use a direct verification via random test matrices.
    """
    # Generate random test matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute using the decomposition
    # U_flat has shape (n*m, rank), V_flat has shape (m*p, rank), W_flat has shape (p*n, rank)
    # (U^T A_flat)_c = sum_{i,j} U[i*m+j, c] * A[i,j]
    A_flat = A.flatten()  # shape (n*m,)
    B_flat = B.flatten()  # shape (m*p,)
    
    U_A = U.T @ A_flat  # shape (rank,)
    V_B = V.T @ B_flat  # shape (rank,)
    
    # C[i,k] = sum_c W[k*n+i, c] * U_A[c] * V_B[c]
    W_UA_VB = W * (U_A * V_B)[np.newaxis, :]  # shape (p*n, rank)
    C_computed_flat = W_UA_VB.sum(axis=1)  # shape (p*n,)
    C_computed = C_computed_flat.reshape(p, n).T  # shape (n, p)
    
    if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
        return False
    return True

def _integer_preserve_gram_schmidt(mat, n_cols):
    """
    Apply integer-preserving Gram-Schmidt orthogonalization to the first n_cols columns.
    This projects columns onto the null space while keeping entries as close to integers as possible.
    """
    if n_cols <= 0:
        return mat
    
    mat = mat.copy()
    for c in range(n_cols):
        # Project out components along previous columns
        for prev in range(c):
            proj = np.dot(mat[:, c], mat[:, prev]) / (np.dot(mat[:, prev], mat[:, prev]) + 1e-15)
            mat[:, c] -= proj * mat[:, prev]
        # Round to nearest integer/half-integer
        mat[:, c] = np.round(mat[:, c] * 2) / 2
    return mat

def _try_randomized_reduction(n, m, p, target_rank, max_iters=200, seed=0):
    """
    Try to find a rank-target_rank decomposition using randomized initialization
    and iterative refinement with integer-preserving Gram-Schmidt.
    """
    np.random.seed(seed)
    
    for attempt in range(max_iters):
        # Initialize with random integer values
        U = np.random.randint(-2, 3, (n * m, target_rank)).astype(np.float64)
        V = np.random.randint(-2, 3, (m * p, target_rank)).astype(np.float64)
        W = np.random.randint(-2, 3, (p * n, target_rank)).astype(np.float64)
        
        # Iterative refinement: project onto constraint space
        for it in range(50):
            # Compute residual: for each (i,j,k), the tensor value should be 1
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # We want this to equal 1 for all valid (i,j,k)
            
            # Compute current tensor values
            # For efficiency, compute via matrix operations
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            
            # Residual for each (i,j,k):
            # res[i,j,k] = 1 - sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            
            # This is a trilinear form. Let's compute it efficiently.
            # For fixed i,k: sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            
            # Build the residual tensor
            # Use broadcasting
            # U_reshaped: (n, m, rank)
            # V_reshaped: (m, p, rank)
            # W_reshaped: (p, n, rank)
            
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            
            # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
            # Compute via einsum
            T_current = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            
            # Target tensor: T_target[i,j,k] = 1 for all i,j,k (since every triple contributes)
            # Wait, that's not right. The tensor for matrix multiplication is:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # The bilinear form has terms A[i,j] * B[j,k] for each j.
            # So the tensor T[i,j,k] = 1 for all i,j,k (each term appears once).
            T_target = np.ones((n, m, p))
            
            residual = T_target - T_current
            
            # Check if we're close enough
            if np.max(np.abs(residual)) < 1e-8:
                # Verify with actual matrix multiplication
                if _verify_decomposition(U, V, W, n, m, p):
                    return U, V, W, True
            
            # Gradient-based update: move in direction that reduces residual
            # For U[i,j,c]: dT/dU[i,j,c] = sum_k V[j,k,c] * W[k,i,c]
            # For V[j,k,c]: dT/dV[j,k,c] = sum_i U[i,j,c] * W[k,i,c]
            # For W[k,i,c]: dT/dW[k,i,c] = sum_j U[i,j,c] * V[j,k,c]
            
            # Compute gradients
            grad_U = np.einsum('jkc,kic->ijc', V_r, W_r)
            grad_V = np.einsum('ijc,kic->jkc', U_r, W_r)
            grad_W = np.einsum('ijc,jkc->kic', U_r, V_r)
            
            # Update in direction that reduces residual
            lr = 0.01
            U_r -= lr * residual * grad_U
            V_r -= lr * residual * grad_V
            W_r -= lr * residual * grad_W
            
            # Apply integer-preserving Gram-Schmidt
            U = _integer_preserve_gram_schmidt(U_r.reshape(-1, target_rank), target_rank).reshape(n, m, target_rank)
            V = _integer_preserve_gram_schmidt(V_r.reshape(-1, target_rank), target_rank).reshape(m, p, target_rank)
            W = _integer_preserve_gram_schmidt(W_r.reshape(-1, target_rank), target_rank).reshape(p, n, target_rank)
        
        # Final check
        if _verify_decomposition(U.reshape(-1, target_rank), V.reshape(-1, target_rank), W.reshape(-1, target_rank), n, m, p):
            return U.reshape(-1, target_rank), V.reshape(-1, target_rank), W.reshape(-1, target_rank), True
    
    return None, None, None, False

def _try_column_merging(n, m, p, U, V, W, max_merges=10):
    """
    Try to merge columns that are linearly dependent or nearly so.
    This is a heuristic that may reduce rank slightly.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    merged = 0
    cols_to_remove = []
    
    # Check for near-duplicate columns using cosine similarity
    from numpy.linalg import norm
    
    for c1 in range(rank):
        if c1 in cols_to_remove:
            continue
        for c2 in range(c1 + 1, rank):
            if c2 in cols_to_remove:
                continue
            # Check if columns are similar
            col_U1 = U[:, c1]
            col_U2 = U[:, c2]
            col_V1 = V[:, c1]
            col_V2 = V[:, c2]
            col_W1 = W[:, c1]
            col_W2 = W[:, c2]
            
            # Compute cosine similarity
            def cos_sim(a, b):
                na = norm(a)
                nb = norm(b)
                if na < 1e-10 or nb < 1e-10:
                    return 0
                return np.dot(a, b) / (na * nb)
            
            sim_U = cos_sim(col_U1, col_U2)
            sim_V = cos_sim(col_V1, col_V2)
            sim_W = cos_sim(col_W1, col_W2)
            
            if sim_U > 0.999 and sim_V > 0.999 and sim_W > 0.999:
                # Merge: keep c1, remove c2
                cols_to_remove.append(c2)
                merged += 1
                if merged >= max_merges:
                    break
        if merged >= max_merges:
            break
    
    if merged > 0:
        keep_cols = [c for c in range(rank) if c not in cols_to_remove]
        U_new = U[:, keep_cols]
        V_new = V[:, keep_cols]
        W_new = W[:, keep_cols]
        
        # Verify the merged decomposition
        if _verify_decomposition(U_new, V_new, W_new, n, m, p):
            return U_new, V_new, W_new, True
    
    return None, None, None, False

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    U_s, V_s, W_s = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    
    # Try to reduce rank using randomized algebraic approach
    # Try ranks from 120 down to 100
    for target_rank in range(120, 99, -1):
        success = False
        for seed in range(10):
            U_try, V_try, W_try, ok = _try_randomized_reduction(n, m, p, target_rank, max_iters=50, seed=seed)
            if ok:
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                success = True
                break
        if success:
            break
    
    # Try column merging on the best found so far
    U_m, V_m, W_m, ok = _try_column_merging(n, m, p, best_U, best_V, best_W, max_merges=5)
    if ok:
        new_rank = U_m.shape[1]
        if new_rank < best_rank:
            best_rank = new_rank
            best_U, best_V, best_W = U_m, V_m, W_m
    
    # Ensure integer/half-integer entries
    best_U = np.round(best_U * 2) / 2
    best_V = np.round(best_V * 2) / 2
    best_W = np.round(best_W * 2) / 2
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook
        best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
