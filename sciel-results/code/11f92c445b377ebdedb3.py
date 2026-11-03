"""Find a rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 75

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds
_SAFETY_MARGIN = 5.0  # seconds before kill

def _time_left():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)

def _validate(U, V, W):
    """Check if decomposition is exact: sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] == T."""
    n, m, p = N, M, P
    rank = U.shape[1]
    # Build tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Target: T[i,j,k] = 1 if all indices valid (which they always are for matrix mult)
    # Actually for matrix multiplication, the tensor is T[i,j,k] = delta_{j,k}... no.
    # Matrix multiplication: (AB)_{ik} = sum_j A_{ij} B_{jk}
    # Tensor T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
    # Wait, no. The multiplication tensor has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # where A is n x m, B is m x p, C = AB is n x p.
    # C_{ik} = sum_j A_{ij} B_{jk}
    # So the tensor is T[i,j,k] = 1 for all i,j,k.
    
    # Check: for each c, contribution to T[i,j,k] is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # We need sum_c of this = 1 for all (i,j,k)
    
    # Efficient check using einsum
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = np.array(U).reshape(n, m, rank)
    V3 = np.array(V).reshape(m, p, rank)
    W3 = np.array(W).reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # This is a contraction over c and we need to verify it equals 1 for all i,j,k
    # Use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # Check if all entries are 1
    if T.shape != (n, m, p):
        return False
    return np.allclose(T, 1.0, atol=1e-10)

def _validate_fast(U, V, W):
    """Faster validation using matrix operations."""
    n, m, p = N, M, P
    rank = U.shape[1]
    U3 = np.array(U).reshape(n, m, rank)
    V3 = np.array(V).reshape(m, p, rank)
    W3 = np.array(W).reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

def _objective(U, V, W):
    """
    Smooth objective: measure deviation from exact decomposition.
    Returns (error, is_exact).
    """
    n, m, p = N, M, P
    rank = U.shape[1]
    U3 = np.array(U).reshape(n, m, rank)
    V3 = np.array(V).reshape(m, p, rank)
    W3 = np.array(W).reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    error = np.sum((T - 1.0)**2)
    is_exact = error < 1e-10
    return error, is_exact

def _als_step(U, V, W, lr=0.1, max_iter=100):
    """
    Alternating least squares to refine U, V, W to minimize sum(T-1)^2.
    This is a smooth surrogate for the nonsmooth "is it exact" objective.
    """
    n, m, p = N, M, P
    rank = U.shape[1]
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    for it in range(max_iter):
        if _time_left() < 2.0:
            break
        
        # Update U: for each (i,j), U[i*m+j,:] = argmin sum_k (T[i,j,k]-1)^2
        # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # For fixed i,j: minimize sum_k (sum_c U_c * A_{k,c} - 1)^2 where A_{k,c} = V3[j,k,c]*W3[k,i,c]
        # This is a least squares problem: U_row @ A^T = 1
        for i in range(n):
            for j in range(m):
                # A[k,c] = V3[j,k,c] * W3[k,i,c]
                A = V3[j, :, :] * W3[:, i, :]  # (p, rank)
                # Minimize ||U_row @ A^T - 1||^2
                # Solution: U_row = (A @ A^T)^{-1} @ A @ 1  ... no
                # min_u ||A^T u - 1||^2 -> u = (A A^T)^{-1} A
                # A is (p, rank), we want u (rank,) such that A^T u ≈ 1 (p,)
                # Normal equations: A A^T u = A
                try:
                    AtA = A @ A.T  # (p, p)
                    b = np.sum(A, axis=1)  # (p,)
                    # Solve AtA u = b for u, then project back
                    # Actually we want to minimize ||A^T u - 1||^2
                    # grad_u = 2 A (A^T u - 1) = 0 -> A A^T u = A 1 = sum(A, axis=1)
                    u_new = np.linalg.solve(AtA + 1e-10 * np.eye(p), b)
                    U3[i, j, :] = u_new
                except:
                    pass
        
        # Update V: for each (j,k), V[j*p+k,:] 
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
                # For fixed j,k: minimize sum_i (sum_c V_c * B_{i,c} - 1)^2 where B_{i,c} = U3[i,j,c]*W3[k,i,c]
                B = U3[:, j, :] * W3[k, :, :]  # (n, rank)
                try:
                    BtB = B @ B.T  # (n, n)
                    b = np.sum(B, axis=1)  # (n,)
                    v_new = np.linalg.solve(BtB + 1e-10 * np.eye(n), b)
                    V3[j, k, :] = v_new
                except:
                    pass
        
        # Update W: for each (k,i), W[k*n+i,:]
        for k in range(p):
            for i in range(n):
                # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
                # For fixed k,i: minimize sum_j (sum_c W_c * C_{j,c} - 1)^2 where C_{j,c} = U3[i,j,c]*V3[j,k,c]
                C = U3[i, :, :] * V3[:, k, :]  # (m, rank)
                try:
                    CtC = C @ C.T  # (m, m)
                    b = np.sum(C, axis=1)  # (m,)
                    w_new = np.linalg.solve(CtC + 1e-10 * np.eye(m), b)
                    W3[k, i, :] = w_new
                except:
                    pass
        
        # Check convergence
        if it % 10 == 0:
            T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            err = np.sum((T - 1.0)**2)
            if err < 1e-12:
                break
    
    return U3.reshape(n*m, rank), V3.reshape(m*p, rank), W3.reshape(p*n, rank)

def _try_reduce_rank(U, V, W, target_rank, max_trials=50):
    """
    Try to reduce rank from current to target_rank.
    Strategy: initialize with random linear combinations of existing columns,
    then refine with ALS.
    """
    n, m, p = N, M, P
    current_rank = U.shape[1]
    
    for trial in range(max_trials):
        if _time_left() < 3.0:
            break
        
        # Create new factor matrices of target_rank
        # Initialize by taking random linear combinations of existing columns
        rng = np.random.default_rng(trial * 12345 + 67890)
        
        # Random mixing matrices
        mix_U = rng.standard_normal((current_rank, target_rank))
        mix_V = rng.standard_normal((current_rank, target_rank))
        mix_W = rng.standard_normal((current_rank, target_rank))
        
        # Scale to maintain approximate norm
        mix_U /= np.linalg.norm(mix_U)
        mix_V /= np.linalg.norm(mix_V)
        mix_W /= np.linalg.norm(mix_W)
        
        U_new = np.array(U) @ mix_U
        V_new = np.array(V) @ mix_V
        W_new = np.array(W) @ mix_W
        
        # Refine with ALS
        U_ref, V_ref, W_ref = _als_step(U_new, V_new, W_new, max_iter=200)
        
        # Check if exact
        if _validate_fast(U_ref, V_ref, W_ref):
            return U_ref, V_ref, W_ref
        
        # Also try with different initialization: keep some columns, mix others
        if trial % 5 == 0:
            # Keep first half of columns, mix second half
            half = current_rank // 2
            U_mix = np.hstack([np.eye(half, current_rank), rng.standard_normal((target_rank - half, current_rank))])
            V_mix = np.hstack([np.eye(half, current_rank), rng.standard_normal((target_rank - half, current_rank))])
            W_mix = np.hstack([np.eye(half, current_rank), rng.standard_normal((target_rank - half, current_rank))])
            
            U_new2 = np.array(U) @ U_mix.T
            V_new2 = np.array(V) @ V_mix.T
            W_new2 = np.array(W) @ W_mix.T
            
            U_ref2, V_ref2, W_ref2 = _als_step(U_new2, V_new2, W_new2, max_iter=200)
            
            if _validate_fast(U_ref2, V_ref2, W_ref2):
                return U_ref2, V_ref2, W_ref2
    
    return None

def _strassen_block_decompose():
    """
    Use Strassen-like recursive decomposition for sub-blocks.
    Split the 3x5 * 5x5 multiplication into blocks and apply Strassen where possible.
    """
    n, m, p = N, M, P
    # For 3x5 * 5x5: 
    # Split A (3x5) into rows, B (5x5) into columns
    # We can try to use the fact that 3 = 1+2, 5 = 2+3, 5 = 2+3
    # Apply Strassen to 2x2 blocks where applicable
    
    # Actually, let's try a different approach:
    # The schoolbook rank is 75. We know from the knowledge that recursive Strassen
    # can reduce to ~71. Let's implement a structured decomposition.
    
    # For now, return the schoolbook as a starting point
    rank = RANK_SCHOOLBOOK
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = N, M, P
    
    # Start with schoolbook
    U, V, W = _strassen_block_decompose()
    best_rank = U.shape[1]
    best_result = (U.copy(), V.copy(), W.copy())
    
    # Try to reduce rank via ALS-based optimization
    # Try target ranks from best_rank-1 down
    for target_rank in range(best_rank - 1, max(best_rank - 10, 0), -1):
        if _time_left() < 10.0:
            break
        
        result = _try_reduce_rank(U, V, W, target_rank, max_trials=30)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_result = (U.copy(), V.copy(), W.copy())
            # Continue trying to reduce further
        else:
            # Try a few more trials with different seeds
            for extra_trial in range(20):
                if _time_left() < 5.0:
                    break
                result = _try_reduce_rank(U, V, W, target_rank, max_trials=1)
                if result is not None:
                    U, V, W = result
                    best_rank = target_rank
                    best_result = (U.copy(), V.copy(), W.copy())
                    break
    
    # Final validation
    U_final, V_final, W_final = best_result
    if not _validate_fast(U_final, V_final, W_final):
        # Fallback to schoolbook
        U_final, V_final, W_final = _strassen_block_decompose()
    
    # Convert to lists (interface expects lists)
    U_list = [[float(x) for x in row] for row in U_final]
    V_list = [[float(x) for x in row] for row in V_final]
    W_list = [[float(x) for x in row] for row in W_final]
    
    return {"U": U_list, "V": V_list, "W": W_list}
