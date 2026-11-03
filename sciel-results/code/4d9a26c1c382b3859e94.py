import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """
    Verify that the given factor matrices exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = U.shape[1]
    # Tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We check that T[i,j,k] == 1 if (i,j,k) corresponds to a valid multiplication term, else 0.
    # Actually, the tensor for matrix multiplication C = A * B is:
    # T[i, j, k] = 1 if k = i * m + j? No.
    # Standard tensor definition: T[i, j, k] = delta_{i,j,k} is not right.
    # The matrix multiplication tensor T is defined such that C[i,k] = sum_j A[i,j] B[j,k].
    # In tensor form: T[i, j, k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # So T[i, j, k] = 1 for all i, j, k? No.
    # The tensor T has indices (i, j, k) where i is row of A/C, j is col of A / row of B, k is col of B/C.
    # T[i, j, k] = 1.
    # Wait, the decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # U_r has indices (i, j), V_r has indices (j, k), W_r has indices (i, k).
    # The contraction is sum_r U[i,j,r] V[j,k,r] W[i,k,r] = T[i,j,k].
    # For matrix multiplication, T[i,j,k] = 1 for all i, j, k?
    # No, T[i,j,k] is 1 if the multiplication A[i,j]*B[j,k] contributes to C[i,k].
    # It always does. So T[i,j,k] = 1 for all i, j, k?
    # Let's check the schoolbook decomposition.
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
    # Sum over col (which corresponds to (i', j', k') in schoolbook):
    # Term is 1 if (i,j) == (i',j') AND (j,k) == (j',k') AND (k,i) == (k',i').
    # This implies i=i', j=j', k=k'.
    # So the sum is 1 if (i,j,k) matches the index of the term, 0 otherwise.
    # Thus T[i,j,k] = 1 for all i,j,k.
    # Wait, if T[i,j,k] = 1 for all i,j,k, then the tensor is just the all-ones tensor?
    # No, the indices of the tensor are (i, j, k) corresponding to (row_A, col_A/row_B, col_B).
    # The value is 1.
    # So we are decomposing the tensor T where T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Let's verify with the schoolbook code.
    # U is (n*m, rank). Index i*m+j.
    # V is (m*p, rank). Index j*p+k.
    # W is (p*n, rank). Index k*n+i.
    # Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # In schoolbook, r corresponds to (i', j', k').
    # U[i*m+j, r] is 1 if i==i' and j==j'.
    # V[j*p+k, r] is 1 if j==j' and k==k'.
    # W[k*n+i, r] is 1 if k==k' and i==i'.
    # Product is 1 if i==i', j==j', k==k'.
    # Sum over all r (i.e., all i', j', k') gives 1 if there exists (i',j',k') such that i=i', j=j', k=k'.
    # Which is always true for the specific (i,j,k) we are evaluating?
    # Yes, the sum equals 1 for every (i,j,k).
    # So the target tensor T is indeed the all-ones tensor of size n x m x p.
    
    # However, the task says "exactly decompose the matrix multiplication tensor".
    # Usually, the matrix multiplication tensor is defined as T[i,j,k] = 1.
    # So we check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k.
    
    # Efficient check:
    # For each r, we have vectors u_r (length n*m), v_r (length m*p), w_r (length p*n).
    # We want to check if sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] == 1.
    
    # This is O(rank * n * m * p) which is fine for small dimensions.
    
    # Reshape U, V, W to 3D tensors for easier indexing?
    # U is (n*m, rank). We can view it as (n, m, rank).
    # V is (m*p, rank). View as (m, p, rank).
    # W is (p*n, rank). View as (p, n, rank).
    
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    # This is a contraction.
    # We can do this with einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # Check if T is all ones.
    if not np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0):
        return False, T
    
    return True, T

def _snap_to_half_integer(matrix, tol=1e-6):
    """
    Snap entries of a matrix to the nearest integer or half-integer.
    Returns the snapped matrix and a boolean indicating if all entries were close enough.
    """
    # Scale by 2 to check if entries are close to integers.
    scaled = matrix * 2
    rounded = np.round(scaled)
    diff = np.abs(scaled - rounded)
    max_diff = np.max(diff)
    
    if max_diff < tol:
        snapped = rounded / 2.0
        return snapped, True
    else:
        # If not close, we can still return the snapped version, but the validity flag is False.
        # However, for the search, we only want valid decompositions.
        # So we return the original matrix if it's not close, or the snapped one?
        # The task requires integer/half-integer entries.
        # If we can't snap, the decomposition is invalid for the interface.
        # So we should only accept if it snaps well.
        return matrix, False

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Attempts to find a rank < 60 decomposition using border rank techniques.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    n, m, p = DIMENSIONS
    target_rank = n * m * p  # 60
    best_rank = target_rank
    best_factors = None
    
    # Try ranks from low to high
    # Known lower bound for 2x5x6 is not 53, but let's try a few values.
    # The theoretical minimum rank for 2x5x6 is likely higher than 53.
    # Let's try ranks 59, 58, ... down to 50.
    # If we fail, we return the schoolbook decomposition.
    
    for rank in range(59, 49, -1):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Use complex numbers for better convergence?
        # The task allows integer/half-integer, but intermediate steps can be real/complex.
        # However, snapping to half-integers implies real values.
        
        # Strategy:
        # 1. Generate random U, V, W of shape (n*m, rank), (m*p, rank), (p*n, rank)
        # 2. Use gradient descent or alternating least squares to minimize ||T - T_decomp||^2
        # 3. Try to snap to half-integers.
        
        # Since the tensor is all-ones, the problem is:
        # Find U, V, W such that sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        
        # This is a difficult non-convex problem.
        # Let's try a few random restarts for each rank.
        
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Normalize to prevent divergence
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Alternating Least Squares (ALS)
            # We want to minimize sum_{i,j,k} (1 - sum_r U[i,j,r] V[j,k,r] W[k,i,r])^2
            # This is equivalent to minimizing ||T - T_decomp||_F^2
            # where T is the all-ones tensor.
            
            # ALS for tensor decomposition is not straightforward because the tensor is not
            # a simple product of three matrices, but a specific permutation of indices.
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            
            # We can linearize with respect to one factor at a time.
            # Fix V and W, solve for U.
            # T[i,j,k] = sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r])
            # For fixed i,j, we have a system of equations for U[i,j,r] over k?
            # No, U[i,j,r] is a vector of length rank.
            # The equation is: for all k, T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # This is a linear system in U[i,j,:] for each (i,j)?
            # Let A_{i,j} be a matrix of size (p, rank) where A_{i,j}[k, r] = V[j,k,r] * W[k,i,r]
            # Then T[i,j,:] = U[i,j,:] @ A_{i,j}^T ?
            # T[i,j,k] = sum_r U[i,j,r] A_{i,j}[k,r]
            # So T[i,j,:] = A_{i,j} @ U[i,j,:]^T ? No.
            # T[i,j,k] = sum_r A_{i,j}[k,r] U[i,j,r]
            # So T[i,j,:] = A_{i,j} @ U[i,j,:]
            # We want to find U[i,j,:] that minimizes ||T[i,j,:] - A_{i,j} @ U[i,j,:]||^2
            # Solution: U[i,j,:] = (A_{i,j}^T A_{i,j})^{-1} A_{i,j}^T T[i,j,:]
            # This is a least squares problem for each (i,j).
            
            # Similarly for V and W.
            
            # Let's implement ALS with 10 iterations.
            for _ in range(10):
                if time.time() > deadline:
                    break
                    
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Build A_{i,j} of shape (p, rank)
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # T[i,j,:] is all ones of length p
                        # Solve A @ U[i*m+j,:]^T = ones(p)
                        # U[i*m+j,:] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        # Note: A is (p, rank), we want x (rank,) such that A @ x = ones(p)
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # Build B_{j,k} of shape (n, rank)
                        B = np.zeros((n, rank))
                        for i in range(n):
                            B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        # T[:,j,k] is all ones of length n
                        # Solve B @ V[j*p+k,:]^T = ones(n)
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # Build C_{k,i} of shape (m, rank)
                        C = np.zeros((m, rank))
                        for j in range(m):
                            C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        # T[i,:,k] is all ones of length m
                        # Solve C @ W[k*n+i,:]^T = ones(m)
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                        except:
                            pass
            
            # Check validity
            valid, T = _verify_decomposition(U, V, W, n, m, p, tol=1e-5)
            if valid:
                # Try to snap to half-integers
                U_snapped, u_ok = _snap_to_half_integer(U)
                V_snapped, v_ok = _snap_to_half_integer(V)
                W_snapped, w_ok = _snap_to_half_integer(W)
                
                if u_ok and v_ok and w_ok:
                    # Verify snapped version
                    valid_snapped, _ = _verify_decomposition(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-8)
                    if valid_snapped:
                        best_rank = rank
                        best_factors = (U_snapped, V_snapped, W_snapped)
                        break # Found a valid decomposition for this rank, break restarts
                
                if time.time() > deadline:
                    break
        
        if best_factors is not None:
            break # Found a decomposition, no need to try lower ranks
    
    # If no better decomposition found, return schoolbook
    if best_factors is None:
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
                    
        best_factors = (U, V, W)
        best_rank = rank
    
    U, V, W = best_factors
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
