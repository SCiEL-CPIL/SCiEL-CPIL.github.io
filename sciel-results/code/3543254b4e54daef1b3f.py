"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or as low as possible) tensor decomposition
    for 3x4 * 4x4 matrix multiplication.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank = 48).
    2. Use a heuristic/local search to try to reduce the rank by finding
       dependencies in the tensor or using known lower-bound constructions.
    3. Deadline guard to ensure we return within time limits.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to find lower rank decompositions through local search
    # The key insight: if we can express one term as a linear combination of others,
    # we can eliminate it.
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify that the decomposition is correct."""
        n, m, p = DIMENSIONS
        # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should equal 1 if i,j,k are valid, 0 otherwise
        # For matrix mult: T[i,j,k] = delta_{i,j,k} where the product A*B has C[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] should be 1 for all i,j,k (each term contributes to exactly one output)
        
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        
        r = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    def try_remove_term(U_mat, V_mat, W_mat, term_idx):
        """Try to remove a term by expressing it as combination of others."""
        n, m, p = DIMENSIONS
        r = U_mat.shape[1]
        
        # Create system: we want to find coefficients c such that
        # U[:, term] = sum_{s != term} c_s * U[:, s]
        # V[:, term] = sum_{s != term} c_s * V[:, s]  
        # W[:, term] = sum_{s != term} c_s * W[:, s]
        
        # This is a linear system. Build the matrix.
        other_indices = [i for i in range(r) if i != term_idx]
        
        # Stack all constraints
        A = np.vstack([
            U_mat[:, other_indices].T,  # (n*m, r-1) -> (r-1, n*m)
            V_mat[:, other_indices].T,  # (m*p, r-1) -> (r-1, m*p)
            W_mat[:, other_indices].T   # (p*n, r-1) -> (r-1, p*n)
        ])
        b = np.concatenate([
            U_mat[:, term_idx],
            V_mat[:, term_idx],
            W_mat[:, term_idx]
        ])
        
        # Solve least squares
        try:
            coeffs, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            return None
        
        # Check if solution is exact (within tolerance)
        residual = np.linalg.norm(A @ coeffs - b)
        if residual > 1e-6:
            return None
        
        # Check if coefficients are "nice" (integers or half-integers)
        # For now, accept any solution
        new_r = r - 1
        new_U = np.delete(U_mat, term_idx, axis=1)
        new_V = np.delete(V_mat, term_idx, axis=1)
        new_W = np.delete(W_mat, term_idx, axis=1)
        
        # Verify the new decomposition
        if verify_decomposition(new_U.tolist(), new_V.tolist(), new_W.tolist()):
            return (new_U, new_V, new_W, coeffs)
        
        return None
    
    # Try removing each term
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        U_mat = np.array(best_U, dtype=float)
        V_mat = np.array(best_V, dtype=float)
        W_mat = np.array(best_W, dtype=float)
        r = U_mat.shape[1]
        
        removed = False
        for term_idx in range(r):
            if time.time() > deadline:
                break
            
            result = try_remove_term(U_mat, V_mat, W_mat, term_idx)
            if result is not None:
                new_U, new_V, new_W, coeffs = result
                best_U = new_U.astype(int) if np.allclose(new_U, np.round(new_U)) else new_U
                best_V = new_V.astype(int) if np.allclose(new_V, np.round(new_V)) else new_V
                best_W = new_W.astype(int) if np.allclose(new_W, np.round(new_W)) else new_W
                best_rank = new_U.shape[1]
                removed = True
                break
        
        if not removed:
            break
    
    # Convert to lists
    def to_list(mat):
        if np.allclose(mat, np.round(mat)):
            return np.round(mat).astype(int).tolist()
        else:
            # Try to express as fractions
            return mat.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END

