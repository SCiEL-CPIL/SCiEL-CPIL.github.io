"""Matrix multiplication 3x4x6 rank minimization."""
import numpy as np
from itertools import product
import time
import sys

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x6 matrix.
    
    The matrix multiplication tensor T has indices (i, j, k) where:
    - i in [0, n) = 3 (rows of A)
    - j in [0, m) = 4 (cols of A / rows of B)  
    - k in [0, p) = 6 (cols of B)
    
    T[i,j,k] = 1 if we multiply A[i,j] * B[j,k] to get C[i,k]
    
    The decomposition is:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    where:
    - U has shape (n*m, rank) = (12, rank), indexed by (i*m+j)
    - V has shape (m*p, rank) = (24, rank), indexed by (j*p+k)
    - W has shape (p*n, rank) = (18, rank), indexed by (k*n+i)
    """
    n, m, p = DIMENSIONS
    
    # Build the multiplication tensor T[i,j,k]
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1  # Every triple contributes to the product
    
    # We want to find U, V, W such that:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Start with the schoolbook decomposition (rank = n*m*p = 72)
    # Then try to reduce the rank using ALS (Alternating Least Squares)
    
    # First, let's try a few known lower bounds and constructions
    # For 3x4x6, the minimum rank is not well-known, but we can try
    # various approaches.
    
    # Approach 1: Try to find a low-rank decomposition using numerical methods
    # We'll use a greedy approach: start with schoolbook, then try to merge terms
    
    # Let's try a direct approach: use the fact that for small dimensions,
    # we can try all possible low-rank decompositions or use optimization.
    
    # Strategy: Start with rank = n*m*p and try to reduce it.
    # For each rank r, try to find U, V, W of that rank.
    
    # Use a simple approach: try ranks from a lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # Upper bound: n*m*p = 72
    
    # Let's try a few candidate ranks and use numerical optimization
    # to see if we can find an exact (or near-exact) decomposition.
    
    # For efficiency, let's first try the schoolbook and then attempt
    # to reduce via ALS with integer/half-integer constraints.
    
    # Actually, for this small problem, let's try a more systematic approach:
    # 1. Start with schoolbook (rank 72)
    # 2. Try to reduce rank by finding linear dependencies
    
    # Let's implement a simple rank reduction algorithm:
    # - Start with the schoolbook decomposition
    # - Try to find if any term can be removed or combined
    
    # For now, let's return the schoolbook decomposition as a baseline
    # and try to improve it.
    
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
    
    # Now try to reduce the rank using ALS
    # We'll try various target ranks and see if we can achieve an exact decomposition
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to reduce rank
    # For each target rank, try to find a decomposition
    min_possible = max(n * m, m * p, p * n)
    
    # Try ranks from min_possible up to current best
    for target_rank in range(min_possible, best_rank + 1):
        if target_rank >= best_rank:
            break
        
        # Try to find a decomposition of this rank
        success = _try_find_decomposition(T, n, m, p, target_rank, U, V, W)
        if success is not None:
            U_new, V_new, W_new = success
            # Verify it's exact
            if _verify_decomposition(U_new, V_new, W_new, T, n, m, p):
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, target_rank
                break  # Found the minimum rank
    
    # Convert to lists with integer/half-integer values
    U_list = _to_half_int_list(best_U)
    V_list = _to_half_int_list(best_V)
    W_list = _to_half_int_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}


def _try_find_decomposition(T, n, m, p, target_rank, U_init, V_init, W_init):
    """Try to find a decomposition of the given rank using ALS."""
    # Initialize with scaled version of schoolbook
    U = np.zeros((n * m, target_rank), dtype=np.float64)
    V = np.zeros((m * p, target_rank), dtype=np.float64)
    W = np.zeros((p * n, target_rank), dtype=np.float64)
    
    # Initialize randomly or from schoolbook
    for r in range(target_rank):
        # Spread the schoolbook terms across the target rank
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i * m + j) * p + k < r * (n * m * p) // target_rank:
                        continue
                    U[i * m + j, r] = 1.0 / max(1, target_rank // (n * m * p))
                    V[j * p + k, r] = 1.0 / max(1, target_rank // (n * m * p))
                    W[k * n + i, r] = 1.0 / max(1, target_rank // (n * m * p))
    
    # Run ALS iterations
    for iteration in range(50):
        # Update U
        U = _als_update_U(T, V, W, n, m, p)
        # Update V
        V = _als_update_V(T, U, W, n, m, p)
        # Update W
        W = _als_update_W(T, U, V, n, m, p)
    
    # Check if we got an exact decomposition
    if _verify_decomposition(U, V, W, T, n, m, p, tol=1e-6):
        return U, V, W
    
    return None


def _als_update_U(T, V, W, n, m, p):
    """Update U in ALS."""
    U = np.zeros((n * m, W.shape[1]), dtype=np.float64)
    rank = W.shape[1]
    
    for r in range(rank):
        # For each (i, j), solve for U[i*m+j, r]
        for i in range(n):
            for j in range(m):
                # T[i,j,k] = sum_r' U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r']
                # For fixed r, we want:
                # T[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] + other terms
                # This is complex, so use a simpler approach
                pass
    
    # Simplified: use least squares
    for i in range(n):
        for j in range(m):
            # Build the linear system for U[i*m+j, :]
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            A_mat = np.zeros((p, rank), dtype=np.float64)
            for k in range(p):
                for r in range(rank):
                    A_mat[k, r] = V[j * p + k, r] * W[k * n + i, r]
            b_vec = np.array([T[i, j, k] for k in range(p)], dtype=np.float64)
            # Solve least squares
            try:
                U[i * m + j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            except:
                U[i * m + j, :] = 0
    
    return U


def _als_update_V(T, U, W, n, m, p):
    """Update V in ALS."""
    V = np.zeros((m * p, W.shape[1]), dtype=np.float64)
    rank = W.shape[1]
    
    for j in range(m):
        for k in range(p):
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            A_mat = np.zeros((n, rank), dtype=np.float64)
            for i in range(n):
                for r in range(rank):
                    A_mat[i, r] = U[i * m + j, r] * W[k * n + i, r]
            b_vec = np.array([T[i, j, k] for i in range(n)], dtype=np.float64)
            try:
                V[j * p + k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            except:
                V[j * p + k, :] = 0
    
    return V


def _als_update_W(T, U, V, n, m, p):
    """Update W in ALS."""
    W = np.zeros((p * n, U.shape[1]), dtype=np.float64)
    rank = U.shape[1]
    
    for k in range(p):
        for i in range(n):
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            A_mat = np.zeros((m, rank), dtype=np.float64)
            for j in range(m):
                for r in range(rank):
                    A_mat[j, r] = U[i * m + j, r] * V[j * p + k, r]
            b_vec = np.array([T[i, j, k] for j in range(m)], dtype=np.float64)
            try:
                W[k * n + i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            except:
                W[k * n + i, :] = 0
    
    return W


def _verify_decomposition(U, V, W, T, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute T[i,j,k] from the decomposition
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - T[i, j, k]) > tol:
                    return False
    return True


def _to_half_int_list(matrix):
    """Convert a numpy array to a list of lists with half-integer values."""
    result = []
    for row in matrix:
        row_list = []
        for val in row:
            # Round to nearest half-integer
            half_int = round(val * 2) / 2
            row_list.append(half_int)
        result.append(row_list)
    return result
# EVOLVE-BLOCK-END
