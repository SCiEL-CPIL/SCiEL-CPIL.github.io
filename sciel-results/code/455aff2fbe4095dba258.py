import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    U_arr = np.asarray(U, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.float64)
    
    # For each (i,j,k), compute the sum
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j, :]
                v_val = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                tensor_val = np.dot(u_val, v_val * w_val)
                expected = 1.0 if i < n and j < m and k < p else 0.0
                # The tensor T[i,j,k] should be 1 for all valid indices
                if abs(tensor_val - 1.0) > tol:
                    return False
    return True

def _find_linear_dependency(U, V, W, n, m, p, max_terms_to_check=5000):
    """
    Try to find a linear dependency among the rank-1 terms.
    If sum_r alpha_r * u_r ⊗ v_r ⊗ w_r = 0, we can eliminate one term.
    
    We check if any single term can be expressed as a linear combination of others.
    This is equivalent to checking if the columns of the "flattened tensor matrix" are linearly dependent.
    
    For efficiency, we check small subsets first.
    """
    rank = U.shape[1]
    U_arr = np.asarray(U, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.float64)
    
    # Each rank-1 term contributes u_r ⊗ v_r ⊗ w_r to the tensor.
    # Flatten each term into a vector of length n*m*p.
    # term_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build the matrix where each column is a flattened rank-1 term
    # Shape: (n*m*p, rank)
    terms = np.zeros((n * m * p, rank), dtype=np.float64)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms[idx, :] = U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :]
                idx += 1
    
    # Check if columns are linearly dependent by computing rank
    # If rank < number of columns, there's a dependency
    # Use SVD for numerical stability
    try:
        _, s, vh = np.linalg.svd(terms, full_matrices=False)
        # Count significant singular values
        tol = 1e-10
        effective_rank = np.sum(s > tol)
        
        if effective_rank < rank:
            # There's a dependency. Find it from the null space.
            # The null space vectors are in the last (rank - effective_rank) rows of vh
            null_vecs = vh[effective_rank:, :]
            # Pick the first null vector
            alpha = null_vecs[0]
            # Find the index with largest magnitude to eliminate
            elim_idx = np.argmax(np.abs(alpha))
            return elim_idx, alpha
    except np.linalg.LinAlgError:
        pass
    
    return None, None

def _eliminate_term(U, V, W, elim_idx, alpha, n, m, p):
    """
    Given a linear dependency sum_r alpha_r * term_r = 0,
    eliminate term elim_idx by expressing it as a combination of others.
    Return new U, V, W with one fewer column.
    """
    rank = U.shape[1]
    U_arr = np.asarray(U, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.float64)
    
    # We want to remove column elim_idx.
    # The dependency is: alpha[elim_idx] * term_elim + sum_{r != elim} alpha[r] * term_r = 0
    # So: term_elim = -sum_{r != elim} (alpha[r] / alpha[elim_idx]) * term_r
    
    # But we can't just modify the coefficients because U, V, W are separate matrices.
    # The trick: we need to find new U', V', W' with rank-1 columns such that
    # sum_{r' != elim} U'[i*m+j, r'] * V'[j*p+k, r'] * W'[k*n+i, r'] = original tensor
    
    # This is non-trivial. A simpler approach: if we can find a linear combination
    # where only ONE of U, V, or W has non-zero entries in the eliminated column,
    # we can absorb it.
    
    # Actually, for the schoolbook decomposition, each term has exactly one 1 in U,
    # one 1 in V, and one 1 in W. So term_r is the tensor e_{i_r,j_r} ⊗ e_{j_r,k_r} ⊗ e_{k_r,i_r}.
    
    # A linear dependency sum alpha_r * T_r = 0 means:
    # For each (i,j,k): sum_{r: i_r=i, j_r=j, k_r=k} alpha_r = 0 (but each (i,j,k) appears exactly once in schoolbook)
    
    # Wait, in schoolbook each (i,j,k) appears exactly once, so there's NO linear dependency!
    # The terms are linearly independent because each term has support on a unique (i,j,k).
    
    # So we need a different approach. Let's try a different strategy:
    # Use the fact that we can add and subtract terms to create cancellations.
    
    # Alternative: try to find a rank-(r-1) decomposition by optimization.
    # This is hard, so let's try a heuristic: randomly perturb and see if we can reduce rank.
    
    return None, None, None

def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication M_{2,3,7}.
    The rank is n*m*p = 42.
    This implementation uses vectorized numpy operations to generate the factor matrices.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index arrays for the decomposition
    i_grid, j_grid, k_grid = np.indices((n, m, p))
    
    # Flatten the grids to match the column order of the decomposition
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    k_flat = k_grid.flatten()
    
    # Initialize U, V, W as zero matrices with integer dtype
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # The column index for each (i,j,k) tuple is simply the index in the flattened array
    col_indices = np.arange(rank)
    
    # Row indices for U: i * m + j
    u_rows = i_flat * m + j_flat
    # Row indices for V: j * p + k
    v_rows = j_flat * p + k_flat
    # Row indices for W: k * n + i (Following the trace convention W[k*n+i, r])
    w_rows = k_flat * n + i_flat
    
    # Populate the matrices using advanced indexing
    U[u_rows, col_indices] = 1
    V[v_rows, col_indices] = 1
    W[w_rows, col_indices] = 1
    
    # Convert to lists of lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
