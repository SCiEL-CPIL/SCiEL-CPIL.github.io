"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import null_space, solve
from itertools import combinations

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Compare with identity tensor for matrix multiplication
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                expected = 1.0 if (i < n and j < m and k < p) else 0.0
                # Actually the matrix multiplication tensor is T[i,j,k] = 1 for all valid i,j,k
                # Wait, no. The matrix multiplication tensor T_{ijk} = A_{ij} B_{jk}
                # So T[i,j,k] = 1 for all i,j,k in the valid range.
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _rank_of_matrix(M):
    """Compute the rank of a matrix using SVD."""
    if M.size == 0:
        return 0
    return int(np.linalg.matrix_rank(M, tol=1e-10))

def _try_remove_term(U, V, W, idx, n, m, p):
    """
    Try to remove term idx from the decomposition.
    Check if the term can be expressed as a linear combination of remaining terms.
    For tensor decomposition, a term c contributes u_c ⊗ v_c ⊗ w_c.
    We need to check if u_idx ⊗ v_idx ⊗ w_idx is in the span of {u_c ⊗ v_c ⊗ w_c : c != idx}.
    
    This is complex. A simpler approach: check if removing the term still gives a valid decomposition
    by checking if the residual can be absorbed.
    
    Actually, for exact decomposition, we can use the following approach:
    The tensor T = sum_c u_c ⊗ v_c ⊗ w_c.
    If we remove term idx, we get T' = T - u_idx ⊗ v_idx ⊗ w_idx.
    We need T' to be expressible as sum_{c != idx} u_c ⊗ v_c ⊗ w_c.
    
    This is a non-trivial check. Let's use a different approach:
    Check linear dependence in the space of terms.
    """
    rank = U.shape[1]
    if rank <= 1:
        return False, U, V, W
    
    # Remove the term
    mask = np.ones(rank, dtype=bool)
    mask[idx] = False
    U_rem = U[:, mask]
    V_rem = V[:, mask]
    W_rem = W[:, mask]
    
    # Check if the removed term is in the span of remaining terms
    # This is a tensor rank reduction check
    # For simplicity, we'll check if the tensor formed by the removed term
    # is in the span of the remaining terms' tensors
    
    # Flatten all terms into vectors in the tensor space
    # Term c corresponds to vector vec(u_c ⊗ v_c ⊗ w_c) in R^{n*m*p}
    # But we need to be careful about the specific structure
    
    # Alternative: use the fact that if we can find coefficients alpha_c such that
    # u_idx = sum alpha_c u_c, v_idx = sum alpha_c v_c, w_idx = sum alpha_c w_c
    # then the term is redundant. But this is sufficient, not necessary.
    
    # Let's try a different approach: check if the matrix [u_idx, U_rem] has the same rank
    # and similarly for V and W, AND the cross conditions are satisfied.
    
    # Actually, let's just verify if the decomposition without term idx still works
    # by checking all entries
    new_U = U_rem
    new_V = V_rem
    new_W = W_rem
    new_rank = new_U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(new_rank):
                    val += new_U[i * m + j, c] * new_V[j * p + k, c] * new_W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False, U, V, W
    return True, new_U, new_V, new_W

def _greedy_reduce(U, V, W, n, m, p, deadline):
    """
    Greedy rank reduction: iteratively try to remove terms.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    removed = []
    while rank > 1:
        if time.time() > deadline:
            break
        found = False
        for idx in range(rank):
            if idx in removed:
                continue
            can_remove, new_U, new_V, new_W = _try_remove_term(U, V, W, idx, n, m, p)
            if can_remove:
                U = new_U
                V = new_V
                W = new_W
                removed.append(idx)
                rank -= 1
                found = True
                break
        if not found:
            break
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize schoolbook decomposition
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Try greedy reduction
    if time.time() < deadline:
        U, V, W = _greedy_reduce(U, V, W, n, m, p, deadline)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
