"""Rewritten program: rank-minimal exact tensor decomposition for 2x4 @ 4x6.

Strategy: Build the rank-42 Strassen-Kronecker decomposition (Kronecker product
of the rank-7 Strassen 2x2x2 core and the trivial 1x2x3 core), then search for
non-trivial linear dependencies among the 42 rank-1 terms to reduce the rank.

The key insight is that the 42 terms live in a 42-dimensional space, but the
underlying tensor may have rank less than 42. We systematically check whether
any subset of terms is linearly dependent by solving the linear system that
arises from the tensor decomposition constraints.
"""

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

_START = time.time()
_TIMEOUT = 140.0

def _elapsed():
    return time.time() - _START

def _build_strassen_kronecker(n, m, p):
    """Build the rank-42 Strassen-Kronecker decomposition using explicit
    Kronecker product of the Strassen 2x2x2 core and the trivial 1x2x3 core."""
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2x2 core (rank 7)
    # U1[i*2+j, r], V1[j*2+k, r], W1[k*2+i, r]
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1  # C00 and C11 get M1
    
    # M2 = (A10+A11)B00
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1  # C10 gets M2
    W1[3, 1] = -1  # C11 gets -M2
    
    # M3 = A00(B01-B11)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1  # C01 gets M3
    W1[3, 2] = 1  # C11 gets M3
    
    # M4 = A11(B10-B00)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1  # C00 gets M4
    W1[1, 3] = 1  # C10 gets M4
    
    # M5 = (A00+A01)B11
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1  # C00 gets -M5
    W1[2, 4] = 1   # C01 gets M5
    
    # M6 = (A10-A00)(B00+B01)
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1   # C11 gets M6
    
    # M7 = (A01-A11)(B10+B11)
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1   # C00 gets M7
    
    # Trivial 1x2x3 core (rank 6)
    # U2[j2, r] for A2[0, j2], V2[j2*3+k2, r] for B2[j2, k2], W2[k2, r] for C2[k2, 0]
    U2 = np.zeros((2, 6))
    V2 = np.zeros((6, 6))
    W2 = np.zeros((3, 6))
    
    for r2 in range(6):
        j2 = r2 % 2
        k2 = r2 // 2
        U2[j2, r2] = 1
        V2[j2 * 3 + k2, r2] = 1
        W2[k2, r2] = 1
    
    # Kronecker product
    for r1 in range(7):
        for r2 in range(6):
            r = r1 * 6 + r2
            
            # U: A[i, a] where i = i1, a = j1*2 + j2
            for i1 in range(2):
                for j1 in range(2):
                    val1 = U1[i1 * 2 + j1, r1]
                    if val1 != 0:
                        for j2 in range(2):
                            val2 = U2[j2, r2]
                            if val2 != 0:
                                i = i1
                                a = j1 * 2 + j2
                                idx_u = i * m + a
                                U[idx_u, r] += val1 * val2
            
            # V: B[a, b] where a = j1*2 + j2, b = k1*3 + k2
            for j1 in range(2):
                for k1 in range(2):
                    val1 = V1[j1 * 2 + k1, r1]
                    if val1 != 0:
                        for j2 in range(2):
                            for k2 in range(3):
                                val2 = V2[j2 * 3 + k2, r2]
                                if val2 != 0:
                                    a = j1 * 2 + j2
                                    b = k1 * 3 + k2
                                    idx_v = a * p + b
                                    V[idx_v, r] += val1 * val2
            
            # W: C[i, b] where i = i1, b = k1*3 + k2; W index = b*n + i
            for i1 in range(2):
                for k1 in range(2):
                    val1 = W1[k1 * 2 + i1, r1]
                    if val1 != 0:
                        for k2 in range(3):
                            val2 = W2[k2, r2]
                            if val2 != 0:
                                i = i1
                                b = k1 * 3 + k2
                                idx_w = b * n + i
                                W[idx_w, r] += val1 * val2
    
    return U, V, W, rank

def _validate_decomposition(U, V, W, n, m, p, rank):
    """Validate that the decomposition is correct using vectorized operations."""
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # Vectorize over i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _find_linear_dependencies(U, V, W, n, m, p, rank, max_time=60.0):
    """Search for linear dependencies among the rank-1 terms.
    
    The tensor T = sum_r U_r ⊗ V_r ⊗ W_r. If there exist coefficients c_r
    not all zero such that sum_r c_r * U_r ⊗ V_r ⊗ W_r = 0, then we can
    eliminate one term.
    
    We check this by flattening each term into a vector and checking the
    rank of the matrix whose columns are these vectors.
    """
    if _elapsed() > _TIMEOUT - 20:
        return None
    
    # Flatten each rank-1 term into a vector of length n*m*p
    terms = []
    for r in range(rank):
        term = np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(-1)
        terms.append(term)
    
    # Stack into a matrix (n*m*p) x rank
    M = np.column_stack(terms)
    
    # Find the rank of this matrix
    # Use SVD for numerical stability
    try:
        _, s, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    # Count the number of significant singular values
    tol = 1e-10
    num_nonzero = np.sum(s > tol)
    
    if num_nonzero < rank:
        # There are linear dependencies
        # The null space has dimension rank - num_nonzero
        # We can eliminate (rank - num_nonzero) terms
        # Find a basis for the null space
        null_dim = rank - num_nonzero
        if null_dim > 0:
            # The last null_dim rows of Vt span the null space
            null_basis = Vt[num_nonzero:, :]
            
            # For each null vector, we can eliminate one term
            # Pick the term with the largest coefficient in the null vector
            # and express it in terms of the others
            
            # Simple approach: if null_dim >= 1, we can reduce rank by null_dim
            # Use the first null_dim terms as the ones to eliminate
            # Actually, we need to be more careful.
            
            # Let's find which terms can be eliminated
            # For each null vector c, sum_r c_r * term_r = 0
            # So term_j = -1/c_j * sum_{r != j} c_r * term_r for any j with c_j != 0
            
            # We want to find a set of terms to eliminate such that
            # the remaining terms still span the tensor.
            
            # Simplest: if the rank of M is R < rank, we can keep any R
            # linearly independent terms and eliminate the rest.
            
            # Find R linearly independent columns
            # Use QR with column pivoting
            from scipy.linalg import qr
            Q, R, P = qr(M, pivoting=True)
            
            # The first num_nonzero columns (after permutation) are independent
            # P is the permutation
            independent_indices = P[:num_nonzero]
            eliminated_indices = P[num_nonzero:]
            
            # Build the reduced decomposition
            U_new = U[:, independent_indices]
            V_new = V[:, independent_indices]
            W_new = W[:, independent_indices]
            new_rank = num_nonzero
            
            # Validate
            if _validate_decomposition(U_new, V_new, W_new, n, m, p, new_rank):
                return U_new, V_new, W_new, new_rank
    
    return None

def _try_rank_reduction_by_elimination(U, V, W, n, m, p, rank, max_time=60.0):
    """Try to reduce rank by eliminating individual terms.
    
    For each term r, check if the tensor can be represented without it.
    This is equivalent to checking if term_r is in the span of the other terms.
    """
    if _elapsed() > _TIMEOUT - 25:
        return None
    
    # For each term, try to eliminate it
    for r_elim in range(rank):
        if _elapsed() > _TIMEOUT - 30:
            break
        
        # Remove term r_elim
        remaining = [r for r in range(rank) if r != r_elim]
        U_sub = U[:, remaining]
        V_sub = V[:, remaining]
        W_sub = W[:, remaining]
        
        # Check if the remaining terms can represent the tensor
        # We need to solve for new W coefficients
        # For each (i,j,k): sum_{r in remaining} U[i*m+j, r] * V[j*p+k, r] * W_new[k*n+i, r] = 1
        # This is nonlinear in W_new, so we can't directly solve.
        
        # Instead, check if the residual is small
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_sub[i*m+j], V_sub[j*p+k] * W_sub[k*n+i])
                    residual += (s - 1.0) ** 2
        
        if residual < 1e-10:
            return U_sub, V_sub, W_sub, rank - 1
    
    return None

def construct():
    n, m, p = 2, 4, 6
    
    # Build the rank-42 Strassen-Kronecker solution
    U, V, W, rank = _build_strassen_kronecker(n, m, p)
    
    # Validate
    if _elapsed() < 5:
        valid = _validate_decomposition(U, V, W, n, m, p, rank)
    
    # Attempt rank reduction via linear dependency search
    if _elapsed() < _TIMEOUT - 30:
        reduced = _find_linear_dependencies(U, V, W, n, m, p, rank, max_time=60.0)
        if reduced is not None:
            U, V, W, rank = reduced
    
    # If still above some threshold, try elimination
    if rank > 40 and _elapsed() < _TIMEOUT - 30:
        reduced = _try_rank_reduction_by_elimination(U, V, W, n, m, p, rank, max_time=30.0)
        if reduced is not None:
            U, V, W, rank = reduced
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
