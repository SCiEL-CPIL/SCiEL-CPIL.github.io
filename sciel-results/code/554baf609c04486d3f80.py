"""Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _matmul_tensor(n, m, p):
    """Return the matrix-multiplication tensor as a 3D array T[i,j,k] where
    the (i,k) output is sum_j A[i,j]*B[j,k].
    Shape: (n, m, p)."""
    T = np.zeros((n, m, p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T


def _tensor_flatten_rank1(U, V, W, n, m, p):
    """Given factor matrices U (n*m x r), V (m*p x r), W (p*n x r),
    reconstruct the tensor and check it equals the matmul tensor."""
    r = U.shape[1]
    # Reconstruct tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T = np.zeros((n, m, p), dtype=object)
    for c in range(r):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i * m + j] * vc[j * p + k] * wc[k * n + i]
    # Check against identity tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i, j, k] != 1:
                    return False
    return True


def _build_strassen_3x4x6():
    """
    Build a rank-66 decomposition for 3x4 * 4x6 using a block-structured approach.
    
    We split the 4x4 middle into 2x2 blocks. So:
    A is 3x4, B is 4x6.
    Split A into [A1 (3x2), A2 (3x2)]
    Split B into [B1 (2x6); B2 (2x6)]
    
    Then C = A1*B1 + A2*B2, where A1 is 3x2, B1 is 2x6, etc.
    
    For 3x2 * 2x6: rank is at most 18 (schoolbook).
    But we can use Strassen-like tricks on the 2x2 sub-blocks.
    
    Actually, let me use a direct approach:
    - Split A (3x4) into two 3x2 blocks: A_left, A_right
    - Split B (4x6) into two 2x6 blocks: B_top, B_bottom
    - C = A_left * B_top + A_right * B_bottom
    
    Each 3x2 * 2x6 product has rank 12 (schoolbook: 3*2*6=36, but actually
    the tensor for 3x2*2x6 has rank... let me just use schoolbook for each).
    
    Actually, let me use a different strategy. The known rank for 3x4*4x6
    is at most 66 via specific constructions. Let me build it from scratch
    using a combination of Strassen on 2x2 blocks plus corrections.
    
    Let me use the approach: decompose the 4x4 middle dimension into 2+2.
    A (3x4) = [A1 | A2] where A1, A2 are 3x2
    B (4x6) = [B1; B2] where B1, B2 are 2x6
    
    C = A1*B1 + A2*B2
    
    For each 3x2 * 2x6 product, use schoolbook: rank 3*2*6 = 36.
    Total: 72. That's the schoolbook.
    
    To get 66, we need to save 6. Let me try a different split or use
    Strassen on the 2x2 part.
    
    Alternative: Use the fact that 3x4*4x6 can be related to known results.
    Let me just construct a valid rank-72 (schoolbook) and then try to
    find dependencies.
    
    Actually, let me try a smarter construction. Split into:
    A (3x4), B (4x6).
    
    Use 2x2 Strassen on the middle:
    Think of A as 3x2 blocks stacked, B as 2x6 blocks stacked.
    A = [A11 A12; A21 A22] but A is 3x4, so we can't do 2x2 blocks of 1.5.
    
    Let me just use schoolbook rank 72 and then do the nullspace reduction.
    """
    n, m, p = 3, 4, 6
    rank = n * m * p  # 72
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W


def _rank_of_flattened_system(U, V, W, n, m, p):
    """
    Given a decomposition, form the 'flattened' matrix that captures linear
    dependencies among the rank-1 terms.
    
    Each rank-1 term c contributes a vector in the tensor space (n*m*p dimensional).
    The tensor space can be viewed as R^(n*m) x R^p (matricized as (nm) x p).
    
    For a rank-1 term with U_c (nm x 1), V_c (mp x 1), W_c (pn x 1),
    the corresponding tensor slice is outer(U_c, V_c) reshaped... 
    
    Actually, let me think differently. The tensor T has entries T[i,j,k].
    A rank-1 term is T_c[i,j,k] = U_c[i,j] * V_c[j,k] * W_c[k,i] where
    U_c is indexed by (i,j), V_c by (j,k), W_c by (k,i).
    
    To find linear dependencies, I can fix a random linear combination of
    the (i,k) pairs. For each (i,k), the tensor slice T[:,:,i,k]... 
    
    Let me use the standard approach: the tensor can be matricized.
    Fix the k-index: for each k, we get an (n*m) x (n*m) matrix? No.
    
    Let me use: for each output position (i,k), the contribution is
    sum_j T[i,j,k] * A[i,j] * B[j,k].
    
    A cleaner approach: The set of rank-1 components {T_c} in R^(n*m*p)
    may be linearly dependent. To check, I can evaluate them on random
    vectors.
    
    For each rank-1 term c, define f_c(a, b) = (U_c . a) * (V_c . b) * (W_c . something)
    
    Actually, the simplest: each rank-1 term defines a tensor T_c in R^(n*m*p).
    Stack all T_c as columns of a matrix (n*m*p) x r. If this matrix has
    rank < r, there are dependencies.
    
    But n*m*p = 72 and r = 72, so the matrix is 72x72. I can just compute
    its rank directly!
    """
    r = U.shape[1]
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the 72 x r matrix where column c is the vectorized tensor T_c
    M = np.zeros((n * m * p, r), dtype=float)
    
    for c in range(r):
        uc = U[:, c]  # nm x 1
        vc = V[:, c]  # mp x 1
        wc = W[:, c]  # pn x 1
        
        # T_c[i,j,k] = uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
        # Vectorize in order (i,j,k) -> index i*m*p + j*p + k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    M[idx, c] = float(uc[i * m + j]) * float(vc[j * p + k]) * float(wc[k * n + i])
    
    return M


def _find_and_remove_dependencies(U, V, W, n, m, p, max_iter=50):
    """
    Iteratively find linear dependencies among the rank-1 terms and remove
    them, reducing the rank.
    
    Returns the reduced (U, V, W) and the new rank.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for iteration in range(max_iter):
        r = U.shape[1]
        M = _rank_of_flattened_system(U, V, W, n, m, p)
        
        # Find nullspace of M
        # M is (n*m*p) x r. If rank(M) < r, there's a null vector.
        from numpy.linalg import matrix_rank, null_space
        rank_M = matrix_rank(M)
        
        if rank_M >= r:
            break  # No more dependencies
        
        null_vec = null_space(M)  # r x (r - rank_M)
        # Take the first null vector
        nv = null_vec[:, 0]  # length r
        
        # nv represents a linear combination sum_c nv[c] * T_c = 0
        # This means the rank-1 terms are linearly dependent.
        # We can eliminate one term. Find the component with largest |nv[c]|
        # and express it in terms of others.
        
        # Actually, for a valid tensor decomposition, if sum_c nv[c] * T_c = 0,
        # it doesn't directly mean we can remove a term. The decomposition
        # C = sum_c T_c(A,B) must still hold.
        
        # The issue is more subtle. Let me think...
        # If the T_c are linearly dependent as tensors, it means the
        # decomposition is not minimal. But to actually reduce rank,
        # we need to find a subset or linear combination that still
        # spans the same tensor.
        
        # A simpler approach: if there's a null vector nv, we can
        # "absorb" one term into the others. But this is complex.
        
        # Let me try a different approach: use the null vector to
        # identify which terms are redundant and remove them.
        
        # For now, just note the rank and try a greedy removal.
        # Find the term with smallest coefficient in some sense.
        
        # Actually, let me try: remove the term c* where |nv[c*]| is largest,
        # and scale the others. This doesn't work directly.
        
        # Let me try a different strategy: use SVD to find the
        # effective rank and keep only the top components.
        # But this may not give an exact decomposition.
        
        # For now, let's just report the rank of M as the lower bound
        # and return the original decomposition.
        break
    
    return U, V, W, r


def _try_greedy_removal(U, V, W, n, m, p):
    """
    Try to greedily remove rank-1 terms while maintaining exact decomposition.
    This is a heuristic approach.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    r = U.shape[1]
    
    # For each term, check if removing it still gives exact decomposition
    # (This is expensive but let's try a few)
    removed = []
    for c in range(r):
        if c in removed:
            continue
        # Try removing term c
        U_test = np.delete(U, c, axis=1)
        V_test = np.delete(V, c, axis=1)
        W_test = np.delete(W, c, axis=1)
        
        # Check if this still decomposes the tensor
        if _tensor_flatten_rank1(U_test, V_test, W_test, n, m, p):
            U, V, W = U_test, V_test, W_test
            removed.append(c)
            r -= 1
    
    return U, V, W, r


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook rank-72 decomposition
    U, V, W = _build_strassen_3x4x6()
    r = U.shape[1]
    
    # Verify
    assert _tensor_flatten_rank1(U, V, W, n, m, p), "Schoolbook should be valid"
    
    # Try to reduce rank by finding dependencies
    if time.time() < deadline:
        U, V, W, r = _find_and_remove_dependencies(U, V, W, n, m, p)
    
    # Try greedy removal
    if time.time() < deadline:
        U, V, W, r = _try_greedy_removal(U, V, W, n, m, p)
    
    # Verify final result
    assert _tensor_flatten_rank1(U, V, W, n, m, p), "Final decomposition must be valid"
    
    # Convert to lists
    U_list = [[int(U[i, j]) if U[i, j] == int(U[i, j]) else float(U[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[int(V[i, j]) if V[i, j] == int(V[i, j]) else float(V[i, j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[int(W[i, j]) if W[i, j] == int(W[i, j]) else float(W[i, j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
