"""Combined program: block-structured (Strassen 2x2) + rank-reduction via SVD projection."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U_np = np.asarray(U, dtype=float)
    V_np = np.asarray(V, dtype=float)
    W_np = np.asarray(W, dtype=float)
    rank = U_np.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.einsum('c,c,c->', U_np[i*m+j, :], V_np[j*p+k, :], W_np[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _round_to_half_integer(arr, tol=1e-7):
    """Round each entry to the nearest half-integer. Return None if any entry is too far."""
    rounded = 0.5 * np.round(2.0 * arr)
    if np.max(np.abs(arr - rounded)) > tol:
        return None
    return rounded


def _get_strassen_2x2():
    """Return the 7 rank-1 terms for 2x2 matrix multiplication (Strassen).
    Each term is (u_vec, v_vec, w_vec) of length 4 each.
    Indexing: [0]=top-left, [1]=top-right, [2]=bottom-left, [3]=bottom-right
    """
    # Standard Strassen formulas:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1 - M5 + M4 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    factors = []
    # M1
    factors.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2
    factors.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    # M3
    factors.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
    # M4
    factors.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
    # M5
    factors.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    # M6
    factors.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # M7
    factors.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    return factors


def _build_block_decomposition(n, m, p, block_sizes, strassen_threshold=2):
    """
    Build a block-structured decomposition.
    For blocks where all three dimensions are >= strassen_threshold and even, use Strassen.
    Otherwise use schoolbook.
    Returns (U, V, W) as numpy arrays, or None if something goes wrong.
    """
    nb = len(block_sizes)
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    strassen_factors = _get_strassen_2x2()
    
    # Calculate total rank
    total_rank = 0
    for ib in range(nb):
        for jb in range(nb):
            for kb in range(nb):
                ni, mj, pk = block_sizes[ib], block_sizes[jb], block_sizes[kb]
                if ni >= strassen_threshold and mj >= strassen_threshold and pk >= strassen_threshold:
                    # For 2x2x2 blocks, use Strassen (rank 7)
                    if ni == 2 and mj == 2 and pk == 2:
                        total_rank += 7
                    else:
                        # For larger even blocks, use schoolbook for simplicity
                        total_rank += ni * mj * pk
                else:
                    total_rank += ni * mj * pk
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    for ib in range(nb):
        for jb in range(nb):
            for kb in range(nb):
                ni, mj, pk = block_sizes[ib], block_sizes[jb], block_sizes[kb]
                oi, oj, ok = offsets[ib], offsets[jb], offsets[kb]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen for 2x2x2
                    for (uf, vf, wf) in strassen_factors:
                        for idx in range(4):
                            if uf[idx] != 0:
                                U[(oi+idx//2)*m + (oj+idx%2), col] += uf[idx]
                            if vf[idx] != 0:
                                V[(oj+idx//2)*p + (ok+idx%2), col] += vf[idx]
                            if wf[idx] != 0:
                                W[(ok+idx//2)*n + (oi+idx%2), col] += wf[idx]
                        col += 1
                else:
                    # Schoolbook
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U[(oi+li)*m + (oj+lj), col] = 1.0
                                V[(oj+lj)*p + (ok+lk), col] = 1.0
                                W[(ok+lk)*n + (oi+li), col] = 1.0
                                col += 1
    
    if col != total_rank:
        raise ValueError(f"Column mismatch: {col} vs {total_rank}")
    
    return U, V, W


def _reduce_rank_via_svd(U, V, W, target_rank, tol=1e-9, max_iter=50):
    """
    Try to reduce the rank by finding a change of basis that zeros out some columns.
    Uses iterative SVD-based projection.
    Returns (U', V', W') with rank <= target_rank, or None if not successful.
    """
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return U, V, W
    
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    for iteration in range(max_iter):
        if U_cur.shape[1] <= target_rank:
            break
        
        # Build the tensor as a matrix for SVD
        # T_{(i,j,k), c} = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We want to find a direction in the rank space that can be eliminated.
        
        # Approach: For each pair of "slices", try to find a linear dependency.
        # Simpler approach: use the fact that if we can find a non-trivial vector
        # in the kernel of a certain linear map, we can reduce rank.
        
        # Let's try: for each k,i pair, the constraint is:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
        # This means: (U_row_i * V_row_k) dot W_row_ki = 1 for all j
        # where U_row_i is the 5xR submatrix of U for row i
        # and V_row_k is the 5xR submatrix of V for column k
        
        # To reduce rank, we look for a vector d in R^R such that:
        # There exist U', V', W' with the same tensor but with one fewer dimension.
        # This is equivalent to finding a rank-1 update that preserves the tensor.
        
        # Practical approach: try to find a column c such that we can express
        # U[:,c], V[:,c], W[:,c] as linear combinations of other columns
        # while preserving the tensor.
        
        # For now, use a simpler heuristic: try to find linear dependencies
        # in the U matrix and absorb them into V and W.
        
        # Check if U has linearly dependent columns
        U_rank = np.linalg.matrix_rank(U_cur, tol=tol)
        if U_rank < U_cur.shape[1]:
            # There's a dependency in U. Find it.
            # SVD of U
            _, s, Vt = np.linalg.svd(U_cur, full_matrices=False)
            # The last column of Vt corresponds to the null space
            null_vec = Vt[-1, :]
            
            # Check if we can use this to reduce rank
            # If U @ null_vec = 0, then for the tensor to be preserved,
            # we need V and W to also be modified.
            # This is complex; for now, just try to zero out the last column
            # by absorbing into others.
            
            # Try: for each row, if U[row, last] != 0, we can try to eliminate it
            # by modifying other columns. But this changes the tensor.
            
            # Alternative: just truncate and check if it still works
            pass
        
        # Simpler approach: try to find a column that can be removed
        # by adjusting others. For each column c, check if the tensor
        # without that column can be compensated.
        
        # For now, use a different strategy: try to find a linear combination
        # of columns that gives zero contribution to the tensor.
        
        # Build the "tensor matrix" for each (i,k) slice
        # A_{j,c} = U[i*m+j, c] * V[j*p+k, c]
        # The constraint is A @ W[k*n+i, :] = ones(m)
        
        # If we can find a vector d such that for all (i,k):
        # sum_c A_{j,c} * d_c * W[k*n+i, c] = 0 for all j
        # then we can modify U, V, W to eliminate one dimension.
        
        # This is getting complex. Let's use a simpler approach:
        # Try to find a column c such that U[:,c], V[:,c], W[:,c] are
        # "redundant" in the sense that removing them and adjusting others
        # still gives the correct tensor.
        
        # For now, just try random projections and check.
        
        # Generate a random projection matrix P of size (current_rank-1) x current_rank
        # and check if U@P, V@P, W@P still give the correct tensor.
        # This won't work directly because the tensor changes.
        
        # Better approach: use the "border rank" concept.
        # For now, implement a simple greedy column elimination.
        
        # For each column c from last to first:
        #   Try to express the contribution of column c as a combination of others.
        #   If possible, eliminate it.
        
        for c in range(U_cur.shape[1] - 1, 0, -1):
            if U_cur.shape[1] <= target_rank:
                break
            
            # Try to eliminate column c
            # The contribution of column c to the tensor is:
            # T_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            
            # We need to find modifications to other columns such that
            # the total tensor remains unchanged.
            
            # Simple approach: if U[:,c] is in the span of U[:,0:c],
            # then we can try to absorb it.
            
            # For now, just check if the column is "small" and try to zero it out.
            # This is a heuristic and may not work.
            
            # Alternative: try to find a 2x2 block where Strassen can be applied
            # to reduce rank further.
            
            # For now, break out of this loop and try a different strategy.
            break
        
        # Different strategy: try to find a better basis using SVD
        # of the "flattened" tensor.
        
        # Flatten the tensor to a matrix and do SVD
        # T_{(i,j,k), c} = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # This is a 125 x R matrix
        T_matrix = np.zeros((n*m*p, U_cur.shape[1]))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    T_matrix[idx, :] = U_cur[i*m+j, :] * V_cur[j*p+k, :] * W_cur[k*n+i, :]
        
        # SVD
        U_svd, s, Vt_svd = np.linalg.svd(T_matrix, full_matrices=False)
        
        # The rank of T_matrix should be n*m*p = 125 (since the tensor is full rank)
        # But the "border rank" might be lower.
        
        # For now, just try to project onto the first target_rank singular vectors
        if s.shape[0] > target_rank:
            # Project
            P = Vt_svd[:target_rank, :].T  # R x target_rank
            U_new = U_cur @ P
            V_new = V_cur @ P
            W_new = W_cur @ P
            
            # Check if this preserves the tensor
            if _verify(U_new, V_new, W_new, n, m, p, tol=1e-6):
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                continue
        
        # If we couldn't reduce further, break
        break
    
    if U_cur.shape[1] <= target_rank:
        # Round to half-integers
        U_rounded = _round_to_half_integer(U_cur)
        V_rounded = _round_to_half_integer(V_cur)
        W_rounded = _round_to_half_integer(W_cur)
        
        if U_rounded is not None and V_rounded is not None and W_rounded is not None:
            if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
                return U_rounded, V_rounded, W_rounded
    
    return None


def _try_cyclic_convolution_approach(n, m, p, deadline):
    """
    Try to model matrix multiplication as cyclic convolution over Z_5.
    A[i,j] * B[j,k] -> C[i,k]
    
    If we identify i,j,k with elements of Z_5, then:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    This is like a "twisted" convolution. Let's see if we can use FFT-like
    techniques over the group ring.
    
    For Z_5, the group ring Q[x]/(x^5-1) has 5 irreducible factors over Q:
    (x-1)(x^4+x^3+x^2+x+1)
    
    The DFT over Z_5 uses 5th roots of unity: 1, ω, ω^2, ω^3, ω^4
    where ω = exp(2πi/5).
    
    The key insight: if we can diagonalize the matrix multiplication tensor
    using the DFT, we might get a low-rank decomposition.
    
    However, the entries of the DFT involve algebraic numbers (5th roots of unity),
    which are not rational. So we can't directly use this for an exact decomposition
    with rational entries.
    
    But we can try to approximate or find a related structure.
    
    For now, this is a placeholder. The actual implementation would be complex.
    """
    # For now, just return None to indicate this approach wasn't implemented.
    # The block-structured approach with Strassen should be the main strategy.
    return None


def _try_random_improvement(U0, V0, W0, target_rank, deadline, max_attempts=100):
    """
    Try to improve upon the initial decomposition by random perturbations
    and rank reduction.
    """
    n, m, p = DIMENSIONS
    U0 = np.asarray(U0, dtype=float)
    V0 = np.asarray(V0, dtype=float)
    W0 = np.asarray(W0, dtype=float)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Randomly perturb U, V, W slightly
        scale = 0.1
        U = U0 + scale * np.random.randn(*U0.shape)
        V = V0 + scale * np.random.randn(*V0.shape)
        W = W0 + scale * np.random.randn(*W0.shape)
        
        # Try to reduce rank
        result = _reduce_rank_via_svd(U, V, W, target_rank, tol=1e-8, max_iter=10)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
    
    return None


def _build_initial_decomposition(n, m, p):
    """Build the initial block-structured decomposition."""
    # Try different block partitions
    # 5 = 2 + 2 + 1, 5 = 2 + 3, 5 = 1 + 1 + 1 + 1 + 1, etc.
    
    best_result = None
    best_rank = n * m * p
    
    block_partitions = [
        [2, 2, 1],
        [2, 3],
        [3, 2],
        [1, 1, 1, 1, 1],
        [5],
    ]
    
    for partition in block_partitions:
        if sum(partition) != n:
            continue
        
        try:
            U, V, W = _build_block_decomposition(n, m, p, partition)
            rank = U.shape[1]
            
            if rank < best_rank:
                if _verify(U, V, W, n, m, p):
                    best_rank = rank
                    best_result = (U, V, W)
        except Exception:
            continue
    
    if best_result is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        best_result = (U, V, W)
        best_rank = rank
    
    return best_result, best_rank


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 130.0  # safety margin below 150s kill
    
    # Step 1: Build initial block-structured decomposition
    (U0, V0, W0), initial_rank = _build_initial_decomposition(n, m, p)
    
    # Convert to list format for return
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = initial_rank
    
    # Step 2: Try to reduce rank via SVD-based methods
    if time.time() < deadline - 20:
        for target_rank in range(min(initial_rank - 1, 100), max(initial_rank - 20, 30), -1):
            if time.time() > deadline - 20:
                break
            
            result = _reduce_rank_via_svd(U0, V0, W0, target_rank, tol=1e-9, max_iter=20)
            if result is not None:
                U_r, V_r, W_r = result
                if _verify(U_r, V_r, W_r, n, m, p):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = U_r.shape[1]
                    break
    
    # Step 3: Try random improvements
    if time.time() < deadline - 10 and best_rank > 60:
        result = _try_random_improvement(best_U, best_V, best_W, best_rank - 1, deadline - 10, max_attempts=50)
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = U_r.shape[1]
    
    # Step 4: Try cyclic convolution approach (placeholder)
    if time.time() < deadline - 5:
        result = _try_cyclic_convolution_approach(n, m, p, deadline - 5)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r, n, m, p):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = U_r.shape[1]
    
    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
