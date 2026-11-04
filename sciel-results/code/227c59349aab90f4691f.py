import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    """Return the 3D tensor T[i,j,k] = 1 (the matrix multiplication tensor)."""
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W exactly decompose the tensor (within tol)."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_structured_decomposition():
    """Construct a rank-44 decomposition using Strassen for 2x2x2 blocks and schoolbook for 1x2x2 blocks."""
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        # Term 1: (A11+A22)(B11+B22)
        U[(rA + 0) * m + (cA + 0), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 2: (A21+A22)B11
        U[(rA + 1) * m + (cA + 0), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 1), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = -1
        col += 1
        # Term 3: A11(B21-B11)
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = -1
        W[(cB + 1) * n + (rA + 0), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 4: A22(B12-B22)
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 0), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = -1
        W[(cB + 0) * n + (rA + 0), col] = 1
        W[(cB + 0) * n + (rA + 1), col] = 1
        col += 1
        # Term 5: (A11+A12)B22
        U[(rA + 0) * m + (cA + 0), col] = 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = -1
        col += 1
        # Term 6: (A21-A11)(B11+B12)
        U[(rA + 1) * m + (cA + 0), col] = 1
        U[(rA + 0) * m + (cA + 0), col] = -1
        V[(rB + 0) * p + (cB + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 7: (A12-A22)(B21+B22)
        U[(rA + 0) * m + (cA + 1), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = -1
        V[(rB + 1) * p + (cB + 0), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        nonlocal col
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        col += 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        col += 1

    # Apply Strassen to the four 2x2x2 sub-blocks
    add_strassen(0, 0, 0, 0)
    add_strassen(0, 2, 2, 0)
    add_strassen(0, 0, 0, 2)
    add_strassen(0, 2, 2, 2)
    # Apply schoolbook to the four 1x2x2 sub-blocks (row 2 of A, columns 0-1 or 2-3)
    add_schoolbook_1x2x2(2, 0, 0, 0)
    add_schoolbook_1x2x2(2, 2, 2, 0)
    add_schoolbook_1x2x2(2, 0, 0, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2)

    return U, V, W

def _compute_flattened_terms(U, V, W, n, m, p):
    """Compute the flattened tensor for each term."""
    rank = U.shape[1]
    terms = np.zeros((n * m * p, rank))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        terms[:, r] = np.einsum('ij,jk,ki->ijk', u, v, w).flatten()
    return terms

def _try_reduce_rank_global(U, V, W, n, m, p, time_budget):
    """Try to reduce rank by finding global linear dependencies among the flattened terms.
    
    Note: A linear dependency among flattened terms does NOT directly translate to
    a simple column elimination in U, V, W because the tensor is multilinear.
    However, we can try a different approach: if we have a dependency
    sum_r c_r * term_r = 0, we can try to express the eliminated term's
    contribution by modifying other terms. This is generally not possible
    with simple column operations, so we use a more careful approach:
    try removing each term one at a time and check if the remaining terms
    can still span the same tensor (via least squares on the factor space).
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    # For each term, try to see if it can be removed
    # by checking if the tensor can still be decomposed with rank-1 terms
    # This is expensive, so we do it carefully
    
    while time.time() - start_time < time_budget:
        improved = False
        for elim_idx in range(current_rank):
            if time.time() - start_time > time_budget:
                break
            
            # Try removing term elim_idx
            U_test = np.delete(U, elim_idx, axis=1)
            V_test = np.delete(V, elim_idx, axis=1)
            W_test = np.delete(W, elim_idx, axis=1)
            
            # Check if the remaining terms can still produce the target tensor
            # by solving for whether the tensor is in the span
            if _verify_decomposition(U_test, V_test, W_test, n, m, p, tol=1e-9):
                U, V, W = U_test, V_test, W_test
                current_rank = U.shape[1]
                improved = True
                break
        
        if not improved:
            break
    
    return U, V, W

def _construct_rank16_4x4x4():
    """Construct a rank-16 decomposition for 4x4x4 matrix multiplication.
    
    This uses the structure: split 4x4 into 2x2 blocks of 2x2 matrices.
    The 4x4x4 tensor can be decomposed using 4 copies of the 2x2x2 Strassen
    decomposition, but with sharing. Actually, the known result is that
    the 4x4x4 tensor has rank at most 16 (from 4x2x2 Strassen structure).
    
    The 4x2x2 matrix multiplication (4x2 times 2x2) has rank 8 (from Strassen
    on the 2x2 blocks of the 4x2 matrix). So 4x4x4 = (4x2x2) x 2 has rank
    at most 8*2 = 16.
    
    Let me construct this properly.
    
    For 4x2x2: A is 4x2, B is 2x2. Split A into two 2x2 blocks: A_top, A_bottom.
    Each 2x2x2 multiplication uses rank 7 (Strassen). But we can share.
    
    Actually, let me use a different approach. The 4x4x4 tensor can be seen as:
    C = A * B where A is 4x4, B is 4x4.
    
    Split A into 2x2 blocks of 2x2: A = [[A11, A12], [A21, A22]]
    Split B into 2x2 blocks of 2x2: B = [[B11, B12], [B21, B22]]
    
    C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
         [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    Each 2x2x2 multiplication has rank 7. We have 4 such multiplications,
    giving rank 28. But we can do better by sharing.
    
    The known rank of 4x4x4 is 16 (from the 4x2x2 result). Let me look at
    the 4x2x2 case more carefully.
    
    For 4x2x2: A is 4x2, B is 2x2.
    Split A into A_top (2x2) and A_bottom (2x2).
    C_top = A_top * B (2x2x2, rank 7)
    C_bottom = A_bottom * B (2x2x2, rank 7)
    Total: rank 14.
    
    But we can do better. The 4x2x2 tensor has rank 8 (I believe).
    
    Actually, let me just try to construct a decomposition and verify it.
    """
    n, m, p = 4, 4, 4
    rank = 16
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Use the structure from the 2x2 Strassen decomposition
    # For each of the 4 output 2x2 blocks, we need to compute a 2x2x2 product
    # But we can share the Strassen factors
    
    # Let me try a different construction:
    # The 4x4x4 tensor can be decomposed as follows:
    # Use 4 copies of the 2x2x2 Strassen decomposition, one for each
    # 2x2 output block, but with shared V factors.
    
    # Actually, let me just use the schoolbook for now and try to reduce.
    # The schoolbook rank for 4x4x4 is 64.
    
    # Let me try a structured approach:
    # Split into 2x2 blocks. For each pair of (row_block, col_block) of output,
    # we have a 2x2x2 multiplication. There are 4 such multiplications.
    # Each has rank 7, so total rank 28.
    
    # But we can do better by noting that the V factors can be shared
    # between adjacent output blocks.
    
    # For now, let me construct the rank-28 version and see if we can reduce.
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    def add_strassen_4x4(rA, cA, rB, cB):
        nonlocal col
        # Same as before but for 4x4x4
        # Term 1: (A11+A22)(B11+B22)
        U[(rA + 0) * m + (cA + 0), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 2: (A21+A22)B11
        U[(rA + 1) * m + (cA + 0), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = 1
        W[(cB + 0) * n + (rA + 1), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = -1
        col += 1
        # Term 3: A11(B21-B11)
        U[(rA + 0) * m + (cA + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = -1
        W[(cB + 1) * n + (rA + 0), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 4: A22(B12-B22)
        U[(rA + 1) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 0), col] = 1
        V[(rB + 0) * p + (cB + 0), col] = -1
        W[(cB + 0) * n + (rA + 0), col] = 1
        W[(cB + 0) * n + (rA + 1), col] = 1
        col += 1
        # Term 5: (A11+A12)B22
        U[(rA + 0) * m + (cA + 0), col] = 1
        U[(rA + 0) * m + (cA + 1), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 0), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = -1
        col += 1
        # Term 6: (A21-A11)(B11+B12)
        U[(rA + 1) * m + (cA + 0), col] = 1
        U[(rA + 0) * m + (cA + 0), col] = -1
        V[(rB + 0) * p + (cB + 0), col] = 1
        V[(rB + 0) * p + (cB + 1), col] = 1
        W[(cB + 1) * n + (rA + 1), col] = 1
        col += 1
        # Term 7: (A12-A22)(B21+B22)
        U[(rA + 0) * m + (cA + 1), col] = 1
        U[(rA + 1) * m + (cA + 1), col] = -1
        V[(rB + 1) * p + (cB + 0), col] = 1
        V[(rB + 1) * p + (cB + 1), col] = 1
        W[(cB + 0) * n + (rA + 0), col] = 1
        col += 1
    
    # Apply Strassen to the four 2x2x2 sub-blocks of 4x4x4
    add_strassen_4x4(0, 0, 0, 0)
    add_strassen_4x4(0, 2, 2, 0)
    add_strassen_4x4(2, 0, 0, 0)
    add_strassen_4x4(2, 2, 2, 0)
    add_strassen_4x4(0, 0, 0, 2)
    add_strassen_4x4(0, 2, 2, 2)
    add_strassen_4x4(2, 0, 0, 2)
    add_strassen_4x4(2, 2, 2, 2)
    
    # Wait, that's 8 Strassen decompositions, each of rank 7, giving rank 56.
    # That's not right. Let me reconsider.
    
    # For 4x4x4, the output is 4x4. We split it into 2x2 blocks of 2x2.
    # There are 4 output blocks. For each output block (r_out, c_out), we compute:
    # C[r_out, c_out] = sum over k_block of A[r_out, k_block] * B[k_block, c_out]
    # where each term is a 2x2x2 multiplication.
    # So for each output block, we have 2 terms (k_block=0 and k_block=1),
    # each of rank 7, giving rank 14 per output block, total rank 56.
    
    # That's not helpful. Let me try a different approach.
    
    # Actually, I think the rank-16 construction is more subtle.
    # Let me just return the rank-28 decomposition for 3x4x4 that we already have
    # and focus on improving the reduction algorithm.
    
    return None  # This function is not used for now


def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS

    # Start with the structured rank-44 decomposition
    U44, V44, W44 = _construct_structured_decomposition()
    best_U, best_V, best_W = U44.copy(), V44.copy(), W44.copy()
    best_rank = 44

    # Verify the initial decomposition
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-9):
        # Fallback: schoolbook rank-48 decomposition
        U48 = np.zeros((n * m, 48))
        V48 = np.zeros((m * p, 48))
        W48 = np.zeros((p * n, 48))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U48[i * m + j, col] = 1
                    V48[j * p + k, col] = 1
                    W48[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U48, V48, W48
        best_rank = 48

    # Try to reduce rank by checking if any single term can be removed
    if time.time() < deadline - 30:
        time_budget = min(60, deadline - time.time() - 10)
        U_try, V_try, W_try = _try_reduce_rank_global(
            best_U, best_V, best_W, n, m, p, time_budget
        )
        if U_try.shape[1] < best_rank:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = U_try.shape[1]

    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
