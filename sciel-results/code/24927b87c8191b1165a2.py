"""Rank-42 exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Strategy: Block-partition into six 2x2 @ 2x2 multiplications, each computed
via Strassen's algorithm (rank 7). Total rank = 6 * 7 = 42.

Additionally attempts rank reduction via linear dependency analysis within
each 14-column block (two Strassen 2x2x2 products summed).
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]


def _strassen_factors():
    """Return (U1, V1, W1) for 2x2x2 Strassen, shapes (4,7),(4,7),(4,7)."""
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)

    # M1 = (A00+A11)(B00+B11)
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1

    # M2 = (A10+A11)B00
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1; W1[3, 1] = -1

    # M3 = A00(B01-B11)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1; W1[3, 2] = 1

    # M4 = A11(B10-B00)
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[1, 3] = 1

    # M5 = (A00+A01)B11
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1

    # M6 = (A10-A00)(B00+B01)
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1

    # M7 = (A01-A11)(B10+B11)
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1

    return U1, V1, W1


def _build_rank42():
    """Build rank-42 decomposition via 6 Strassen blocks."""
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()
    total_rank = 42

    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    for j_block in range(3):
        for side in range(2):
            b = j_block * 2 + side
            col_offset = b * 7

            for i in range(2):
                for j_local in range(2):
                    a = side * 2 + j_local
                    for s in range(7):
                        coeff = U1[i * 2 + j_local, s]
                        if coeff != 0:
                            U[i * m + a, col_offset + s] += coeff

            for j_local in range(2):
                a = side * 2 + j_local
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = V1[j_local * 2 + k_local, s]
                        if coeff != 0:
                            V[a * p + bcol, col_offset + s] += coeff

            for i in range(2):
                for k_local in range(2):
                    bcol = j_block * 2 + k_local
                    for s in range(7):
                        coeff = W1[k_local * 2 + i, s]
                        if coeff != 0:
                            W[bcol * n + i, col_offset + s] += coeff

    return U, V, W


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            acc = np.zeros(rank)
            for j in range(m):
                acc += U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
            if np.any(np.abs(acc - 1.0) > tol):
                return False
    return True


def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """Full verification for all (i,j,k) triples."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                          for s in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True


def _to_output(mat):
    """Convert numpy array to list of lists with int where possible."""
    return [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in mat]


def _build_schoolbook(n, m, p):
    """Build schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U[ii * m + jj][col] = 1
                V[jj * p + kk][col] = 1
                W[kk * n + ii][col] = 1
                col += 1
    return U, V, W


def _try_block_reduction(U, V, W, n, m, p, deadline=None):
    """
    Attempt to reduce rank by finding linear dependencies within each
    14-column block (two 7-column Strassen sub-blocks).
    
    For each block of 14 columns, we check if any column can be expressed
    as a linear combination of the others while preserving the tensor identity.
    
    The tensor identity for the block contributes constraints on the output
    entries. For a 2x2 output (4 entries per (i,k) pair, but here we have
    2 rows x 2 cols = 4 output entries per block), we have 4 constraints.
    
    Actually, the full constraint is: for each (i, k) in the output,
    sum over the block's columns of U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
    for all j. But since we're only modifying one block, the constraint is
    that the contribution from this block to each (i,k) output must remain unchanged.
    
    For a 14-column block affecting output positions (i, b0), (i, b0+1) for i=0,1:
    that's 4 output entries, each giving one constraint.
    
    To eliminate column s_elim from the block, we need to find coefficients
    c_1, ..., c_13 such that for each of the 4 output entries:
    sum_{s in block, s != s_elim} c_s * (U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s])
    = U[i*m+j,s_elim] * V[j*p+k,s_elim] * W[k*n+i,s_elim]
    
    This is a 4x13 system (underdetermined), so solutions exist generically.
    But we also need the solution to be "clean" (small rational numbers).
    
    A simpler approach: check if removing column s_elim and adjusting the
    remaining 13 columns via a rank-1 update works.
    
    Actually, let's try a different approach: for each pair of columns in the
    same 14-block, check if their difference can be absorbed.
    """
    if deadline is not None and time.time() > deadline - 10:
        return U, V, W

    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    current_rank = U_cur.shape[1]
    n, m, p = 2, 4, 6

    # The 6 blocks of 7 columns each (or 3 blocks of 14)
    # In our construction, blocks are indexed by (j_block, side) -> col_offset
    # Block (jb, side) has columns [jb*14 + side*7, jb*14 + side*7 + 6]
    
    # Try to reduce each 14-block (two consecutive 7-blocks for same j_block)
    for jb in range(3):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        block_start = jb * 14
        block_end = jb * 14 + 14
        
        # For this 14-column block, the affected output entries are:
        # C[i, jb*2] and C[i, jb*2+1] for i=0,1 -> 4 entries
        # Each entry (i, k) where k = jb*2 or jb*2+1
        # Constraint: for each (i,k), sum_j sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] = 1
        # The block's contribution to (i,k) is:
        #   sum_{j=0}^{3} sum_{s in block} U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
        
        # Try to eliminate each column in the block
        for s_elim in range(block_end - 1, block_start - 1, -1):
            if deadline is not None and time.time() > deadline - 3:
                    break
            
            # Build the 4x13 system
            # For each output (i, k) where i in {0,1}, k in {jb*2, jb*2+1}:
            # We need: sum_{s in block, s != s_elim} x_s * T_{i,k}(s) = T_{i,k}(s_elim)
            # where T_{i,k}(s) = sum_j U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
            
            # But wait - this is not quite right. The constraint is that the
            # TOTAL contribution equals 1, and other blocks contribute fixed amounts.
            # So the block's contribution must equal 1 - (other blocks' contribution).
            # If we change the block, we need the new block contribution to equal
            # the old block contribution.
            
            # So: sum_{s in block} T_{i,k}(s) = block_contrib_{i,k} (fixed)
            # After eliminating s_elim and adjusting others:
            # sum_{s in block, s!=s_elim} x_s * T_{i,k}(s) = block_contrib_{i,k}
            # where x_s are the new values (initially x_s = 1 for all s in block)
            
            # Equivalently: sum_{s in block, s!=s_elim} (x_s - 1) * T_{i,k}(s) = -T_{i,k}(s_elim)
            
            # Let d_s = x_s - 1. We need:
            # sum_{s in block, s!=s_elim} d_s * T_{i,k}(s) = -T_{i,k}(s_elim)
            
            # This is 4 equations in 13 unknowns. Use least squares or null space.
            
            # Build the 4x13 matrix
            remaining_cols = [s for s in range(block_start, block_end) if s != s_elim]
            A_mat = np.zeros((4, len(remaining_cols)))
            b_vec = np.zeros(4)
            
            out_idx = 0
            for i in range(2):
                for k_off in range(2):
                    k = jb * 2 + k_off
                    for ci, s in enumerate(remaining_cols):
                        # T_{i,k}(s) = sum_j U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
                        t_val = 0.0
                        for j in range(m):
                            t_val += U_cur[i * m + j, s] * V_cur[j * p + k, s] * W_cur[k * n + i, s]
                        A_mat[out_idx, ci] = t_val
                    # RHS: -T_{i,k}(s_elim)
                    t_elim = 0.0
                    for j in range(m):
                        t_elim += U_cur[i * m + j, s_elim] * V_cur[j * p + k, s_elim] * W_cur[k * n + i, s_elim]
                    b_vec[out_idx] = -t_elim
                    out_idx += 1
            
            # Check if the system is consistent and has a small solution
            # Use SVD to find minimum-norm solution
            try:
                U_svd, S, Vt = np.linalg.svd(A_mat, full_matrices=False)
                # Minimum norm solution: x = Vt^T @ S^{-1} @ U^T @ b
                # But we want a sparse/small solution. Let's check if b is in the column space.
                # Project b onto column space
                U_proj = U_svd[:, S > 1e-10]
                b_proj = U_proj.T @ b_vec
                residual = np.linalg.norm(b_vec - U_proj @ b_proj)
                
                if residual < 1e-8:
                    # Consistent. Find minimum norm solution.
                    S_inv = np.where(S > 1e-10, 1.0 / S, 0.0)
                    x_min = Vt.T @ (S_inv * (U_svd.T @ b_vec))
                    
                    # Check if solution is small (close to integers or simple fractions)
                    if np.max(np.abs(x_min)) < 10:
                        # Apply the adjustment
                        for ci, s in enumerate(remaining_cols):
                            if abs(x_min[ci]) > 1e-10:
                                U_cur[:, s] += x_min[ci] * U_cur[:, s_elim]
                                V_cur[:, s] += x_min[ci] * V_cur[:, s_elim]
                                W_cur[:, s] += x_min[ci] * W_cur[:, s_elim]
                        
                        # Remove the eliminated column
                        mask = np.ones(current_rank, dtype=bool)
                        mask[s_elim] = False
                        U_cur = U_cur[:, mask]
                        V_cur = V_cur[:, mask]
                        W_cur = W_cur[:, mask]
                        current_rank -= 1
                        
                        # Verify
                        if _verify_fast(U_cur, V_cur, W_cur, n, m, p):
                            break
                        else:
                            # Revert
                            U_cur = U.copy()
                            V_cur = V.copy()
                            W_cur = W.copy()
                            current_rank = U.shape[1]
            except Exception:
                continue
    
    return U_cur, V_cur, W_cur


def _try_cross_block_reduction(U, V, W, n, m, p, deadline=None):
    """
    Try to find cross-block dependencies: columns from different blocks
    that are linearly related in a way that preserves the tensor identity.
    
    This is more complex. For now, try a simple approach: check if any
    column is a linear combination of columns from other blocks.
    """
    if deadline is not None and time.time() > deadline - 5:
        return U, V, W

    # Skip for now - the within-block reduction is more likely to succeed
    return U, V, W


def _try_rank_reduction(U, V, W, n, m, p, max_reduce=5, deadline=None):
    """
    Try to reduce the rank by finding linear dependencies.
    First try within-block reduction, then cross-block.
    """
    if deadline is not None and time.time() > deadline - 10:
        return U, V, W

    # First: within-block reduction
    U1, V1, W1 = _try_block_reduction(U, V, W, n, m, p, deadline)
    
    if U1.shape[1] < U.shape[1]:
        # Verify and return
        if _verify_fast(U1, V1, W1, n, m, p):
            return U1, V1, W1
    
    # Second: try simple zero-column removal
    U_cur = U1.copy()
    V_cur = V1.copy()
    W_cur = W1.copy()
    current_rank = U_cur.shape[1]
    
    for attempt in range(max_reduce):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        eliminated = False
        for s_elim in range(current_rank - 1, -1, -1):
            if deadline is not None and time.time() > deadline - 3:
                break
            
            if (np.allclose(U_cur[:, s_elim], 0, atol=1e-12) and
                np.allclose(V_cur[:, s_elim], 0, atol=1e-12) and
                np.allclose(W_cur[:, s_elim], 0, atol=1e-12)):
                U_cur = np.delete(U_cur, s_elim, axis=1)
                V_cur = np.delete(V_cur, s_elim, axis=1)
                W_cur = np.delete(W_cur, s_elim, axis=1)
                current_rank -= 1
                eliminated = True
                break
        
        if not eliminated:
            break
    
    if _verify_fast(U_cur, V_cur, W_cur, n, m, p):
        return U_cur, V_cur, W_cur
    
    return U1, V1, W1


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Build rank-42 Strassen block decomposition
    U, V, W = _build_rank42()

    # Verify correctness
    if _verify_fast(U, V, W, n, m, p):
        # Try to reduce rank
        U2, V2, W2 = _try_rank_reduction(U, V, W, n, m, p, max_reduce=5, deadline=deadline)
        if U2.shape[1] < U.shape[1]:
            # Final verification
            if _verify_full(U2, V2, W2, n, m, p):
                return {"U": _to_output(U2), "V": _to_output(V2), "W": _to_output(W2)}
        return {"U": _to_output(U), "V": _to_output(V), "W": _to_output(W)}

    # Fallback: schoolbook rank 48
    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    return {"U": U_fb, "V": V_fb, "W": W_fb}


if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    rank = U.shape[1]
    print(f"Rank: {rank}")
    print(f"Verified: {_verify_fast(U, V, W, n, m, p)}")
