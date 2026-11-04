import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def strassen_factors():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    Convention: U[i*2+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k].
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # m2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def schoolbook_factors(n, m, p):
    """
    Returns (U, V, W) for n x m x p schoolbook decomposition.
    """
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

    return U, V, W

def build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes):
    """
    Build the full factor matrices using block decomposition.
    """
    total_rank = 0
    n_offsets = []
    offset = 0
    for s in n_sizes:
        n_offsets.append(offset)
        offset += s

    m_offsets = []
    offset = 0
    for s in m_sizes:
        m_offsets.append(offset)
        offset += s

    p_offsets = []
    offset = 0
    for s in p_sizes:
        p_offsets.append(offset)
        offset += s

    for i_idx, ni in enumerate(n_sizes):
        for j_idx, mj in enumerate(m_sizes):
            for k_idx, pk in enumerate(p_sizes):
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk

    U_full = np.zeros((n * m, total_rank), dtype=np.float64)
    V_full = np.zeros((m * p, total_rank), dtype=np.float64)
    W_full = np.zeros((p * n, total_rank), dtype=np.float64)

    col_offset = 0
    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]

                if ni == 2 and mj == 2 and pk == 2:
                    U_block, V_block, W_block = strassen_factors()
                else:
                    U_block, V_block, W_block = schoolbook_factors(ni, mj, pk)
                r = U_block.shape[1]

                for li in range(ni):
                    for lj in range(mj):
                        full_row = (i_start + li) * m + (j_start + lj)
                        U_full[full_row, col_offset:col_offset + r] = U_block[li * mj + lj, :]

                for lj in range(mj):
                    for lk in range(pk):
                        full_row = (j_start + lj) * p + (k_start + lk)
                        V_full[full_row, col_offset:col_offset + r] = V_block[lj * pk + lk, :]

                for lk in range(pk):
                    for li in range(ni):
                        full_row = (k_start + lk) * n + (i_start + li)
                        W_full[full_row, col_offset:col_offset + r] = W_block[lk * ni + li, :]

                col_offset += r

    return U_full, V_full, W_full, total_rank

def verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    """
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx

    products = U[u_rows, :] * V[v_rows, :] * W[w_rows, :]
    vals = products.sum(axis=1)

    return bool(np.allclose(vals, 1.0, atol=atol))

def try_reduce_rank(U, V, W, n, m, p, max_reduction=5, time_limit=100.0):
    """
    Attempt to reduce the rank by finding linear dependencies.
    For each column, check if removing it and solving for corrections
    to other columns still reproduces the tensor.
    """
    start_time = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    current_rank = U.shape[1]
    
    # Precompute index arrays for tensor evaluation
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    
    # Number of tensor entries
    N = n * m * p
    
    # Try to remove columns one by one
    removed = []
    for _ in range(max_reduction):
        if time.time() - start_time > time_limit - 10:
            break
        
        found_removal = False
        for c in range(U.shape[1]):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Check if column c can be removed
            # We need to find adjustments to other columns such that:
            # sum_{c' != c} U[u,c'] * V[v,c'] * W[w,c'] * delta_{c'} = U[u,c]*V[v,c]*W[w,c]
            # This is a bilinear equation, hard to solve directly.
            # 
            # Alternative: check if the "slice" for column c is in the span
            # of other columns' contributions.
            
            # Compute the contribution of column c
            target = U[u_rows, c] * V[v_rows, c] * W[w_rows, c]
            
            # For each other column, compute its "direction" in tensor space
            # and check if target is in their span
            # This is expensive, so use a heuristic:
            # Check if target is nearly zero (column c contributes nothing)
            if np.max(np.abs(target)) < 1e-12:
                # Column c is already zero, remove it
                U = np.delete(U, c, axis=1)
                V = np.delete(V, c, axis=1)
                W = np.delete(W, c, axis=1)
                removed.append(c)
                found_removal = True
                break
            
            # Check if column c's contribution can be absorbed by others
            # Build the matrix of other columns' contributions
            other_cols = [cc for cc in range(U.shape[1]) if cc != c]
            if len(other_cols) < 2:
                continue
            
            # For each tensor entry, the contribution from column c is:
            # U[u,c] * V[v,c] * W[w,c]
            # We want to see if this can be written as sum over other columns
            # of alpha_{c'} * U[u,c'] * V[v,c'] * W[w,c']
            # This is a rank-1 update problem, which is generally hard.
            # 
            # Heuristic: if U[:,c], V[:,c], W[:,c] are all in the column spans
            # of the other matrices, then c might be removable.
            # Check if U[:,c] is in col(U[:, other_cols])
            try:
                U_other = U[:, other_cols]
                V_other = V[:, other_cols]
                W_other = W[:, other_cols]
                
                # Check rank
                if np.linalg.matrix_rank(np.hstack([U_other, U[:, [c]]])) <= np.linalg.matrix_rank(U_other):
                    if np.linalg.matrix_rank(np.hstack([V_other, V[:, [c]]])) <= np.linalg.matrix_rank(V_other):
                        if np.linalg.matrix_rank(np.hstack([W_other, W[:, [c]]])) <= np.linalg.matrix_rank(W_other):
                            # All three vectors are in the spans, try removal
                            U_new = np.delete(U, c, axis=1)
                            V_new = np.delete(V, c, axis=1)
                            W_new = np.delete(W, c, axis=1)
                            if verify_decomposition(U_new, V_new, W_new, n, m, p, atol=1e-9):
                                U, V, W = U_new, V_new, W_new
                                removed.append(c)
                                found_removal = True
                                break
            except:
                continue
        
        if not found_removal:
            break
    
    return U, V, W, U.shape[1]

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    deadline = time.time() + 140.0  # safety margin before 150s wall-clock

    # Try multiple block partitions and pick the best (lowest rank)
    # Generate candidate partitions
    def gen_partitions(total, max_part=4):
        """Generate all compositions of total into parts of size <= max_part."""
        if total == 0:
            yield []
            return
        for first in range(1, min(total, max_part) + 1):
            for rest in gen_partitions(total - first, max_part):
                yield [first] + rest
    
    best_U, best_V, best_W, best_rank = None, None, None, float('inf')
    
    # Candidate partitions for n=3, m=5, p=6
    n_parts = list(gen_partitions(3, max_part=4))
    m_parts = list(gen_partitions(5, max_part=4))
    p_parts = list(gen_partitions(6, max_part=4))
    
    # Filter to reasonable candidates (not too many parts)
    n_parts = [p for p in n_parts if len(p) <= 3]
    m_parts = [p for p in m_parts if len(p) <= 4]
    p_parts = [p for p in p_parts if len(p) <= 4]
    
    # Sort by expected rank (fewer 2x2x2 blocks = higher rank, but we want to try all)
    # Just iterate through all combinations
    for n_sizes in n_parts:
        if time.time() > deadline - 30:
            break
        for m_sizes in m_parts:
            if time.time() > deadline - 30:
                break
            for p_sizes in p_parts:
                if time.time() > deadline - 30:
                    break
                
                # Compute expected rank
                expected_rank = 0
                for ni, mj, pk in zip(n_sizes, m_sizes, p_sizes):
                    if ni == 2 and mj == 2 and pk == 2:
                        expected_rank += 7
                    else:
                        expected_rank += ni * mj * pk
                
                if expected_rank >= best_rank:
                    continue
                
                try:
                    U, V, W, total_rank = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)
                    if verify_decomposition(U, V, W, n, m, p, atol=1e-9):
                        best_U, best_V, best_W, best_rank = U, V, W, total_rank
                except:
                    continue
    
    if best_U is None:
        # Fallback: schoolbook
        U_fb, V_fb, W_fb = schoolbook_factors(n, m, p)
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Attempt rank reduction if time permits
    if time.time() < deadline - 20:
        U_red, V_red, W_red, new_rank = try_reduce_rank(best_U, best_V, best_W, n, m, p, 
                                                         max_reduction=5,
                                                         time_limit=min(100.0, deadline - time.time() - 5))
        if new_rank < best_rank and verify_decomposition(U_red, V_red, W_red, n, m, p):
            return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
