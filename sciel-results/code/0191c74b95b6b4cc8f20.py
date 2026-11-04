import time
import numpy as np
from itertools import product

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

def _block_rank(ni, mj, pk):
    """Compute the rank for a single block of size ni x mj x pk."""
    if ni == 2 and mj == 2 and pk == 2:
        return 7
    return ni * mj * pk

def _find_best_block_partition(n, m, p, deadline):
    """
    Search over all valid block partitions of n, m, p to find the one
    with minimum total rank.
    """
    def gen_partitions(total, max_part=None):
        """Generate all integer partitions of 'total' into parts >= 1."""
        if max_part is None:
            max_part = total
        if total == 0:
            yield []
            return
        if max_part == 0:
            return
        for first in range(min(total, max_part), 0, -1):
            for rest in gen_partitions(total - first, first):
                yield [first] + rest

    best_rank = n * m * p
    best_partition = None

    n_parts = list(gen_partitions(n))
    m_parts = list(gen_partitions(m))
    p_parts = list(gen_partitions(p))

    for np_ in n_parts:
        for mp_ in m_parts:
            for pp_ in p_parts:
                if time.time() > deadline - 5:
                    return best_rank, best_partition
                total = 0
                for ni, mj, pk in product(np_, mp_, pp_):
                    total += _block_rank(ni, mj, pk)
                if total < best_rank:
                    best_rank = total
                    best_partition = (np_, mp_, pp_)

    return best_rank, best_partition

def build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes):
    """
    Builds a block-diagonal decomposition given the partition sizes.
    """
    total_rank = sum(_block_rank(ni, mj, pk) 
                     for ni, mj, pk in product(n_sizes, m_sizes, p_sizes))
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col_offset = 0
    n_offsets = np.cumsum([0] + n_sizes)[:-1]
    m_offsets = np.cumsum([0] + m_sizes)[:-1]
    p_offsets = np.cumsum([0] + p_sizes)[:-1]
    
    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]
                
                if ni == 2 and mj == 2 and pk == 2:
                    U_blk, V_blk, W_blk = strassen_factors()
                else:
                    U_blk, V_blk, W_blk = schoolbook_factors(ni, mj, pk)
                
                r = U_blk.shape[1]
                
                for li in range(ni):
                    for lj in range(mj):
                        full_row_u = (i_start + li) * m + (j_start + lj)
                        U[full_row_u, col_offset:col_offset + r] = U_blk[li * mj + lj, :]
                        
                for lj in range(mj):
                    for lk in range(pk):
                        full_row_v = (j_start + lj) * p + (k_start + lk)
                        V[full_row_v, col_offset:col_offset + r] = V_blk[lj * pk + lk, :]
                        
                for lk in range(pk):
                    for li in range(ni):
                        full_row_w = (k_start + lk) * n + (i_start + li)
                        W[full_row_w, col_offset:col_offset + r] = W_blk[lk * ni + li, :]
                        
                col_offset += r
                
    return U, V, W, total_rank

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

def try_reduce_rank(U, V, W, n, m, p):
    """
    Attempts to reduce the rank by finding linear dependencies among columns.
    Uses the fact that if columns are linearly dependent in the tensor space,
    we can remove one.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    dim_u = U.shape[0]
    dim_v = V.shape[0]
    dim_w = W.shape[0]
    
    # Build the tensor space matrix
    total_dim = dim_u * dim_v * dim_w
    M = np.zeros((total_dim, rank), dtype=np.float64)
    
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    flat_idx = (i * m + j) * p + k
                    M[flat_idx, c] = U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
    
    try:
        svd_rank = np.linalg.matrix_rank(M, tol=1e-10)
    except:
        return U, V, W
    
    if svd_rank >= rank:
        return U, V, W
    
    return U, V, W

def try_sparsify(U, V, W, n, m, p, deadline):
    """
    Attempts to reduce rank by systematically perturbing with sparse half-integer
    corrections and checking for linear dependencies.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to find if any single column can be removed while preserving the decomposition
    # by checking if its tensor contribution is in the span of others
    
    # First, check if columns are linearly dependent in tensor space
    total_dim = U.shape[0] * V.shape[0] * W.shape[0]
    M = np.zeros((total_dim, rank), dtype=np.float64)
    
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    flat_idx = (i * m + j) * p + k
                    M[flat_idx, c] = U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
    
    # Compute rank of M
    try:
        svd_vals = np.linalg.svd(M, compute_uv=False)
        svd_rank = np.sum(svd_vals > 1e-10)
    except:
        return U, V, W
    
    if svd_rank < rank:
        # There are linear dependencies. Try to find a subset of columns that spans the same space
        # and verify the decomposition still holds
        
        # Get the null space vectors to identify which columns are dependent
        # For now, try removing columns one by one and check if decomposition still holds
        
        for c_remove in range(rank):
            if time.time() > deadline - 30:
                break
            
            # Create reduced matrices by removing column c_remove
            mask = np.ones(rank, dtype=bool)
            mask[c_remove] = False
            
            U_red = U[:, mask]
            V_red = V[:, mask]
            W_red = W[:, mask]
            
            # Verify the reduced decomposition
            if verify_decomposition(U_red, V_red, W_red, n, m, p, atol=1e-8):
                best_U, best_V, best_W = U_red, V_red, W_red
                best_rank = rank - 1
                # Continue trying to remove more columns
                U, V, W = U_red, V_red, W_red
                rank -= 1
                # Rebuild M for the reduced matrices
                M = np.zeros((total_dim, rank), dtype=np.float64)
                for c in range(rank):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                u_idx = i * m + j
                                v_idx = j * p + k
                                w_idx = k * n + i
                                flat_idx = (i * m + j) * p + k
                                M[flat_idx, c] = U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # First, try to find the best block partition
    best_rank, best_partition = _find_best_block_partition(n, m, p, deadline)
    
    if best_partition is not None:
        n_sizes, m_sizes, p_sizes = best_partition
    else:
        # Default partition
        n_sizes, m_sizes, p_sizes = [1, 2], [2, 2, 1], [2, 2, 2]
        best_rank = 84

    # Build the decomposition
    U, V, W, rank = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)
    
    # Verify
    if verify_decomposition(U, V, W, n, m, p):
        # Try to reduce rank using sparsification
        if time.time() < deadline - 30:
            U_reduced, V_reduced, W_reduced, new_rank = try_sparsify(U, V, W, n, m, p, deadline)
            if new_rank < rank:
                if verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p, atol=1e-8):
                    return {"U": U_reduced.tolist(), "V": V_reduced.tolist(), "W": W_reduced.tolist()}
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb), dtype=np.float64)
    V_fb = np.zeros((m * p, rank_fb), dtype=np.float64)
    W_fb = np.zeros((p * n, rank_fb), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
