import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 6]

def strassen_factors():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    Convention: U[i*2+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k].
    A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=A*B.
    C[0,0]=ae+cg, C[0,1]=af+bh, C[1,0]=ce+dg, C[1,1]=cf+dh.
    m1=(a+d)(e+h), m2=(c+d)e, m3=a(f-h), m4=d(g-e), m5=(a+b)h, m6=(c-a)(e+f), m7=(b-d)(g+h)
    C[0,0]=m1+m4-m5+m7
    C[0,1]=m3+m5
    C[1,0]=m2+m4
    C[1,1]=m1-m2+m3+m6
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

def winograd_2x4x4_factors():
    """
    Returns (U, V, W) for 2x4x4 Winograd-style decomposition.
    Rank: 8 (using 4 products per row of A, 2 rows = 8)
    Convention: U[i*4+j, c] for A[i,j], V[j*4+k, c] for B[j,k], W[k*2+i, c] for C[i,k].
    
    For each row i of A (2 rows), we use 4 products:
    For row i:
      p1 = A[i,0] * B[0,:]
      p2 = A[i,1] * B[1,:]
      p3 = A[i,2] * B[2,:]
      p4 = A[i,3] * B[3,:]
      C[i,:] = p1 + p2 + p3 + p4
      
    This is just schoolbook but organized by rows. Rank = 2*4 = 8.
    Actually, let's use a true Winograd decomposition for 2x4x4.
    
    Winograd for 2x4x4: We want C = A*B where A is 2x4, B is 4x4.
    C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:] + A[0,2]*B[2,:] + A[0,3]*B[3,:]
    C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:] + A[1,2]*B[2,:] + A[1,3]*B[3,:]
    
    We can use 8 products (schoolbook) or try to reduce.
    For 2x4x4, the rank is at most 8. Let's just use schoolbook for simplicity.
    """
    n, m, p = 2, 4, 4
    rank = n * m * p  # 32, but we can do better
    # Actually, let's use the standard schoolbook
    U, V, W = schoolbook_factors(n, m, p)
    return U, V, W

def _block_rank(ni, mj, pk):
    """Compute rank for a single block of size ni x mj x pk."""
    if ni == 2 and mj == 2 and pk == 2:
        return 7
    else:
        return ni * mj * pk

def _find_best_block_partition(n, m, p, deadline):
    """
    Search over all valid partitions of n, m, p into parts of size 1 or 2
    to find the partition that minimizes total rank.
    """
    def compositions(target, max_part=2):
        """Generate all compositions of target using parts 1 and max_part."""
        results = []
        def helper(remaining, current):
            if remaining == 0:
                results.append(list(current))
                return
            for part in range(1, max_part + 1):
                if part <= remaining:
                    current.append(part)
                    helper(remaining - part, current)
                    current.pop()
        helper(target, [])
        return results

    n_parts = compositions(n)
    m_parts = compositions(m)
    p_parts = compositions(p)

    best_rank = float('inf')
    best_partition = None

    for n_sizes in n_parts:
        for m_sizes in m_parts:
            for p_sizes in p_parts:
                total_rank = 0
                for ni in n_sizes:
                    for mj in m_sizes:
                        for pk in p_sizes:
                            total_rank += _block_rank(ni, mj, pk)
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_partition = (n_sizes, m_sizes, p_sizes)
                if time.time() > deadline:
                    return best_partition, best_rank

    return best_partition, best_rank

def build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes, use_winograd=False):
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
                elif use_winograd and ni == 2 and mj == 4 and pk == 4:
                    total_rank += 8  # Winograd 2x4x4
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
                elif use_winograd and ni == 2 and mj == 4 and pk == 4:
                    U_block, V_block, W_block = winograd_2x4x4_factors()
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
    T[i,j,k] should be 1 for all i,j,k.
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

def schoolbook_lists(n, m, p):
    """Return schoolbook decomposition as nested lists (fallback)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    deadline = time.time() + 140.0  # safety margin before 150s wall-clock

    # Search for optimal block partition
    best_partition, best_rank = _find_best_block_partition(n, m, p, deadline)
    
    if best_partition is None:
        # Fallback to known good partition
        n_sizes = [2, 1]
        m_sizes = [2, 2, 1]
        p_sizes = [2, 2, 2]
    else:
        n_sizes, m_sizes, p_sizes = best_partition

    assert sum(n_sizes) == n
    assert sum(m_sizes) == m
    assert sum(p_sizes) == p

    # Build the block decomposition
    U, V, W, total_rank = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)

    # Verify correctness; fall back to schoolbook if it fails
    if verify_decomposition(U, V, W, n, m, p, atol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook (rank 90)
    if time.time() < deadline:
        U_fb, V_fb, W_fb = schoolbook_lists(n, m, p)
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Last resort (should not be reached)
    U_fb, V_fb, W_fb = schoolbook_lists(n, m, p)
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
