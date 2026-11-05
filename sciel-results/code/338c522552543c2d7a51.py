import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _compute_rank(n, m, p, n_blocks, m_blocks, p_blocks):
    """Compute the rank without building matrices."""
    rank = 0
    for ns in n_blocks:
        for ms in m_blocks:
            for ps in p_blocks:
                if ns == 2 and ms == 2 and ps == 2:
                    rank += 7
                else:
                    rank += ns * ms * ps
    return rank

def _strassen_222_data():
    """
    Strassen 2x2x2 tensor decomposition.
    Returns list of 7 tuples, each with (a_factors, b_factors, c_factors).
    Each factor list is a list of (row, col, coeff).
    """
    return [
        # M1 = (A11+A22)(B11-B22) -> C11+=M1, C12-=M1, C21-=M1, C22+=M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,-1)], [(0,0,1),(0,1,-1),(1,0,-1),(1,1,1)]),
        # M2 = (A21+A22)B11 -> C21+=M2, C22+=M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,1)]),
        # M3 = A11(B12-B22) -> C12+=M3, C22+=M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4 = A22(B21-B11) -> C11+=M4, C21+=M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (A11+A12)B22 -> C11-=M5, C12+=M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (A21-A11)(B11+B12) -> C22+=M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (A12-A22)(B21+B22) -> C11+=M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

def _block_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
    """Generate a decomposition using block-wise Strassen."""
    n_offsets = [0]
    for s in n_blocks:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_blocks:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_blocks:
        p_offsets.append(p_offsets[-1] + s)
    
    rank = 0
    terms = []
    for ni in range(len(n_blocks)):
        for mi in range(len(m_blocks)):
            for pi in range(len(p_blocks)):
                ns = n_blocks[ni]
                ms = m_blocks[mi]
                ps = p_blocks[pi]
                if ns == 2 and ms == 2 and ps == 2:
                    rank += 7
                    terms.append((ni, mi, pi, 'strassen'))
                else:
                    rank += ns * ms * ps
                    terms.append((ni, mi, pi, 'trivial'))
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    strassen_data = _strassen_222_data()
    
    for ni, mi, pi, method in terms:
        n0 = n_offsets[ni]
        m0 = m_offsets[mi]
        p0 = p_offsets[pi]
        ns = n_blocks[ni]
        ms = m_blocks[mi]
        ps = p_blocks[pi]
        
        if method == 'strassen':
            for a_factors, b_factors, c_factors in strassen_data:
                for (ar, ac, acoeff) in a_factors:
                    U[(n0+ar)*m + (m0+ac), col] += acoeff
                for (br, bc, bcoeff) in b_factors:
                    V[(m0+br)*p + (p0+bc), col] += bcoeff
                for (cr, cc, ccoeff) in c_factors:
                    W[(p0+cc)*n + (n0+cr), col] += ccoeff
                col += 1
        else:
            for i in range(ns):
                for j in range(ms):
                    for k in range(ps):
                        U[(n0+i)*m + (m0+j), col] = 1
                        V[(m0+j)*p + (p0+k), col] = 1
                        W[(p0+k)*n + (n0+i), col] = 1
                        col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    return U, V, W

def _get_all_splits(dim):
    """Generate all compositions of dim into parts of size 1 and 2."""
    splits = []
    def _gen(remaining, current):
        if remaining == 0:
            splits.append(current[:])
            return
        if remaining >= 1:
            current.append(1)
            _gen(remaining - 1, current)
            current.pop()
        if remaining >= 2:
            current.append(2)
            _gen(remaining - 2, current)
            current.pop()
    _gen(dim, [])
    return splits

def _find_best_block_split(n, m, p):
    """Search over block splits to find the one with minimum rank."""
    n_splits = _get_all_splits(n)
    m_splits = _get_all_splits(m)
    p_splits = _get_all_splits(p)
    
    best_rank = n * m * p
    best_split = None
    
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                rank = _compute_rank(n, m, p, ns, ms, ps)
                if rank < best_rank:
                    best_rank = rank
                    best_split = (ns, ms, ps)
    
    return best_rank, best_split

def _try_svd_rank_reduction(U0, V0, W0, n, m, p, deadline):
    """
    Try to reduce rank using SVD-based approach.
    The matrix multiplication tensor can be viewed as a matrix M of shape (n*m, m*p) 
    where M[i,j] = sum over k of ... No, that's not right.
    
    Actually, the tensor T has indices (i,j,k) where i in [0,n), j in [0,m), k in [0,p).
    We want T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j,k is diagonal}
    
    Wait, the tensor for matrix multiplication C = AB where C[i,k] = sum_j A[i,j]B[j,k].
    The tensor is T[i,j,k] = delta_{i,k} when... no.
    
    The bilinear form: C[i,k] = sum_{j} A[i,j] B[j,k].
    So the "target" tensor has T[i,j,k] = 1 if the (i,k) entry depends on A[i,j]*B[j,k].
    Actually T[i,j,k] = 1 for all i,j,k (since every A[i,j]B[j,k] contributes to C[i,k]).
    
    Wait, re-reading the task:
    "exactly decompose the matrix multiplication tensor"
    "The output factor follows the official trace convention W[k*n+i,r]"
    
    The tensor is T[i,j,k] such that:
    sum_j A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k] T[i,j,k]
    
    Actually, the standard tensor decomposition for matrix multiplication:
    (C = AB) means C[i,k] = sum_j A[i,j] B[j,k]
    
    So the tensor T has T[i,j,k] = 1 for all valid i,j,k.
    And we want: T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    Wait, that would mean sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k.
    
    Let me check the verification:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    
    Yes! T[i,j,k] = 1 for all i,j,k.
    
    So the "matrix" to SVD is... well, this is a 3-tensor, not a matrix.
    But we can flatten it: T_flat[i*m+j, j*p+k, k*n+i] doesn't make sense directly.
    
    Actually, for rank minimization, we can use the fact that the tensor is:
    T[i,j,k] = 1
    
    This is a rank-1 tensor in the sense that it's the all-ones tensor.
    Wait, no. T[i,j,k] = 1 for all i,j,k means T = 1 * 1 * 1 where each is an all-ones vector.
    So the rank is 1? No, that's not right either.
    
    The issue is that the three factors have different index mappings:
    U is indexed by (i,j) -> i*m+j
    V is indexed by (j,k) -> j*p+k
    W is indexed by (k,i) -> k*n+i
    
    So T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    
    This is equivalent to: for each (i,j,k), the product of three values must sum to 1.
    
    The trivial decomposition uses rank = n*m*p, with U[i*m+j, r] = 1 when r corresponds to (i,j,k), V[j*p+k, r] = 1 when r corresponds to (i,j,k), W[k*n+i, r] = 1 when r corresponds to (i,j,k).
    
    For SVD-based approach:
    We can think of this as finding the minimal rank R such that there exist matrices U, V, W satisfying the constraints.
    
    One approach: fix U and V, then W is determined by:
    W[k*n+i, r] = (1 - sum_{r'<r} U[i*m+j, r'] V[j*p+k, r'] W[k*n+i, r']) / (U[i*m+j, r] V[j*p+k, r])
    
    This is complicated. Let me think of a simpler approach.
    
    Actually, for the all-ones tensor with these specific index mappings, I don't think SVD directly helps.
    The Strassen-based block decomposition is already quite good.
    
    Let me try a different approach: use the fact that for small dimensions, we can try to find a decomposition with rank < 99.
    
    Actually, let me just try to optimize the existing decomposition further.
    The current best is 99. Let me see if we can do better.
    
    For 3x5x7:
    Splits for 3: [3], [1,2], [2,1], [1,1,1]
    Splits for 5: [5], [1,4], [2,3], [3,2], [4,1], [1,1,3], [1,3,1], [3,1,1], [1,2,2], [2,1,2], [2,2,1], [1,1,1,2], [1,1,2,1], [1,2,1,1], [2,1,1,1], [1,1,1,1,1]
    Splits for 7: similar
    
    The rank is sum over blocks of (7 if 2x2x2 else ns*ms*ps).
    
    To minimize rank, we want as many 2x2x2 blocks as possible (cost 7 instead of 8).
    
    For n=3: max 1 block of size 2 (remaining 1)
    For m=5: max 2 blocks of size 2 (remaining 1)
    For p=7: max 3 blocks of size 2 (remaining 1)
    
    So we can have at most 1*2*3 = 6 blocks that are 2x2x2.
    But the blocks are aligned, so we need the 2-blocks in n, m, p to overlap.
    
    Best case: n=[2,1], m=[2,2,1], p=[2,2,2,1]
    Blocks:
    (2,2,2): 7, (2,2,2): 7, (2,2,2): 7, (2,2,1): 4, (2,1,2): 4, (2,1,2): 4, (2,1,1): 2
    (1,2,2): 4, (1,2,2): 4, (1,2,2): 4, (1,2,1): 2, (1,1,2): 2, (1,1,2): 2, (1,1,1): 1
    
    Wait, I need to be more careful. The blocks are a Cartesian product.
    
    Let me just trust the search.
    
    For the SVD approach, I'll try a heuristic:
    1. Start with the best block decomposition (rank 99).
    2. Try to reduce rank by 1 at a time by finding linear dependencies.
    
    Actually, this is getting too complex. Let me just return the best block decomposition.
    """
    # This function is a placeholder for future work.
    # For now, we'll just return None to indicate no improvement.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # First, get the trivial decomposition as fallback
    U_trivial, V_trivial, W_trivial = _block_decomposition(n, m, p, [n], [m], [p])
    
    # Search for better block splits
    best_rank, best_split = _find_best_block_split(n, m, p)
    
    if time.time() > deadline:
        return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
    
    if best_split is not None and best_rank < n*m*p:
        U, V, W = _block_decomposition(n, m, p, best_split[0], best_split[1], best_split[2])
    else:
        U, V, W = U_trivial, V_trivial, W_trivial
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

