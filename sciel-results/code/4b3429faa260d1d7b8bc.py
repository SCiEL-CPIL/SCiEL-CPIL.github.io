import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 7]

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=1e-9)

def _hankel_flatten(n, m, p):
    """
    Flatten the multiplication tensor T[i,j,k] = delta[i*k] into a Hankel-like matrix.
    The tensor has indices (i in [0,n), j in [0,m), k in [0,p)).
    We flatten to a matrix where rows are (i,k) pairs and columns are j.
    T[i,j,k] = 1 if i==k else 0.
    
    Actually, let's think more carefully. The tensor T[i,j,k] for matrix multiplication
    satisfies: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    So T[i,j,k] = 1 if i==k, else 0.
    
    We can flatten this into a matrix M of shape (n*p, m) where M[(i,k), j] = T[i,j,k].
    The rank of this matrix gives a lower bound on the tensor rank.
    
    But actually, the tensor rank is the minimum number of rank-1 tensors that sum to T.
    A rank-1 tensor in this context is u_i * v_j * w_k for vectors u, v, w.
    
    Let's use the flattening approach: flatten T into a matrix and compute its rank.
    The flattening (i,k) vs j gives a matrix of shape (n*p, m).
    """
    # T[i,j,k] = 1 if i == k else 0
    # Flatten to matrix of shape (n*p, m)
    M = np.zeros((n * p, m))
    for i in range(n):
        for k in range(p):
            if i == k:
                M[i * p + k, :] = 1.0  # For all j
    return M

def _compute_flattened_ranks(n, m, p):
    """Compute ranks of different flattenings of the tensor."""
    ranks = {}
    
    # Flattening 1: (i,k) vs j -> shape (n*p, m)
    M1 = np.zeros((n * p, m))
    for i in range(n):
        for k in range(p):
            if i == k:
                M1[i * p + k, :] = 1.0
    ranks['ik_vs_j'] = np.linalg.matrix_rank(M1)
    
    # Flattening 2: (i,j) vs k -> shape (n*m, p)
    M2 = np.zeros((n * m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i == k:
                    M2[i * m + j, k] = 1.0
    ranks['ij_vs_k'] = np.linalg.matrix_rank(M2)
    
    # Flattening 3: (j,k) vs i -> shape (m*p, n)
    M3 = np.zeros((m * p, n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i == k:
                    M3[j * p + k, i] = 1.0
    ranks['jk_vs_i'] = np.linalg.matrix_rank(M3)
    
    return ranks

def _block_rank(ns, ms, ps):
    """
    Compute the rank needed for a single block of size ns x ms x ps.
    T[i,j,k] = delta[i,k] for i in [0,ns), j in [0,ms), k in [0,ps).
    
    Key insight: if ns > ps, then for i >= ps, T[i,j,k] = 0 for all j,k.
    So the effective size is min(ns,ps) x ms x ps.
    Similarly, if ps > ns, for k >= ns, T[i,j,k] = 0 for all i,j.
    So effective size is ns x ms x min(ns,ps).
    
    For 2x2x2: Strassen gives rank 7.
    For blocks that reduce to 2x2x2: rank 7.
    For 2x3x2 or 3x2x2 (when effective): need custom decompositions.
    """
    # Determine effective size
    eff_ns = min(ns, ps)
    eff_ps = min(ns, ps)
    
    if eff_ns == 2 and ms == 2 and eff_ps == 2:
        return 7
    if eff_ns == 2 and ms == 3 and eff_ps == 2:
        # 2x3x2: use Strassen for 2x2x2 part (7) + 2 terms for the extra j column
        return 9
    if eff_ns == 3 and ms == 2 and eff_ps == 2:
        # 3x2x2: effectively 2x2x2 since i=2 row is zero
        return 7
    if eff_ns == 2 and ms == 2 and eff_ps == 3:
        # 2x2x3: effectively 2x2x2 since k=2 column is zero
        return 7
    if eff_ns == 3 and ms == 3 and eff_ps == 2:
        # 3x3x2: effectively 2x3x2
        return 9
    if eff_ns == 3 and ms == 2 and eff_ps == 3:
        # 3x2x3: T[i,j,k]=delta[i,k], i in [0,3), j in [0,2), k in [0,3)
        # Non-zero: (0,0,0),(0,1,0),(1,0,1),(1,1,1),(2,0,2),(2,1,2)
        # Use Strassen for 2x2x2 (i=0,1; j=0,1; k=0,1) = 7
        # Plus 2 terms for i=2: (2,0,2) and (2,1,2)
        return 9
    if eff_ns == 2 and ms == 3 and eff_ps == 3:
        # 2x3x3: T[i,j,k]=delta[i,k], i in [0,2), j in [0,3), k in [0,3)
        # Non-zero: (0,0,0),(0,1,0),(0,2,0),(1,0,1),(1,1,1),(1,2,1)
        # Use Strassen for 2x2x2 (j=0,1; k=0,1) = 7
        # Plus 2 terms for j=2: (0,2,0) and (1,2,1)
        return 9
    if eff_ns == 3 and ms == 3 and eff_ps == 3:
        # 3x3x3: use schoolbook for safety
        return 27
    # Fallback: schoolbook
    return ns * ms * ps

def _compute_rank(n_blocks, m_blocks, p_blocks):
    """Compute the total rank for a given block partition."""
    rank = 0
    for ns in n_blocks:
        for ms in m_blocks:
            for ps in p_blocks:
                rank += _block_rank(ns, ms, ps)
    return rank

def _get_all_splits(dim, max_part=2):
    """Generate all compositions of dim into parts of size 1 and 2."""
    splits = []
    def _gen(remaining, current):
        if remaining == 0:
            splits.append(current[:])
            return
        for part in [1, 2]:
            if part <= remaining:
                current.append(part)
                _gen(remaining - part, current)
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
                rank = _compute_rank(ns, ms, ps)
                if rank < best_rank:
                    best_rank = rank
                    best_split = (ns, ms, ps)
    
    # Also try some specific promising splits
    specific_splits = [
        ([2,1], [2,2,1], [2,2,2,1]),
        ([2,1], [2,2,1], [2,2,1,2]),
        ([1,2], [2,2,1], [2,2,2,1]),
        ([2,1], [1,2,2], [2,2,2,1]),
        ([3], [2,2,1], [2,2,2,1]),
        ([2,1], [2,2,1], [2,1,2,2]),
        ([2,1], [2,1,2], [2,2,2,1]),
        ([1,2], [2,1,2], [2,2,2,1]),
        ([2,1], [2,2,1], [1,2,2,2]),
        ([2,1], [1,2,2], [2,2,1,2]),
    ]
    
    for ns, ms, ps in specific_splits:
        # Verify the splits are valid
        if sum(ns) == n and sum(ms) == m and sum(ps) == p:
            rank = _compute_rank(ns, ms, ps)
            if rank < best_rank:
                best_rank = rank
                best_split = (ns, ms, ps)
    
    return best_rank, best_split

def _strassen_terms():
    """
    Returns the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (A_factors, B_factors, C_factors) where each is a list of (row, col, coeff).
    A is indexed by (i, a), B by (b, k), C by (i, k) where W is indexed by (k, i).
    """
    return [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

def _strassen_322_decomposition():
    """
    Returns decomposition for 3x2x2 matrix multiplication.
    T[i,j,k] = delta[i,k] for i in [0,3), j in [0,2), k in [0,2).
    
    We can write this as sum of rank-1 tensors.
    The tensor has 3*2*2 = 12 non-zero entries (one for each (i,j,k) where i==k).
    Actually, T[i,j,k] = 1 if i==k else 0.
    For i=0: T[0,j,0] = 1 for j=0,1
    For i=1: T[1,j,1] = 1 for j=0,1
    For i=2: T[2,j,k] = 0 for all j,k (since k only goes to 1)
    
    So the tensor is:
    T[0,0,0] = 1, T[0,1,0] = 1
    T[1,0,1] = 1, T[1,1,1] = 1
    All other entries are 0.
    
    This can be decomposed as:
    e0_i * e0_j * e0_k + e0_i * e1_j * e0_k + e1_i * e0_j * e1_k + e1_i * e1_j * e1_k
    
    That's 4 terms. Can we do better?
    
    Let's try: 
    Term 1: a=[1,0,0], b=[1,1], c=[1,0] -> contributes to T[0,0,0] and T[0,1,0]
    Term 2: a=[0,1,0], b=[1,1], c=[0,1] -> contributes to T[1,0,1] and T[1,1,1]
    
    That's only 2 terms! But wait, we need to verify.
    
    T[i,j,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*3+i, r]
    
    With 2 terms:
    r=0: U[0,0]=1, U[1,0]=1, U[2,0]=0, U[3,0]=0, U[4,0]=0, U[5,0]=0
          V[0,0]=1, V[1,0]=0, V[2,0]=1, V[3,0]=0
          W[0,0]=1, W[1,0]=0, W[2,0]=0, W[3,0]=0, W[4,0]=0, W[5,0]=0
    
    Hmm, this is getting complex. Let me just use schoolbook for 3x2x2 and focus on optimizing the overall block structure.
    """
    return None

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
    
    rank = _compute_rank(n_blocks, m_blocks, p_blocks)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen_data = _strassen_terms()
    col = 0
    for ni in range(len(n_blocks)):
        for mi in range(len(m_blocks)):
            for pi in range(len(p_blocks)):
                n0 = n_offsets[ni]
                m0 = m_offsets[mi]
                p0 = p_offsets[pi]
                ns = n_blocks[ni]
                ms = m_blocks[mi]
                ps = p_blocks[pi]
                
                blk_rank = _block_rank(ns, ms, ps)
                
                # Determine effective dimensions
                eff_ns = min(ns, ps)
                eff_ps = min(ns, ps)
                
                if eff_ns == 2 and ms == 2 and eff_ps == 2:
                    # Pure 2x2x2 Strassen
                    for a_factors, b_factors, c_factors in strassen_data:
                        for (ar, ac, acoeff) in a_factors:
                            U[(n0 + ar) * m + (m0 + ac), col] += acoeff
                        for (br, bc, bcoeff) in b_factors:
                            V[(m0 + br) * p + (p0 + bc), col] += bcoeff
                        for (cr, cc, ccoeff) in c_factors:
                            W[(p0 + cc) * n + (n0 + cr), col] += ccoeff
                        col += 1
                elif eff_ns == 2 and ms == 3 and eff_ps == 2:
                    # 2x3x2: Strassen for j=0,1 + schoolbook for j=2
                    for a_factors, b_factors, c_factors in strassen_data:
                        for (ar, ac, acoeff) in a_factors:
                            U[(n0 + ar) * m + (m0 + ac), col] += acoeff
                        for (br, bc, bcoeff) in b_factors:
                            V[(m0 + br) * p + (p0 + bc), col] += bcoeff
                        for (cr, cc, ccoeff) in c_factors:
                            W[(p0 + cc) * n + (n0 + cr), col] += ccoeff
                        col += 1
                    # j=2 terms
                    for i in range(eff_ns):
                        U[(n0 + i) * m + (m0 + 2), col] = 1
                        V[(m0 + 2) * p + (p0 + i), col] = 1
                        W[(p0 + i) * n + (n0 + i), col] = 1
                        col += 1
                elif eff_ns == 3 and ms == 2 and eff_ps == 2:
                    # 3x2x2: effectively 2x2x2 (i=2 row is zero)
                    for a_factors, b_factors, c_factors in strassen_data:
                        for (ar, ac, acoeff) in a_factors:
                            U[(n0 + ar) * m + (m0 + ac), col] += acoeff
                        for (br, bc, bcoeff) in b_factors:
                            V[(m0 + br) * p + (p0 + bc), col] += bcoeff
                        for (cr, cc, ccoeff) in c_factors:
                            W[(p0 + cc) * n + (n0 + cr), col] += ccoeff
                        col += 1
                elif eff_ns == 2 and ms == 2 and eff_ps == 3:
                    # 2x2x3: effectively 2x2x2 (k=2 col is zero)
                    for a_factors, b_factors, c_factors in strassen_data:
                        for (ar, ac, acoeff) in a_factors:
                            U[(n0 + ar) * m + (m0 + ac), col] += acoeff
                        for (br, bc, bcoeff) in b_factors:
                            V[(m0 + br) * p + (p0 + bc), col] += bcoeff
                        for (cr, cc, ccoeff) in c_factors:
                            W[(p0 + cc) * n + (n0 + cr), col] += ccoeff
                        col += 1
                elif eff_ns == 3 and ms == 3 and eff_ps == 2:
                    # 3x3x2: effectively 2x3x2
                    for a_factors, b_factors, c_factors in strassen_data:
                        for (ar, ac, acoeff) in a_factors:
                            U[(n0 + ar) * m + (m0 + ac), col] += acoeff
                        for (br, bc, bcoeff) in b_factors:
                            V[(m0 + br) * p + (p0 + bc), col] += bcoeff
                        for (cr, cc, ccoeff) in c_factors:
                            W[(p0 + cc) * n + (n0 + cr), col] += ccoeff
                        col += 1
                    for i in range(2):
                        U[(n0 + i) * m + (m0 + 2), col] = 1
                        V[(m0 + 2) * p + (p0 + i), col] = 1
                        W[(p0 + i) * n + (n0 + i), col] = 1
                        col += 1
                elif eff_ns == 3 and ms == 2 and eff_ps == 3:
                    # 3x2x3: Strassen for i=0,1; k=0,1 + schoolbook for i=2,k=2
                    for a_factors, b_factors, c_factors in strassen_data:
                        for (ar, ac, acoeff) in a_factors:
                            U[(n0 + ar) * m + (m0 + ac), col] += acoeff
                        for (br, bc, bcoeff) in b_factors:
                            V[(m0 + br) * p + (p0 + bc), col] += bcoeff
                        for (cr, cc, ccoeff) in c_factors:
                            W[(p0 + cc) * n + (n0 + cr), col] += ccoeff
                        col += 1
                    # i=2, k=2 terms
                    for j in range(ms):
                        U[(n0 + 2) * m + (m0 + j), col] = 1
                        V[(m0 + j) * p + (p0 + 2), col] = 1
                        W[(p0 + 2) * n + (n0 + 2), col] = 1
                        col += 1
                elif eff_ns == 2 and ms == 3 and eff_ps == 3:
                    # 2x3x3: Strassen for j=0,1; k=0,1 + schoolbook for j=2
                    for a_factors, b_factors, c_factors in strassen_data:
                        for (ar, ac, acoeff) in a_factors:
                            U[(n0 + ar) * m + (m0 + ac), col] += acoeff
                        for (br, bc, bcoeff) in b_factors:
                            V[(m0 + br) * p + (p0 + bc), col] += bcoeff
                        for (cr, cc, ccoeff) in c_factors:
                            W[(p0 + cc) * n + (n0 + cr), col] += ccoeff
                        col += 1
                    for i in range(eff_ns):
                        U[(n0 + i) * m + (m0 + 2), col] = 1
                        V[(m0 + 2) * p + (p0 + i), col] = 1
                        W[(p0 + i) * n + (n0 + i), col] = 1
                        col += 1
                else:
                    # Fallback: schoolbook
                    for i in range(ns):
                        for j in range(ms):
                            for k in range(ps):
                                U[(n0 + i) * m + (m0 + j), col] = 1
                                V[(m0 + j) * p + (p0 + k), col] = 1
                                W[(p0 + k) * n + (n0 + i), col] = 1
                                col += 1
    
    return U, V, W

def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _try_direct_rank_decomposition(n, m, p, target_rank, time_limit):
    """
    Try to find a rank-`target_rank` decomposition using the structure of the tensor.
    
    The tensor T[i,j,k] = delta[i,k]. We want to write:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For each r, the contribution is u_r[i] * v_r[j] * w_r[k] where:
    - U row index = i*m + j, so U[i*m+j, r] = u_r[i] * v_r[j] (outer product structure)
    - V row index = j*p + k, so V[j*p+k, r] = v_r[j] * w_r[k]
    - W row index = k*n + i, so W[k*n+i, r] = w_r[k] * u_r[i]
    
    Wait, that's not quite right. Let me reconsider.
    
    U is shape (n*m, rank), V is shape (m*p, rank), W is shape (p*n, rank).
    The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    We need T[i,j,k] = 1 if i==k else 0.
    
    Let's try a structured approach. For each r, define:
    - a_r: vector of length n (for i-index in U and W)
    - b_r: vector of length m (for j-index in U and V)  
    - c_r: vector of length p (for k-index in V and W)
    
    Then U[i*m+j, r] = a_r[i] * b_r[j]
         V[j*p+k, r] = b_r[j] * c_r[k]
         W[k*n+i, r] = c_r[k] * a_r[i]
    
    So T[i,j,k] = sum_r (a_r[i]*b_r[j]) * (b_r[j]*c_r[k]) * (c_r[k]*a_r[i])
                = sum_r a_r[i]^2 * b_r[j]^2 * c_r[k]^2
    
    We need: sum_r a_r[i]^2 * b_r[j]^2 * c_r[k]^2 = delta[i,k] for all i,j,k.
    
    This is a system of equations. For i != k, we need sum_r a_r[i]^2 * b_r[j]^2 * c_r[k]^2 = 0.
    For i == k, we need sum_r a_r[i]^2 * b_r[j]^2 * c_r[i]^2 = 1.
    
    This is complex. Let me try a different approach: use the block decomposition but optimize it.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # First, compute flattened ranks to understand the structure
    ranks = _compute_flattened_ranks(n, m, p)
    
    # Search for best block split
    best_rank, best_split = _find_best_block_split(n, m, p)
    
    if time.time() > deadline:
        U, V, W = _schoolbook_decomposition(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try the best split and several promising alternatives
    candidate_splits = []
    if best_split is not None:
        candidate_splits.append(best_split)
    
    # Add promising splits with larger blocks
    extra_splits = [
        ([2,1], [2,3], [2,2,3]),
        ([2,1], [3,2], [2,2,3]),
        ([2,1], [2,3], [3,2,2]),
        ([2,1], [3,2], [3,2,2]),
        ([2,1], [2,3], [2,3,2]),
        ([2,1], [3,2], [2,3,2]),
        ([2,1], [2,3], [3,3,1]),
        ([2,1], [3,2], [3,3,1]),
        ([3], [2,3], [2,2,3]),
        ([3], [3,2], [2,2,3]),
        ([3], [2,3], [3,2,2]),
        ([3], [3,2], [3,2,2]),
        ([2,1], [2,2,1], [2,2,2,1]),
        ([2,1], [2,2,1], [2,2,1,2]),
        ([1,2], [2,2,1], [2,2,2,1]),
        ([2,1], [1,2,2], [2,2,2,1]),
        ([3], [2,2,1], [2,2,2,1]),
    ]
    for s in extra_splits:
        if s not in candidate_splits and sum(s[0])==n and sum(s[1])==m and sum(s[2])==p:
            candidate_splits.append(s)
    
    best_result = None
    best_result_rank = float('inf')
    
    for ns, ms, ps in candidate_splits:
        if time.time() > deadline - 5:
            break
        try:
            U, V, W = _block_decomposition(n, m, p, ns, ms, ps)
            if _verify_fast(U, V, W, n, m, p):
                r = U.shape[1]
                if r < best_result_rank:
                    best_result_rank = r
                    best_result = (U, V, W)
        except Exception:
            continue
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
