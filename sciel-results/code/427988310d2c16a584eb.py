"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    U[:,0]=[1,0,0,1]; V[:,0]=[1,0,0,1]; W[:,0]=[1,0,0,1]
    U[:,1]=[0,0,1,1]; V[:,1]=[1,0,0,0]; W[:,1]=[0,1,0,-1]
    U[:,2]=[1,0,0,0]; V[:,2]=[0,1,0,-1]; W[:,2]=[0,0,1,1]
    U[:,3]=[0,0,0,1]; V[:,3]=[-1,0,1,0]; W[:,3]=[1,1,0,0]
    U[:,4]=[1,1,0,0]; V[:,4]=[0,0,0,1]; W[:,4]=[-1,0,1,0]
    U[:,5]=[-1,0,1,0]; V[:,5]=[1,1,0,0]; W[:,5]=[0,0,0,1]
    U[:,6]=[0,1,0,-1]; V[:,6]=[0,0,1,1]; W[:,6]=[1,0,0,0]
    return U, V, W

def _winograd_3x3x3():
    """Rank-23 Winograd-type decomposition for 3x3x3 matrix multiplication.
    Indexing: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    r = 23
    U = np.zeros((9, r)); V = np.zeros((9, r)); W = np.zeros((9, r))
    # 6 terms of form (e_i+e_j)(e_k+e_l)(e_m+e_n) with i!=j, k!=l, m!=n
    pairs = [(0,1),(0,2),(1,2)]
    col = 0
    for (i,j) in pairs:
        for (k,l) in pairs:
            for (m,n) in pairs:
                U[i*3+j, col] = 1
                V[k*3+l, col] = 1
                W[m*3+n, col] = 1
                col += 1
    # 6 terms of form (e_i+e_j)(e_k)(e_m) with i!=j
    for (i,j) in pairs:
        for k in range(3):
            for m in range(3):
                if k == 0:  # skip duplicates handled above? No, these are different
                    pass
                U[i*3+j, col] = 1
                V[k*3+0, col] = 1  # This won't work, need to be more careful
                col += 1
    # Let me use a known explicit rank-23 construction
    # The standard Winograd 3x3x3 rank-23 uses the following structure:
    # 6 terms: (e0+e1)(e0+e1)(e0+e1), etc. -- actually let me use the known one
    # 
    # Known rank-23 for 3x3x3 (from Winograd 1980):
    # Uses the fact that 3x3x3 tensor can be decomposed with 23 terms
    # Let me construct it properly using the standard algorithm
    
    U = np.zeros((9, r)); V = np.zeros((9, r)); W = np.zeros((9, r))
    
    # The 3x3x3 matmul tensor T[i,j,k] = delta_{i,k} (C[i,k] = sum_j A[i,j]B[j,k])
    # Wait, T[i,j,k] = 1 for all i,j,k in the standard tensor
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid (i,j,k)
    
    # Standard rank-23 construction for 3x3x3:
    # 6 terms of type (a_i + a_j)(b_k + b_l)(c_m + c_n) where {i,j} != {k,l} != {m,n}
    # 6 terms of type (a_i + a_j)(b_k)(c_m)  
    # 6 terms of type (a_i)(b_k + b_l)(c_m)
    # 5 terms of type (a_i)(b_k)(c_m)
    
    # Let me use a verified construction
    # Term 1-6: sum over all (i,j) pairs with i<j, and k,l,m,n
    col = 0
    # 6 terms: (e_i+e_j) x (e_k+e_l) x (e_m+e_n)
    for i, j in [(0,1),(0,2),(1,2)]:
        for k, l in [(0,1),(0,2),(1,2)]:
            for m, n in [(0,1),(0,2),(1,2)]:
                if col >= r:
                    break
                U[i*3+j, col] = 1
                V[k*3+l, col] = 1
                W[m*3+n, col] = 1
                col += 1
            if col >= r:
                break
        if col >= r:
            break
    
    # This is getting complicated. Let me use a different approach:
    # Use the known fact that rank(3x3x3) = 23 and construct via
    # combining smaller blocks.
    
    # Actually, let me just verify if we can get a valid rank-23 by
    # using the block structure: split 3 = 1+2
    # 1x2x2 block: rank 4 (schoolbook)
    # 2x1x2 block: rank 4
    # 2x2x1 block: rank 4  
    # 2x2x2 block: rank 7 (Strassen)
    # Total: 4+4+4+7 = 19? No, that's not right because of cross terms.
    
    # Let me just use schoolbook for 3x3x3 (rank 27) for now,
    # and focus on the overall block structure optimization.
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _embed_block(U_local, V_local, W_local, n, m, p, n_off, m_off, p_off, ni, mi, pi):
    r = U_local.shape[1]
    U_g = np.zeros((n * m, r))
    V_g = np.zeros((m * p, r))
    W_g = np.zeros((p * n, r))
    for li in range(ni):
        for lj in range(mi):
            U_g[(n_off+li)*m + (m_off+lj), :] = U_local[li*mi+lj, :]
    for lj in range(mi):
        for lk in range(pi):
            V_g[(m_off+lj)*p + (p_off+lk), :] = V_local[lj*pi+lk, :]
    for lk in range(pi):
        for li in range(ni):
            W_g[(p_off+lk)*n + (n_off+li), :] = W_local[lk*ni+li, :]
    return U_g, V_g, W_g

def _block_rank(ni, mi, pi):
    """Return the best known rank for a ni x mi x pi block."""
    if ni == 2 and mi == 2 and pi == 2:
        return 7
    if ni == 1 and mi == 1 and pi == 1:
        return 1
    # For 3x3x3, we could potentially use rank 23, but let's be conservative
    # and use schoolbook for now unless we have a verified construction
    return ni * mi * pi

def _block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    n_offsets = [sum(n_parts[:i]) for i in range(len(n_parts))]
    m_offsets = [sum(m_parts[:i]) for i in range(len(m_parts))]
    p_offsets = [sum(p_parts[:i]) for i in range(len(p_parts))]
    
    total_rank = 0
    block_ranks = {}
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                block_ranks[(i, j, k)] = _block_rank(ni, mi, pi)
                total_rank += block_ranks[(i, j, k)]
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                if ni == 2 and mi == 2 and pi == 2:
                    U_local, V_local, W_local = _strassen_2x2x2()
                else:
                    U_local, V_local, W_local = _schoolbook(ni, mi, pi)
                
                U_g, V_g, W_g = _embed_block(
                    U_local, V_local, W_local,
                    n, m, p,
                    n_offsets[i], m_offsets[j], p_offsets[k],
                    ni, mi, pi
                )
                
                U[:, col:col+block_ranks[(i,j,k)]] += U_g
                V[:, col:col+block_ranks[(i,j,k)]] += V_g
                W[:, col:col+block_ranks[(i,j,k)]] += W_g
                col += block_ranks[(i,j,k)]
    
    return U, V, W, total_rank

def _generate_partitions(d, max_parts=5):
    """Generate all compositions of d into up to max_parts positive parts."""
    results = []
    def _gen(remaining, parts_left, current):
        if remaining == 0:
            results.append(tuple(current))
            return
        if parts_left == 0:
            return
        for first in range(1, remaining + 1):
            current.append(first)
            _gen(remaining - first, parts_left - 1, current)
            current.pop()
    _gen(d, max_parts, [])
    return results

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    rank = U.shape[1]
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_reduction(U, V, W, n, m, p, target_rank, max_iter=50):
    """
    Try to reduce the rank by finding linear dependencies among the terms.
    If sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T, and there's a linear relation
    among the terms, we can reduce the rank.
    
    We look for a matrix M of shape (r, r') with r' < r such that
    the decomposition with M applied still gives the same tensor.
    """
    r = U.shape[1]
    if r <= target_rank:
        return U, V, W, r
    
    # The tensor is T = sum_c U_c ⊗ V_c ⊗ W_c
    # We want to find if there's a lower rank representation.
    # One approach: check if the "slice" matrices have low rank structure.
    
    # Form the matrix S of shape (n*m*p, r) where S[i*m*p + j*p + k, c] 
    # = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Actually, the tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # For this to equal 1 for all (i,j,k), we need the sum to work out.
    
    # A simpler approach: try to find a rank-reducing transformation.
    # If we have terms c1, c2 such that U[:,c1] = a*U[:,c2], V[:,c1] = b*V[:,c2],
    # W[:,c1] = c*W[:,c2] with a*b*c = 1, then we can merge them.
    
    # For now, let's try a simple approach: check for duplicate or proportional terms
    # and merge them.
    
    # Actually, the block decomposition already gives an exact decomposition.
    # The question is whether we can do better.
    
    # Let's try a different approach: use the fact that for the block structure,
    # cross-block terms might be combined.
    
    # For now, return the input unchanged (no reduction found).
    return U, V, W, r

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    best_rank = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Generate all partitions
    n_parts_list = _generate_partitions(n, max_parts=5)
    m_parts_list = _generate_partitions(m, max_parts=5)
    p_parts_list = _generate_partitions(p, max_parts=5)
    
    # Sort partitions by estimated rank (ascending)
    def est_rank(np_, mp_, pp_):
        total = 0
        for ni, mi, pi in zip(np_, mp_, pp_):
            total += _block_rank(ni, mi, pi)
        return total
    
    all_partitions = []
    for np_ in n_parts_list:
        for mp_ in m_parts_list:
            for pp_ in p_parts_list:
                r = est_rank(np_, mp_, pp_)
                all_partitions.append((r, np_, mp_, pp_))
    
    all_partitions.sort(key=lambda x: x[0])
    
    for rank_est, np_, mp_, pp_ in all_partitions:
        if time.time() > deadline - 10:
            break
        if rank_est >= best_rank:
            break  # No point continuing if sorted
        
        try:
            U, V, W, rank = _block_decomposition(n, m, p, np_, mp_, pp_)
            if rank < best_rank:
                if _verify_decomposition(U, V, W, n, m, p):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
        except Exception:
            pass
    
    if best_U is None:
        best_U, best_V, best_W = _schoolbook(n, m, p)
        best_rank = n * m * p
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
