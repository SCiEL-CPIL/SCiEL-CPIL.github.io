"""Matrix multiplication tensor decomposition for 3x4 and 4x7 matrices."""
import numpy as np
import itertools
import time

DIMENSIONS = [3, 4, 7]

def _build_decomp(n, m, p, ranks):
    """
    ranks: list of (n_i, m_i, p_i) tuples that partition n, m, p
    Returns U, V, W as lists of lists
    """
    # Partition n into segments
    n_parts = []
    idx = 0
    for (ni, mi, pi) in ranks:
        n_parts.append((idx, idx + ni))
        idx += ni
    
    m_parts = []
    idx = 0
    for (ni, mi, pi) in ranks:
        m_parts.append((idx, idx + mi))
        idx += mi
    
    p_parts = []
    idx = 0
    for (ni, mi, pi) in ranks:
        p_parts.append((idx, idx + pi))
        idx += pi
    
    total_rank = sum(ni * mi * pi for (ni, mi, pi) in ranks)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for part_idx, (ni, mi, pi) in enumerate(ranks):
        n_start, n_end = n_parts[part_idx]
        m_start, m_end = m_parts[part_idx]
        p_start, p_end = p_parts[part_idx]
        
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    U[(n_start + i) * m + (m_start + j)][col] = 1
                    V[(m_start + j) * p + (p_start + k)][col] = 1
                    W[(p_start + k) * n + (n_start + i)][col] = 1
                    col += 1
    
    return U, V, W

def _validate(U, V, W, n, m, p):
    """Check if decomposition is exact"""
    rank = len(U[0]) if U else 0
    # Build tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 if i,j,k valid, else 0 (but indices are always valid)
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m + j]
                v_val = V[j*p + k]
                w_val = W[k*n + i]
                tensor_val = np.dot(u_val, v_val * w_val)
                if abs(tensor_val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin
    
    n, m, p = DIMENSIONS
    
    # Generate all possible partitions of n, m, p
    # A partition of d is a list of positive integers summing to d
    def get_partitions(d, max_part=None):
        if max_part is None:
            max_part = d
        if d == 0:
            yield ()
            return
        for first in range(1, min(d, max_part) + 1):
            for rest in get_partitions(d - first, first):
                yield (first,) + rest
    
    n_parts = list(get_partitions(n))
    m_parts = list(get_partitions(m))
    p_parts = list(get_partitions(p))
    
    best_rank = n * m * p  # schoolbook
    best_decomp = None
    
    # Try all combinations of partitions
    for npart in n_parts:
        if time.time() > deadline - 5.0:
            break
        for mpart in m_parts:
            if time.time() > deadline - 5.0:
                break
            for ppart in p_parts:
                # All partitions must have the same length
                if len(npart) != len(mpart) or len(mpart) != len(ppart):
                    continue
                
                # Build the list of (ni, mi, pi)
                blocks = list(zip(npart, mpart, ppart))
                rank = sum(ni * mi * pi for (ni, mi, pi) in blocks)
                
                if rank >= best_rank:
                    continue
                
                # Build decomposition
                U, V, W = _build_decomp(n, m, p, blocks)
                
                # Validate
                if _validate(U, V, W, n, m, p):
                    best_rank = rank
                    best_decomp = (U, V, W)
                    break
            
            if time.time() > deadline - 5.0:
                break
        
        if time.time() > deadline - 5.0:
            break
    
    if best_decomp is None:
        # Fallback to schoolbook
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
        return {"U": U, "V": V, "W": W}
    
    U, V, W = best_decomp
    return {"U": U, "V": V, "W": W}
