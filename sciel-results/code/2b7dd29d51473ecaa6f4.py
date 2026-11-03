"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p).
    The tensor has dimensions (n*m, m*p, p*n) indexed by (i*m+j, j*p+k, k*n+i)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor.
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    The decomposition satisfies: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    """
    T = _build_tensor(n, m, p)
    T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2.
    U: (4,7), V: (4,7), W: (4,7)
    T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    where i = i1*2+i2, j = j1*2+j2, k = k1*2+k2
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # c=0: M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # c=1: M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # c=2: M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # c=3: M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # c=4: M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # c=5: M6 = (c-d)(f+h)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1
    
    # c=6: M7 = (b-d)(e+g)
    U[1, 6] = 1; U[3, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_block_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
    """Build a decomposition using a mix of Strassen (2x2x2) and schoolbook blocks.
    
    n_blocks, m_blocks, p_blocks: lists of tuples of indices.
    For each triple (nb, mb, pb):
      - if len(nb)==2 and len(mb)==2 and len(pb)==2: use Strassen (rank 7)
      - else: use schoolbook (rank = sn*sm*sp)
    
    Returns (U, V, W, rank) or None if invalid.
    """
    total_rank = 0
    for nb in n_blocks:
        for mb in m_blocks:
            for pb in p_blocks:
                sn, sm, sp = len(nb), len(mb), len(pb)
                if sn == 2 and sm == 2 and sp == 2:
                    total_rank += 7
                else:
                    total_rank += sn * sm * sp
    
    if total_rank >= n * m * p:
        return None
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for nb in n_blocks:
        for mb in m_blocks:
            for pb in p_blocks:
                sn, sm, sp = len(nb), len(mb), len(pb)
                if sn == 2 and sm == 2 and sp == 2:
                    U_s, V_s, W_s = _strassen_2x2x2()
                    for ci in range(sn):
                        for cj in range(sm):
                            for ck in range(sp):
                                i = nb[ci]
                                j = mb[cj]
                                k = pb[ck]
                                for c in range(7):
                                    U[i * m + j, col + c] = U_s[ci * 2 + cj, c]
                                    V[j * p + k, col + c] = V_s[cj * 2 + ck, c]
                                    W[k * n + i, col + c] = W_s[ck * 2 + ci, c]
                    col += 7
                else:
                    for ci in range(sn):
                        for cj in range(sm):
                            for ck in range(sp):
                                i = nb[ci]
                                j = mb[cj]
                                k = pb[ck]
                                U[i * m + j, col] = 1
                                V[j * p + k, col] = 1
                                W[k * n + i, col] = 1
                                col += 1
    
    if col != total_rank:
        return None
    
    if not _check_decomposition(U, V, W, n, m, p):
        return None
    
    return U, V, W, total_rank

def _generate_partitions(n):
    """Generate all ways to partition range(n) into contiguous blocks of size 1 or 2.
    Returns a list of partitions, where each partition is a list of tuples."""
    results = []
    def backtrack(start, current):
        if start >= n:
            results.append(list(current))
            return
        # Try block of size 1
        current.append((start,))
        backtrack(start + 1, current)
        current.pop()
        # Try block of size 2
        if start + 1 < n:
            current.append((start, start + 1))
            backtrack(start + 2, current)
            current.pop()
    backtrack(0, [])
    return results

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Start with schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Generate all partitions for n, m, p
    n_parts = _generate_partitions(n)
    m_parts = _generate_partitions(m)
    p_parts = _generate_partitions(p)
    
    # Try all combinations of partitions
    for nb in n_parts:
        for mb in m_parts:
            for pb in p_parts:
                elapsed = time.time() - start_time
                if elapsed > deadline - 10:
                    break
                result = _build_block_decomposition(n, m, p, nb, mb, pb)
                if result is not None:
                    U, V, W, rank = result
                    if rank < best_rank:
                        best_U, best_V, best_W = U, V, W
                        best_rank = rank
            if time.time() - start_time > deadline - 10:
                break
        if time.time() - start_time > deadline - 10:
            break
    
    # Verify final result
    if _check_decomposition(best_U, best_V, best_W, n, m, p):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
