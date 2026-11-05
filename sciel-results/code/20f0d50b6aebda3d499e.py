"""Tensor decomposition for 3x5 * 5x7 matrix multiplication - rank reduction via Strassen blocks."""
import time
import numpy as np
from itertools import product

N, M, P = 3, 5, 7
_DEADLINE = None


def _strassen_2x2x2():
    """Strassen decomposition for 2x2 * 2x2, rank 7.
    Returns U (4,7), V (4,7), W (4,7) with convention:
    U indexed by (i,j) for A[i,j], V by (j,k) for B[j,k], W by (k,i) for C[k,i]
    """
    U = np.zeros((4, 7))
    U[0, 0] = 1
    U[0, 1] = 1
    U[2, 1] = 1
    U[3, 2] = 1
    U[1, 3] = 1
    U[3, 3] = -1
    U[0, 4] = 1
    U[3, 4] = 1
    U[1, 5] = 1
    U[0, 5] = -1
    U[2, 6] = 1
    U[3, 6] = -1
    
    V = np.zeros((4, 7))
    V[1, 0] = 1
    V[3, 0] = -1
    V[3, 1] = 1
    V[0, 2] = 1
    V[2, 2] = -1
    V[2, 3] = 1
    V[3, 3] = 1
    V[0, 4] = 1
    V[3, 4] = 1
    V[0, 5] = 1
    V[2, 5] = 1
    V[1, 6] = 1
    V[3, 6] = 1
    
    W = np.zeros((4, 7))
    W[0, 4] = 1
    W[0, 3] = 1
    W[0, 1] = -1
    W[0, 6] = -1
    W[2, 0] = 1
    W[2, 1] = 1
    W[1, 2] = 1
    W[1, 3] = 1
    W[3, 4] = 1
    W[3, 0] = 1
    W[3, 2] = -1
    W[3, 5] = -1
    
    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook decomposition for n*m * m*p."""
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


def _block_3x2x3():
    """Decomposition for 3x2 * 2x3.
    Use Strassen-like: split the 2x2 core.
    Actually, 3x2*2x3: for each of 3 rows of A, we have 2x2*2x3... no.
    A is 3x2, B is 2x3, C is 3x3.
    C[i,k] = sum_j A[i,j]*B[j,k]
    This is 3 independent 1x2*2x3 multiplications? No, they share the B matrix.
    Schoolbook rank = 3*2*3 = 18.
    Let's just use schoolbook for now, or try to find a better one.
    """
    n, m, p = 3, 2, 3
    return _schoolbook(n, m, p)


def _block_3x2x2():
    """Decomposition for 3x2 * 2x2.
    A is 3x2, B is 2x2, C is 3x2.
    C[i,k] = sum_j A[i,j]*B[j,k]
    Schoolbook rank = 3*2*2 = 12.
    For each row i of A, we compute a 1x2 * 2x2 = 1x2 result.
    But all rows share the same B.
    We can use Strassen on the 2x2 part: for each of 3 rows, use 7 mults = 21 total. Worse.
    Actually, the structure is: C = A @ B where A is 3x2, B is 2x2.
    We can think of it as 3 separate 1x2 * 2x2 multiplications, but they share B.
    Schoolbook: 12 multiplications.
    Alternative: use the fact that B is 2x2. Write C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:].
    That's 2 multiplications per row, 3 rows = 6... but that's not rank-1 tensor terms.
    Actually schoolbook IS 12 rank-1 terms. Let me just use schoolbook.
    """
    n, m, p = 3, 2, 2
    return _schoolbook(n, m, p)


def _block_3x1x3():
    """Decomposition for 3x1 * 1x3. Rank = 3*1*3 = 9 (schoolbook, and it's optimal since it's outer product of column and row)."""
    n, m, p = 3, 1, 3
    return _schoolbook(n, m, p)


def _block_3x1x1():
    """Decomposition for 3x1 * 1x1. Rank = 3."""
    n, m, p = 3, 1, 1
    return _schoolbook(n, m, p)


def _combine_blocks(blocks, n, m, p):
    """Combine block decompositions into full decomposition.
    blocks: list of (m_start, m_end, p_start, p_end, U, V, W)
    U is (n*(m_end-m_start), rank), V is ((m_end-m_start)*(p_end-p_start), rank), W is ((p_end-p_start)*n, rank)
    """
    total_rank = sum(W.shape[1] for (_, _, _, _, _, _, W) in blocks)
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    offset = 0
    for (m_start, m_end, p_start, p_end, U_blk, V_blk, W_blk) in blocks:
        rank_blk = W_blk.shape[1]
        m_size = m_end - m_start
        p_size = p_end - p_start
        
        for i in range(n):
            for j in range(m_size):
                U[i * m + (m_start + j), offset:offset + rank_blk] = U_blk[i * m_size + j, :]
        for j in range(m_size):
            for k in range(p_size):
                V[(m_start + j) * p + (p_start + k), offset:offset + rank_blk] = V_blk[j * p_size + k, :]
        for k in range(p_size):
            for i in range(n):
                W[(p_start + k) * n + i, offset:offset + rank_blk] = W_blk[k * n + i, :]
        offset += rank_blk
    
    return U, V, W

def _get_best_block(n, m, p):
    """Get best known decomposition for n×m×p block."""
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    # For 3×2×2, try to see if we can do better than schoolbook
    # For now, use schoolbook
    return _schoolbook(n, m, p)

def _all_splits(total, max_part=2):
    """Generate all ways to split total into parts of size 1 or max_part."""
    if total == 0:
        yield []
    else:
        for part in range(1, min(max_part, total) + 1):
            for rest in _all_splits(total - part, max_part):
                yield [part] + rest


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    r = U.shape[1]
    # T[a,b,d] should be 1 if a=i*m+j, b=j*p+k, d=k*n+i for same i,j,k, else 0
    T = np.zeros((n * m, m * p, p * n))
    for c in range(r):
        T += np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                if abs(T[a, b, d] - 1.0) > tol:
                    return False
    # Check that all other entries are 0
    for a in range(n * m):
        for b in range(m * p):
            for d in range(p * n):
                i = a // m
                j = a % m
                j2 = b // p
                k2 = b % p
                k = d // n
                i2 = d % n
                if i == i2 and j == j2 and k == k2:
                    continue
                if abs(T[a, b, d]) > tol:
                    return False
    return True


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 140
    
    n, m, p = N, M, P
    
    # Schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p  # 105
    
    # Search over all block splits using parts of size 1 and 2
    # For each block, use Strassen if 2x2x2, else schoolbook
    m_splits = list(_all_splits(m, max_part=2))
    p_splits = list(_all_splits(p, max_part=2))
    
    for m_split in m_splits:
        if time.time() > _DEADLINE - 5:
            break
        for p_split in p_splits:
            if time.time() > _DEADLINE - 5:
                break
            
            blocks = []
            m_start = 0
            for m_part in m_split:
                p_start = 0
                for p_part in p_split:
                    U_b, V_b, W_b = _get_best_block(n, m_part, p_part)
                    blocks.append((m_start, m_start + m_part, p_start, p_start + p_part, U_b, V_b, W_b))
                    p_start += p_part
                m_start += m_part
            
            U, V, W = _combine_blocks(blocks, n, m, p)
            rank = U.shape[1]
            if _verify(U, V, W, n, m, p):
                if rank < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
    
    # Convert to lists for the return interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
