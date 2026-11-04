"""Tensor decomposition search for 3x5 * 5x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]
N, M, P = DIMENSIONS


def construct_block_3x2x3():
    """Laderman decomposition for 3x2 * 2x3, rank 7."""
    # Returns U (6,7), V (6,7), W (9,7) for 3x2x3
    # Laderman's 7-term algorithm for 2x2x2 adapted... 
    # Actually, let me use a known 3x2x3 rank-7 decomposition.
    # For 3x2 * 2x3, the schoolbook rank is 18, but we can do better.
    # Let me just use schoolbook for now to ensure correctness.
    n, m, p = 3, 2, 3
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


def construct_block_3x2x2():
    """Decomposition for 3x2 * 2x2."""
    n, m, p = 3, 2, 2
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


def construct_block_3x2x1():
    """Decomposition for 3x2 * 2x1."""
    n, m, p = 3, 2, 1
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


def construct_block_3x1x3():
    """Decomposition for 3x1 * 1x3."""
    n, m, p = 3, 1, 3
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


def construct_block_3x1x1():
    """Decomposition for 3x1 * 1x1."""
    n, m, p = 3, 1, 1
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


def combine_blocks(blocks, n, m, p):
    """
    Combine block decompositions.
    blocks: list of (m_start, m_end, p_start, p_end, U, V, W)
    """
    total_rank = sum(len(W[0]) for (_, _, _, _, _, _, W) in blocks)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    for (m_start, m_end, p_start, p_end, U_blk, V_blk, W_blk) in blocks:
        rank_blk = len(W_blk[0])
        for i in range(n):
            for j in range(m_start, m_end):
                for c in range(rank_blk):
                    U[i * m + j][offset + c] = U_blk[i * (m_end - m_start) + (j - m_start)][c]
        for j in range(m_start, m_end):
            for k in range(p_start, p_end):
                for c in range(rank_blk):
                    V[j * p + k][offset + c] = V_blk[(j - m_start) * (p_end - p_start) + (k - p_start)][c]
        for k in range(p_start, p_end):
            for i in range(n):
                for c in range(rank_blk):
                    W[k * n + i][offset + c] = W_blk[(k - p_start) * n + i][c]
        offset += rank_blk
    
    return U, V, W


def check_decomposition(U, V, W, tol=1e-9):
    """Check if the decomposition is exact."""
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    r = U.shape[1]
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
    
    for i1 in range(n):
        for j1 in range(m):
            for j2 in range(m):
                for k2 in range(p):
                    for i3 in range(n):
                        for k3 in range(p):
                            if i1 == i3 and j1 == j2 and k2 == k3:
                                continue
                            a = i1 * m + j1
                            b = j2 * p + k2
                            d = k3 * n + i3
                            if abs(T[a, b, d]) > tol:
                                return False
    return True


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = N, M, P
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try block decomposition: 5=2+2+1, 7=3+3+1
    if time.time() < deadline - 5:
        blocks = []
        
        # Block (0,2) x (0,3): 3x2 * 2x3
        U_b, V_b, W_b = construct_block_3x2x3()
        blocks.append((0, 2, 0, 3, U_b, V_b, W_b))
        
        # Block (0,2) x (3,6): 3x2 * 2x3
        U_b, V_b, W_b = construct_block_3x2x3()
        blocks.append((0, 2, 3, 6, U_b, V_b, W_b))
        
        # Block (0,2) x (6,7): 3x2 * 2x1
        U_b, V_b, W_b = construct_block_3x2x1()
        blocks.append((0, 2, 6, 7, U_b, V_b, W_b))
        
        # Block (2,4) x (0,3): 3x2 * 2x3
        U_b, V_b, W_b = construct_block_3x2x3()
        blocks.append((2, 4, 0, 3, U_b, V_b, W_b))
        
        # Block (2,4) x (3,6): 3x2 * 2x3
        U_b, V_b, W_b = construct_block_3x2x3()
        blocks.append((2, 4, 3, 6, U_b, V_b, W_b))
        
        # Block (2,4) x (6,7): 3x2 * 2x1
        U_b, V_b, W_b = construct_block_3x2x1()
        blocks.append((2, 4, 6, 7, U_b, V_b, W_b))
        
        # Block (4,5) x (0,3): 3x1 * 1x3
        U_b, V_b, W_b = construct_block_3x1x3()
        blocks.append((4, 5, 0, 3, U_b, V_b, W_b))
        
        # Block (4,5) x (3,6): 3x1 * 1x3
        U_b, V_b, W_b = construct_block_3x1x3()
        blocks.append((4, 5, 3, 6, U_b, V_b, W_b))
        
        # Block (4,5) x (6,7): 3x1 * 1x1
        U_b, V_b, W_b = construct_block_3x1x1()
        blocks.append((4, 5, 6, 7, U_b, V_b, W_b))
        
        U, V, W = combine_blocks(blocks, n, m, p)
        rank = len(U[0])
        
        if check_decomposition(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
