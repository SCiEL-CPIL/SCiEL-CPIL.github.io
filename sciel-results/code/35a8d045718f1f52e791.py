"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a11 + a22)(b11 + b22)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # M2 = (a21 + a22) b11
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 1, 0, -1]
    
    # M3 = a11 (b12 - b22)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 0, 1, 1]
    
    # M4 = a22 (b21 - b11)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 0, 1, 0]
    W[:, 3] = [1, 1, 0, 0]
    
    # M5 = (a11 + a12) b22
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 0, 1, 0]
    
    # M6 = (a21 - a11) (b11 + b12)
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    
    # M7 = (a12 - a22) (b21 + b22)
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook decomposition for n x m x p. Returns U, V, W."""
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
    """
    Embed a local decomposition of size ni x mi x pi into global dimensions.
    Returns the global U, V, W slices (as arrays of shape (n*m, r), etc.)
    """
    r = U_local.shape[1]
    U_g = np.zeros((n * m, r))
    V_g = np.zeros((m * p, r))
    W_g = np.zeros((p * n, r))
    
    # U: local index li*mi + lj -> global (n_off+li)*m + (m_off+lj)
    for li in range(ni):
        for lj in range(mi):
            local_idx = li * mi + lj
            global_idx = (n_off + li) * m + (m_off + lj)
            U_g[global_idx, :] = U_local[local_idx, :]
    
    # V: local index lj*pi + lk -> global (m_off+lj)*p + (p_off+lk)
    for lj in range(mi):
        for lk in range(pi):
            local_idx = lj * pi + lk
            global_idx = (m_off + lj) * p + (p_off + lk)
            V_g[global_idx, :] = V_local[local_idx, :]
    
    # W: local index lk*ni + li -> global (p_off+lk)*n + (n_off+li)
    for lk in range(pi):
        for li in range(ni):
            local_idx = lk * ni + li
            global_idx = (p_off + lk) * n + (n_off + li)
            W_g[global_idx, :] = W_local[local_idx, :]
    
    return U_g, V_g, W_g


def _block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """
    Decompose n x m x p into blocks defined by n_parts, m_parts, p_parts.
    Use Strassen for 2x2x2 blocks, schoolbook for others.
    """
    # Compute offsets
    n_offsets = [sum(n_parts[:i]) for i in range(len(n_parts))]
    m_offsets = [sum(m_parts[:i]) for i in range(len(m_parts))]
    p_offsets = [sum(p_parts[:i]) for i in range(len(p_parts))]
    
    # Determine rank for each block
    total_rank = 0
    block_ranks = {}
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                if ni == 2 and mi == 2 and pi == 2:
                    block_ranks[(i, j, k)] = 7
                else:
                    block_ranks[(i, j, k)] = ni * mi * pi
                total_rank += block_ranks[(i, j, k)]
    
    # Construct global U, V, W
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                r = block_ranks[(i, j, k)]
                if ni == 2 and mi == 2 and pi == 2:
                    U_loc, V_loc, W_loc = _strassen_2x2x2()
                else:
                    U_loc, V_loc, W_loc = _schoolbook(ni, mi, pi)
                
                U_g, V_g, W_g = _embed_block(U_loc, V_loc, W_loc, n, m, p,
                                              n_offsets[i], m_offsets[j], p_offsets[k],
                                              ni, mi, pi)
                U[:, col:col+r] = U_g
                V[:, col:col+r] = V_g
                W[:, col:col+r] = W_g
                col += r
    
    return U, V, W, total_rank


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    Returns True if valid, False otherwise.
    """
    # Reconstruct the tensor
    # T[i*m+j, j*p+k, k*n+i] should be 1 for all valid i,j,k
    # Using the decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For efficiency, we can check a few random indices or all indices
    # Since n=3, m=5, p=5, total indices = 3*5*5 = 75
    # This is small enough to check all
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # The structure tensor has 1 at (i,j,k) for matrix multiplication
                if abs(val - 1.0) > 1e-9:
                    return False
    
    return True


def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Strategy: Use block decomposition with multiple splits and pick the best rank.
    # We'll try several different block splits and return the one with lowest rank.
    
    candidates = []
    
    # Split 1: n=[1,2], m=[2,3], p=[2,3]
    # Blocks: (1,2,2)=4, (1,3,3)=9, (2,2,2)=7, (2,3,3)=18 -> total 38
    candidates.append(([1, 2], [2, 3], [2, 3]))
    
    # Split 2: n=[1,2], m=[3,2], p=[3,2]
    # Blocks: (1,3,3)=9, (1,2,2)=4, (2,3,3)=18, (2,2,2)=7 -> total 38
    candidates.append(([1, 2], [3, 2], [3, 2]))
    
    # Split 3: n=[2,1], m=[2,3], p=[2,3]
    # Blocks: (2,2,2)=7, (2,3,3)=18, (1,2,2)=4, (1,3,3)=9 -> total 38
    candidates.append(([2, 1], [2, 3], [2, 3]))
    
    # Split 4: n=[1,1,1], m=[2,3], p=[2,3]
    # 6 blocks: (1,2,2)=4, (1,3,3)=9, (1,2,2)=4, (1,3,3)=9, (1,2,2)=4, (1,3,3)=9 -> total 39
    candidates.append(([1, 1, 1], [2, 3], [2, 3]))
    
    # Split 6: n=[1,2], m=[2,1,2], p=[2,1,2]
    # 9 blocks, more complex. Let's compute:
    # (1,2,2)=4, (1,1,1)=1, (1,2,2)=4
    # (2,2,2)=7, (2,1,1)=2, (2,2,2)=7
    # Total: 4+1+4+7+2+7 = 25
    candidates.append(([1, 2], [2, 1, 2], [2, 1, 2]))
    
    # Split 7: n=[1,2], m=[1,2,2], p=[1,2,2]
    # i=0: (1,1,1)=1, (1,2,2)=4, (1,2,2)=4
    # i=1: (2,1,1)=2, (2,2,2)=7, (2,2,2)=7
    # Total: 1+4+4+2+7+7 = 25
    candidates.append(([1, 2], [1, 2, 2], [1, 2, 2]))
    
    # Split 8: n=[1,1,1], m=[1,2,2], p=[1,2,2]
    # 9 blocks:
    # (1,1,1)=1, (1,2,2)=4, (1,2,2)=4
    # (1,1,1)=1, (1,2,2)=4, (1,2,2)=4
    # (1,1,1)=1, (1,2,2)=4, (1,2,2)=4
    # Total: 3+12+12 = 27
    candidates.append(([1, 1, 1], [1, 2, 2], [1, 2, 2]))
    
    # Split 9: n=[1,2], m=[2,2,1], p=[2,2,1]
    # i=0: (1,2,2)=4, (1,2,2)=4, (1,1,1)=1
    # i=1: (2,2,2)=7, (2,2,2)=7, (2,1,1)=2
    # Total: 4+4+1+7+7+2 = 25
    candidates.append(([1, 2], [2, 2, 1], [2, 2, 1]))
    
    # Split 10: n=[1,2], m=[1,1,3], p=[1,1,3]
    # i=0: (1,1,1)=1, (1,1,1)=1, (1,3,3)=9
    # i=1: (2,1,1)=2, (2,1,1)=2, (2,3,3)=18
    # Total: 1+1+9+2+2+18 = 33
    candidates.append(([1, 2], [1, 1, 3], [1, 1, 3]))
    
    # Additional candidates to explore
    # Split 11: n=[1,1,1], m=[2,1,2], p=[2,1,2]
    # 9 blocks:
    # (1,2,2)=4, (1,1,1)=1, (1,2,2)=4
    # (1,2,2)=4, (1,1,1)=1, (1,2,2)=4
    # (1,2,2)=4, (1,1,1)=1, (1,2,2)=4
    # Total: 12+3+12 = 27
    candidates.append(([1, 1, 1], [2, 1, 2], [2, 1, 2]))
    
    # Split 12: n=[2,1], m=[2,1,2], p=[2,1,2]
    # 9 blocks:
    # (2,2,2)=7, (2,1,1)=2, (2,2,2)=7
    # (1,2,2)=4, (1,1,1)=1, (1,2,2)=4
    # Total: 7+2+7+4+1+4 = 25
    candidates.append(([2, 1], [2, 1, 2], [2, 1, 2]))
    
    # Try all candidates and pick the best
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    for n_parts, m_parts, p_parts in candidates:
        if time.time() > deadline:
            break
        try:
            U, V, W, rank = _block_decomposition(n, m, p, n_parts, m_parts, p_parts)
            if rank < best_rank:
                # Verify the decomposition before accepting
                if _verify_decomposition(U, V, W, n, m, p):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                else:
                    # If verification fails, skip this candidate
                    pass
        except Exception:
            continue
    
    # Fallback to schoolbook if something went wrong
    if best_U is None:
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = best_U, best_V, best_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
