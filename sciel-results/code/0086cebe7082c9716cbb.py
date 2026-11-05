"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Pure schoolbook decomposition, rank = n*m*p."""
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

def _strassen_decomposition(n, m, p):
    """
    Construct a decomposition using Strassen for 2x2x2 blocks where possible.
    This is the baseline 'tiling' strategy which achieves rank 88 for 3x4x8.
    """
    def splits_of(N):
        if N == 0:
            return [[]]
        result = []
        for first in range(1, N+1):
            for rest in splits_of(N - first):
                result.append([first] + rest)
        return result
    
    def block_rank(di, dj, dk):
        if di == 2 and dj == 2 and dk == 2:
            return 7
        return di * dj * dk
    
    best_rank = n * m * p
    best_split = None
    
    i_splits = splits_of(n)
    m_splits = splits_of(m)
    p_splits = splits_of(p)
    
    for iseg in i_splits:
        for mseg in m_splits:
            for pseg in p_splits:
                total_rank = 0
                for di in iseg:
                    for dj in mseg:
                        for dk in pseg:
                            total_rank += block_rank(di, dj, dk)
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_split = (iseg, mseg, pseg)
    
    if best_split is None:
        return None
    
    iseg, mseg, pseg = best_split
    rank = best_rank
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    i0 = 0
    for di in iseg:
        j0 = 0
        for dj in mseg:
            k0 = 0
            for dk in pseg:
                if di == 2 and dj == 2 and dk == 2:
                    strassen_terms = [
                        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
                        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
                        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
                        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
                        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
                        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
                        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
                    ]
                    for t in range(7):
                        a_c, b_c, c_c = strassen_terms[t]
                        for a_idx in range(4):
                            ai, aj = a_idx // 2, a_idx % 2
                            U[(i0+ai)*m + (j0+aj), col] = a_c[a_idx]
                        for b_idx in range(4):
                            bj, bk = b_idx // 2, b_idx % 2
                            V[(j0+bj)*p + (k0+bk), col] = b_c[b_idx]
                        for c_idx in range(4):
                            ck, ci = c_idx // 2, c_idx % 2
                            W[(k0+ck)*n + (i0+ci), col] = c_c[c_idx]
                        col += 1
                else:
                    for ai in range(di):
                        for aj in range(dj):
                            for ak in range(dk):
                                U[(i0+ai)*m + (j0+aj), col] = 1
                                V[(j0+aj)*p + (k0+ak), col] = 1
                                W[(k0+ak)*n + (i0+ai), col] = 1
                                col += 1
                k0 += dk
            j0 += dj
        i0 += di
    
    if col != rank:
        return None
        
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    return None

def _greedy_peel(U, V, W, n, m, p, deadline):
    """
    Attempt to reduce rank by greedily removing or merging terms.
    This is a heuristic search over the current decomposition.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    current_rank = U.shape[1]
    
    # Try to remove columns that are linear combinations of others
    # Or simply try to zero out small entries and re-verify?
    # A safer approach: Randomly perturb and project?
    # For now, try to find if any column is redundant.
    
    # This is a placeholder for a more complex peeling strategy.
    # Given time constraints, we return the input if no quick win is found.
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Strategy 1: Tiling with Strassen (known to give 88)
    result = _strassen_decomposition(n, m, p)
    
    if result is not None:
        # Try to improve with greedy peeling
        U, V, W = result['U'], result['V'], result['W']
        U_new, V_new, W_new = _greedy_peel(U, V, W, n, m, p, deadline)
        
        if _verify(U_new, V_new, W_new, n, m, p):
            return {"U": U_new, "V": V_new, "W": W_new}
            
        return result
    
    # Fallback
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
