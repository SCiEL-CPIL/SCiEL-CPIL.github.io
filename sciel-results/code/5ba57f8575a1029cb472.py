"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return 7 rank-1 decompositions for 2x2x2 matrix multiplication."""
    decomps = []
    def add(U, V, W):
        decomps.append((U[:], V[:], W[:]))
    U = [1,0,0,1]; V = [1,0,0,1]; W = [1,0,0,1]; add(U, V, W)
    U = [0,0,1,1]; V = [1,0,0,0]; W = [1,0,0,0]; add(U, V, W)
    U = [1,0,0,0]; V = [0,0,1,-1]; W = [0,0,1,-1]; add(U, V, W)
    U = [0,0,0,1]; V = [-1,1,0,0]; W = [-1,1,0,0]; add(U, V, W)
    U = [1,1,0,0]; V = [0,0,0,1]; W = [0,0,0,1]; add(U, V, W)
    U = [-1,0,1,0]; V = [1,1,0,0]; W = [1,1,0,0]; add(U, V, W)
    U = [0,1,0,-1]; V = [0,0,1,1]; W = [0,0,1,1]; add(U, V, W)
    return decomps

def _strassen_full(N):
    """
    Full Strassen decomposition for N x N x N where N is a power of 2.
    Returns (U, V, W) as numpy arrays of shape (N*N, rank).
    U[i*N+j, c], V[j*N+k, c], W[k*N+i, c]
    """
    if N == 1:
        return np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])
    if N == 2:
        decomp = _strassen_2x2x2()
        rank = len(decomp)
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        for c, (u, v, w) in enumerate(decomp):
            U[:, c] = u
            V[:, c] = v
            W[:, c] = w
        return U, V, W
    
    N1 = N // 2
    U_sub, V_sub, W_sub = _strassen_full(N1)
    rank_sub = U_sub.shape[1]
    
    # 7 Strassen terms
    strassen_2x2 = _strassen_2x2x2()
    
    U_full = np.zeros((N * N, 7 * rank_sub))
    V_full = np.zeros((N * N, 7 * rank_sub))
    W_full = np.zeros((N * N, 7 * rank_sub))
    
    for t, (U_s, V_s, W_s) in enumerate(strassen_2x2):
        for c in range(rank_sub):
            col = t * rank_sub + c
            # U_s has 4 entries for 2x2 A blocks
            for ui in range(4):
                val = U_s[ui]
                if val != 0:
                    bi = ui // 2
                    bj = ui % 2
                    for su in range(N1 * N1):
                        if U_sub[su, c] != 0:
                            si = su // N1
                            sj = su % N1
                            full_idx = (bi * N1 + si) * N + (bj * N1 + sj)
                            U_full[full_idx, col] += val * U_sub[su, c]
            
            for vi in range(4):
                val = V_s[vi]
                if val != 0:
                    bj = vi // 2
                    bk = vi % 2
                    for sv in range(N1 * N1):
                        if V_sub[sv, c] != 0:
                            sj = sv // N1
                            sk = sv % N1
                            full_idx = (bj * N1 + sj) * N + (bk * N1 + sk)
                            V_full[full_idx, col] += val * V_sub[sv, c]
            
            for wi in range(4):
                val = W_s[wi]
                if val != 0:
                    bk = wi // 2
                    bi = wi % 2
                    for sw in range(N1 * N1):
                        if W_sub[sw, c] != 0:
                            sk = sw // N1
                            si = sw % N1
                            full_idx = (bk * N1 + sk) * N + (bi * N1 + si)
                            W_full[full_idx, col] += val * W_sub[sw, c]
    
    return U_full, V_full, W_full

def _next_pow2(x):
    x = max(1, x)
    p2 = 1
    while p2 < x:
        p2 *= 2
    return p2

def _embed_and_extract(n, m, p):
    """
    Embed n x m x p into N x N x N (N = next power of 2 >= max(n,m,p)),
    apply full Strassen, extract sub-tensor, prune zero columns.
    Returns (U, V, W) as lists of lists.
    """
    N = _next_pow2(max(n, m, p))
    
    U_full, V_full, W_full = _strassen_full(N)
    rank_full = U_full.shape[1]
    
    # Extract sub-tensor
    U_sub = np.zeros((n * m, rank_full))
    V_sub = np.zeros((m * p, rank_full))
    W_sub = np.zeros((p * n, rank_full))
    
    for i in range(n):
        for j in range(m):
            U_sub[i * m + j, :] = U_full[i * N + j, :]
    
    for j in range(m):
        for k in range(p):
            V_sub[j * p + k, :] = V_full[j * N + k, :]
    
    for k in range(p):
        for i in range(n):
            W_sub[k * n + i, :] = W_full[k * N + i, :]
    
    # Prune zero columns
    keep_cols = []
    for c in range(rank_full):
        if np.any(U_sub[:, c] != 0) and np.any(V_sub[:, c] != 0) and np.any(W_sub[:, c] != 0):
            keep_cols.append(c)
    
    U_final = U_sub[:, keep_cols].tolist()
    V_final = V_sub[:, keep_cols].tolist()
    W_final = W_sub[:, keep_cols].tolist()
    
    return U_final, V_final, W_final

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                computed = 0.0
                for c in range(rank):
                    computed += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(computed - expected) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try embed-and-extract with Strassen
    try:
        U, V, W = _embed_and_extract(n, m, p)
        rank = len(U[0])
        
        if rank < best_rank:
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
