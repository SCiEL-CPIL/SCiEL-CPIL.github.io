"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _winograd_init_4x4x5(n, m, p, rank, seed=42):
    """Winograd-based initialization for 4x4x5 using 2x2 block structure."""
    np.random.seed(seed)
    
    F2 = np.array([[1, 0, 0, 1],
                   [1, 1, -1, 0],
                   [1, -1, 0, 0],
                   [0, 1, 0, -1]], dtype=float)
    G2 = np.array([[1, 1, 0, 0],
                   [1, -1, 0, 0],
                   [0, 0, 1, 1],
                   [0, 0, 1, -1]], dtype=float)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    base_rank = 16
    if rank < base_rank:
        return _sparse_random_init(n, m, p, rank, seed=seed)
    
    for r in range(base_rank):
        bi = r // 8
        bj = (r // 4) % 2
        bk = (r // 2) % 2
        lr = r % 2
        
        for i in range(n):
            for j in range(m):
                U[i*m+j, r] = F2[i%2, lr] * G2[j%2, lr]
        
        for j in range(m):
            for k in range(4):
                V[j*p+k, r] = G2[j%2, lr] * G2[k%2, lr]
            V[j*p+4, r] = 0.0
        
        for k in range(4):
            for i in range(n):
                W[k*n+i, r] = G2[k%2, lr] * F2[i%2, lr]
        for i in range(n):
            W[4*n+i, r] = 0.0
    
    extra = min(rank - base_rank, 4)
    for r in range(base_rank, base_rank + extra):
        lr = (r - base_rank) % 4
        for i in range(n):
            for j in range(m):
                U[i*m+j, r] = F2[i%2, lr] * 0.5
        for j in range(m):
            V[j*p+4, r] = G2[j%2, lr] * 0.5
        for i in range(n):
            W[4*n+i, r] = F2[i%2, lr] * 0.5
    
    if rank > base_rank + extra:
        for r in range(base_rank + extra, rank):
            for idx in range(n*m):
                if np.random.random() < 0.3:
                    U[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            for idx in range(m*p):
                if np.random.random() < 0.3:
                    V[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            for idx in range(p*n):
                if np.random.random() < 0.3:
                    W[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
    
    return U, V, W

def _sparse_random_init(n, m, p, rank, seed=42, density=0.3):
    """Sparse random initialization with half-integer values."""
    np.random.seed(seed)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    half_ints = [-1.0, -0.5, 0.5, 1.0]
    
    for r in range(rank):
        for idx in range(n*m):
            if np.random.random() < density:
                U[idx, r] = np.random.choice(half_ints)
        for idx in range(m*p):
            if np.random.random() < density:
                V[idx, r] = np.random.choice(half_ints)
        for idx in range(p*n):
            if np.random.random() < density:
                W[idx, r] = np.random.choice(half_ints)
    
    return U, V, W

def _als_pass_fast(U, V, W, n, m, p, rank, max_iters=100, tol=1e-10):
    """Fast ALS to refine U, V, W."""
    prev_err = float('inf')
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iters):
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (s - 1.0) ** 2
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W

def _polish_high(U, V, W, n, m, p, rank, max_iters=200):
    """High-precision polish."""
    return _als_pass_fast(U, V, W, n, m, p, rank, max_iters=max_iters, tol=1e-14)

def _try_rank(rank, n, m, p, seed=42, max_iters=80, polish_iters=100):
    """Try to find a decomposition at given rank."""
    init_methods = [
        _winograd_init_4x4x5,
        _sparse_random_init,
    ]
    
    for init_method in init_methods:
        U, V, W = init_method(n, m, p, rank, seed=seed)
        U, V, W = _als_pass_fast(U, V, W, n, m, p, rank, max_iters=max_iters)
        U, V, W = _polish_high(U, V, W, n, m, p, rank, max_iters=polish_iters)
        
        U_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in U])
        V_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in V])
        W_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in W])
        
        if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    for rank in range(48, 39, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(12):
            if time.time() > deadline - 30:
                break
            
            result = _try_rank(rank, n, m, p, seed=seed, max_iters=60, polish_iters=80)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
