"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import random
import time as _time

def _verify_vectorized(U, V, W, n=2, m=3, p=10):
    """Fast verification using einsum."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    T_computed = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
    T_expected = np.ones((n, m, p))
    return np.allclose(T_computed, T_expected, atol=1e-10)

def _is_valid_entries(U, V, W):
    """Check that all entries are integer or half-integer."""
    for mat in [U, V, W]:
        mat = np.array(mat, dtype=np.float64)
        doubled = 2.0 * mat
        if not np.allclose(doubled, np.round(doubled), atol=1e-9):
            return False
    return True

def _build_strassen55():
    """Build the known rank-55 decomposition."""
    n, m, p = 2, 3, 10
    rank = 55
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    for i in range(n):
        a_idx = i * m + 2
        for k in range(p):
            b_idx = 2 * p + k
            c_idx = k * n + i
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
    
    X00, X01, X10, X11 = 0, 1, 3, 4
    
    for j in range(5):
        Y00 = 2 * j
        Y01 = 2 * j + 1
        Y10 = 10 + 2 * j
        Y11 = 10 + 2 * j + 1
        
        Z00 = 4 * j
        Z10 = 4 * j + 1
        Z01 = 4 * j + 2
        Z11 = 4 * j + 3
        
        U[X00, col] += 1; U[X11, col] += 1
        V[Y00, col] += 1; V[Y11, col] += 1
        W[Z00, col] += 1; W[Z11, col] += 1
        col += 1
        
        U[X10, col] += 1; U[X11, col] += 1
        V[Y00, col] += 1
        W[Z10, col] += 1; W[Z11, col] -= 1
        col += 1
        
        U[X00, col] += 1
        V[Y01, col] += 1; V[Y11, col] -= 1
        W[Z01, col] += 1; W[Z11, col] += 1
        col += 1
        
        U[X11, col] += 1
        V[Y10, col] += 1; V[Y00, col] -= 1
        W[Z00, col] += 1; W[Z10, col] += 1
        col += 1
        
        U[X00, col] += 1; U[X01, col] += 1
        V[Y11, col] += 1
        W[Z00, col] -= 1; W[Z01, col] += 1
        col += 1
        
        U[X10, col] += 1; U[X00, col] -= 1
        V[Y00, col] += 1; V[Y01, col] += 1
        W[Z11, col] += 1
        col += 1
        
        U[X01, col] += 1; U[X11, col] -= 1
        V[Y10, col] += 1; V[Y11, col] += 1
        W[Z00, col] += 1
        col += 1
    
    assert col == rank
    return U, V, W

def _try_eliminate_columns(U, V, W, n=2, m=3, p=10, time_limit=20.0):
    """Try to eliminate redundant columns."""
    start_time = _time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    changed = True
    while changed:
        changed = False
        for col in range(rank):
            if _time.time() - start_time > time_limit:
                break
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            U_test[:, col] = 0
            V_test[:, col] = 0
            W_test[:, col] = 0
            if _verify_vectorized(U_test, V_test, W_test, n, m, p):
                mask = [c for c in range(rank) if c != col]
                U = U[:, mask]
                V = V[:, mask]
                W = W[:, mask]
                rank -= 1
                changed = True
                break
    return U, V, W, rank

def _local_search(U0, V0, W0, n=2, m=3, p=10, time_limit=40.0, max_iter=5000):
    """Local search to find lower rank."""
    start_time = _time.time()
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        if _time.time() - start_time > time_limit:
            break
        n_changes = random.randint(1, 3)
        for _ in range(n_changes):
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                idx = random.randint(0, U.shape[0]-1)
                col = random.randint(0, U.shape[1]-1)
                U[idx, col] += random.choice([-1, 1])
            elif mat == 'V':
                idx = random.randint(0, V.shape[0]-1)
                col = random.randint(0, V.shape[1]-1)
                V[idx, col] += random.choice([-1, 1])
            else:
                idx = random.randint(0, W.shape[0]-1)
                col = random.randint(0, W.shape[1]-1)
                W[idx, col] += random.choice([-1, 1])
        
        if _verify_vectorized(U, V, W, n, m, p) and _is_valid_entries(U, V, W):
            U2, V2, W2, r2 = _try_eliminate_columns(U, V, W, n, m, p, time_limit=5.0)
            if r2 < best_rank:
                best_rank = r2
                best_U, best_V, best_W = U2.copy(), V2.copy(), W2.copy()
                U, V, W = U2.copy(), V2.copy(), W2.copy()
    
    if best_rank < rank:
        return best_U, best_V, best_W, best_rank
    return None

def _structured_search(n=2, m=3, p=10, time_limit=70.0):
    """Structured search for lower-rank decompositions."""
    start_time = _time.time()
    U, V, W = _build_strassen55()
    best_rank = 55
    
    U, V, W, rank = _try_eliminate_columns(U, V, W, n, m, p, time_limit=20.0)
    best_rank = min(best_rank, rank)
    
    if _time.time() - start_time < time_limit - 10:
        result = _local_search(U, V, W, n, m, p, time_limit=time_limit - 20.0, max_iter=3000)
        if result is not None:
            U, V, W, rank = result
            best_rank = rank
    
    return U, V, W, best_rank

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 x 3x10 matrix multiplication."""
    start_time = _time.time()
    deadline = start_time + 120.0
    
    U, V, W = _build_strassen55()
    best_rank = 55
    
    if _time.time() < deadline - 30:
        U, V, W, rank = _structured_search(n=2, m=3, p=10, time_limit=90.0)
        best_rank = rank
    
    if not _verify_vectorized(U, V, W, 2, 3, 10):
        U, V, W = _build_strassen55()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
