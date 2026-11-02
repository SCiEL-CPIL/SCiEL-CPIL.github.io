import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition"""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact"""
    rng = np.random.RandomState(42)
    for _ in range(10):
        A = rng.randint(-5, 6, size=(n, m))
        B = rng.randint(-5, 6, size=(m, p))
        C_expected = A @ B
        
        C_reconstructed = np.zeros((n, p), dtype=float)
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for r in range(U.shape[1]):
                        total += A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_reconstructed[i, k] = total
        
        if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
            return False
    return True

def _greedy_sparse_decomposition(n, m, p, max_rank, deadline, rng):
    """
    Greedy sparse linear regression to find low-rank decomposition.
    """
    start_time = time.time()
    
    residual = np.ones((n, m, p), dtype=float)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    candidate_vals = [-2, -1, 0, 1, 2]
    
    current_rank = 0
    while current_rank < max_rank:
        if time.time() > deadline - 5:
            break
        
        best_score = -np.inf
        best_u, best_v, best_w = None, None, None
        
        n_attempts = 300
        for attempt in range(n_attempts):
            if time.time() > deadline - 5:
                break
            
            u = rng.choice(candidate_vals, size=n*m)
            v = rng.choice(candidate_vals, size=m*p)
            w = rng.choice(candidate_vals, size=p*n)
            
            # Compute score: <residual, u⊗v⊗w>
            score = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        score += residual[i, j, k] * u[i*m+j] * v[j*p+k] * w[k*n+i]
            
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        if best_u is None:
            break
        
        U_cols.append(best_u)
        V_cols.append(best_v)
        W_cols.append(best_w)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    residual[i, j, k] -= best_u[i*m+j] * best_v[j*p+k] * best_w[k*n+i]
        
        current_rank += 1
        
        if np.max(np.abs(residual)) < 1e-10:
            break
    
    if current_rank == 0:
        return None
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    return U, V, W, current_rank

def _local_search_improve(U, V, W, n, m, p, deadline, rng):
    """
    Try to improve the decomposition by removing terms or replacing them.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    for col in range(current_rank - 1, -1, -1):
        if time.time() > deadline - 10:
            break
        
        U_test = np.delete(U, col, axis=1)
        V_test = np.delete(V, col, axis=1)
        W_test = np.delete(W, axis=1)
        
        if _verify_decomposition(U_test, V_test, W_test, n, m, p):
            U, V, W = U_test, V_test, W_test
            current_rank -= 1
    
    return U, V, W, current_rank

def construct():
    start = time.time()
    deadline = start + 140.0
    
    n, m, p = DIMENSIONS
    
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p
    
    rng = np.random.RandomState(123)
    
    # Try to find a better decomposition
    for max_rank in [35, 33, 30, 28, 26]:
        if time.time() > deadline - 20:
            break
        
        result = _greedy_sparse_decomposition(n, m, p, max_rank, deadline, rng)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                U, V, W, rank = _local_search_improve(U, V, W, n, m, p, deadline, rng)
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
    
    return best
