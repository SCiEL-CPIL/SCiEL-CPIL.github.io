import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _verify(U, V, W, n, m, p):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _compute_error(U, V, W, n, m, p):
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err += (val - 1.0)**2
    return err

def _round_to_halfint(x):
    return 2.0 * np.round(x / 2.0) / 2.0

def _als_solve_U(V, W, n, m, p, rank):
    U = np.zeros((n*m, rank))
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, rank))
            for k in range(p):
                A[k, :] = V[j*p+k, :] * W[k*n+i, :]
            try:
                ATA = A @ A.T
                ATA += 1e-12 * np.eye(p)
                sol = np.linalg.solve(ATA, np.ones(p))
                U[i*m+j, :] = A.T @ sol
            except:
                U[i*m+j, :] = 1.0 / rank
    return U

def _als_solve_V(U, W, n, m, p, rank):
    V = np.zeros((m*p, rank))
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, rank))
            for i in range(n):
                A[i, :] = U[i*m+j, :] * W[k*n+i, :]
            try:
                ATA = A @ A.T
                ATA += 1e-12 * np.eye(n)
                sol = np.linalg.solve(ATA, np.ones(n))
                V[j*p+k, :] = A.T @ sol
            except:
                V[j*p+k, :] = 1.0 / rank
    return V

def _als_solve_W(U, V, n, m, p, rank):
    W = np.zeros((p*n, rank))
    for k in range(p):
        for i in range(n):
            A = np.zeros((m, rank))
            for j in range(m):
                A[j, :] = U[i*m+j, :] * V[j*p+k, :]
            try:
                ATA = A @ A.T
                ATA += 1e-12 * np.eye(m)
                sol = np.linalg.solve(ATA, np.ones(m))
                W[k*n+i, :] = A.T @ sol
            except:
                W[k*n+i, :] = 1.0 / rank
    return W

def _run_als(n, m, p, rank, max_iter=150, tol=1e-14, seed=0):
    np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    prev_err = float('inf')
    for it in range(max_iter):
        U = _als_solve_U(V, W, n, m, p, rank)
        V = _als_solve_V(U, W, n, m, p, rank)
        W = _als_solve_W(U, V, n, m, p, rank)
        
        err = _compute_error(U, V, W, n, m, p)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, err

def _local_search(U, V, W, n, m, p, rank, max_iters=100):
    """Local search on half-integer lattice."""
    U = _round_to_halfint(U).copy()
    V = _round_to_halfint(V).copy()
    W = _round_to_halfint(W).copy()
    
    current_err = _compute_error(U, V, W, n, m, p)
    
    for it in range(max_iters):
        improved = False
        
        # Try perturbing U
        for i in range(n*m):
            for c in range(rank):
                for delta in [-0.5, 0.5]:
                    old = U[i, c]
                    U[i, c] = old + delta
                    new_err = _compute_error(U, V, W, n, m, p)
                    if new_err < current_err:
                        current_err = new_err
                        improved = True
                    else:
                        U[i, c] = old
        
        # Try perturbing V
        for i in range(m*p):
            for c in range(rank):
                for delta in [-0.5, 0.5]:
                    old = V[i, c]
                    V[i, c] = old + delta
                    new_err = _compute_error(U, V, W, n, m, p)
                    if new_err < current_err:
                        current_err = new_err
                        improved = True
                    else:
                        V[i, c] = old
        
        # Try perturbing W
        for i in range(p*n):
            for c in range(rank):
                for delta in [-0.5, 0.5]:
                    old = W[i, c]
                    W[i, c] = old + delta
                    new_err = _compute_error(U, V, W, n, m, p)
                    if new_err < current_err:
                        current_err = new_err
                        improved = True
                    else:
                        W[i, c] = old
        
        if not improved:
            break
    
    if _verify(U, V, W, n, m, p):
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    best_rank = n * m * p
    best_result = None
    
    # Schoolbook
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    
    # Try ALS + local search for lower ranks
    for rank in range(79, 19, -1):
        if time.time() > deadline - 60:
            break
        
        found = False
        for seed in range(8):
            if time.time() > deadline - 60:
                break
            
            U, V, W, err = _run_als(n, m, p, rank, max_iter=100, tol=1e-14, seed=seed)
            
            if err < 1e-8:
                result = _local_search(U, V, W, n, m, p, rank, max_iters=50)
                if result is not None:
                    best_rank = rank
                    best_result = result
                    found = True
                    break
        
        if found:
            continue
    
    U_list = [[float(x) for x in row] for row in best_result[0]]
    V_list = [[float(x) for x in row] for row in best_result[1]]
    W_list = [[float(x) for x in row] for row in best_result[2]]
    
    return {"U": U_list, "V": V_list, "W": W_list}
