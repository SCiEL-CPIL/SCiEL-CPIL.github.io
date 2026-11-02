"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _validate(U, V, W, n=2, m=3, p=4):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check shapes
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    # Check entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    # Check tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # T[i,j,k] should be 1 for all valid i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _compute_error(U, V, W, n=2, m=3, p=4):
    """Compute the sum of squared errors for a decomposition."""
    rank = U.shape[1]
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err += (s - 1.0)**2
    return err

def _try_structured_search(rank, n=2, m=3, p=4, max_iter=200, seed=None):
    """Try to find a rank-rank decomposition using structured local search."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Initialize with schoolbook-like structure then try to merge
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Random initialization
    for c in range(rank):
        for idx in range(n*m):
            U[idx, c] = random.choice(values)
        for idx in range(m*p):
            V[idx, c] = random.choice(values)
        for idx in range(p*n):
            W[idx, c] = random.choice(values)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        err = _compute_error(U, V, W, n, m, p)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-12:
                break
        
        # Try multiple random perturbations and keep the best
        best_candidate_err = err
        best_candidate_U, best_candidate_V, best_candidate_W = U.copy(), V.copy(), W.copy()
        
        for _ in range(10):
            # Perturb one element in one matrix
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                row = random.randint(0, n*m - 1)
                col = random.randint(0, rank - 1)
                U[row, col] = random.choice(values)
            elif mat_idx == 1:
                row = random.randint(0, m*p - 1)
                col = random.randint(0, rank - 1)
                V[row, col] = random.choice(values)
            else:
                row = random.randint(0, p*n - 1)
                col = random.randint(0, rank - 1)
                W[row, col] = random.choice(values)
            
            cand_err = _compute_error(U, V, W, n, m, p)
            if cand_err < best_candidate_err:
                best_candidate_err = cand_err
                best_candidate_U, best_candidate_V, best_candidate_W = U.copy(), V.copy(), W.copy()
        
        U, V, W = best_candidate_U, best_candidate_V, best_candidate_W
    
    if _validate(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    
    # Schoolbook decomposition as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    for rank in range(20, schoolbook_rank):  # Try ranks 20-23
        if time.time() > deadline:
            break
        
        attempts = 0
        max_attempts = 100
        while time.time() < deadline - 5 and attempts < max_attempts:
            result = _try_structured_search(rank, n, m, p, max_iter=50, seed=attempts * 1000 + rank)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
            attempts += 1
    
    return best_result
# EVOLVE-BLOCK-END
