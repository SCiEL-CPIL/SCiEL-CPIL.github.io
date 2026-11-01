import numpy as np
import time
import random

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
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

def _random_decomposition(n, m, p, rank, scale=1.0):
    """Generate a random tensor decomposition."""
    U = np.random.randn(n * m, rank) * scale
    V = np.random.randn(m * p, rank) * scale
    W = np.random.randn(p * n, rank) * scale
    return U, V, W

def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors."""
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                error += (s - 1.0) ** 2
    return error

def _local_search(n, m, p, rank, max_iter=1000, lr=0.01, deadline=None):
    """Perform local search on a random initialization."""
    U, V, W = _random_decomposition(n, m, p, rank, scale=1.0)
    best_err = _compute_error(U, V, W, n, m, p)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Compute gradients numerically or use simple random perturbation
        # For speed, we use a simple coordinate descent or random gradient step
        # Let's try a simple random perturbation approach (hill climbing)
        
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        # Perturb one element at a time or small batch
        # To save time, we'll perturb a few random elements
        for _ in range(3):
            mat = random.choice([U_new, V_new, W_new])
            i, j = random.randint(0, mat.shape[0]-1), random.randint(0, mat.shape[1]-1)
            mat[i, j] += random.gauss(0, lr)
            
        err = _compute_error(U_new, V_new, W_new, n, m, p)
        if err < best_err:
            U, V, W = U_new, V_new, W_new
            best_err = err
            lr *= 0.99
        else:
            lr *= 1.05
            lr = min(lr, 0.1)

    return U, V, W, best_err

def _strassen_2x2x4():
    """
    Construct a rank-14 tensor decomposition for 2x2 * 2x4 matrix multiplication.
    Two independent Strassen blocks exploiting the shared A matrix structure.
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: output columns k in {0,1}, rank indices 0..6
    # Strassen for 2x2 * 2x2: A is 2x2, B1 is 2x2 (cols 0,1 of B)
    # U rows: [A00, A01, A10, A11] (indices 0,1,2,3)
    # V rows for block 1: [B00, B01, B10, B11] (indices 0,1,2,3)
    # W rows for block 1: [C00, C01, C10, C11] (indices 0,1,2,3)
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)(B10) -> C10, C11
    U[2][1] = 1; U[3][1] = 1
    V[2][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = (A00)(B10-B11) -> C00, C01
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = 1
    
    # M4 = (A11)(B01-B00) -> C00, C10
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)(B11) -> C01, C11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = -1
    
    # M6 = (A10-A00)(B00+B01) -> C10, C00
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[0][5] = -1
    
    # M7 = (A01-A11)(B10+B11) -> C01, C10
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[2][6] = -1
    
    # Block 2: output columns k in {2,3}, rank indices 7..13
    # V rows for block 2: [B02, B03, B12, B13] (indices 4,5,6,7)
    # W rows for block 2: [C02, C03, C12, C13] (indices 4,5,6,7)
    
    # M8 = (A00+A11)(B02+B13) -> C02, C13
    U[0][7] = 1; U[3][7] = 1
    V[4][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # M9 = (A10+A11)(B12) -> C12, C13
    U[2][8] = 1; U[3][8] = 1
    V[6][8] = 1
    W[6][8] = 1; W[7][8] = -1
    
    # M10 = (A00)(B12-B13) -> C02, C03
    U[0][9] = 1
    V[6][9] = 1; V[7][9] = -1
    W[4][9] = 1; W[5][9] = 1
    
    # M11 = (A11)(B03-B02) -> C02, C12
    U[3][10] = 1
    V[5][10] = 1; V[4][10] = -1
    W[4][10] = 1; W[6][10] = 1
    
    # M12 = (A00+A01)(B13) -> C03, C13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[5][11] = 1; W[7][11] = -1
    
    # M13 = (A10-A00)(B02+B03) -> C12, C02
    U[2][12] = 1; U[0][12] = -1
    V[4][12] = 1; V[5][12] = 1
    W[6][12] = 1; W[4][12] = -1
    
    # M14 = (A01-A11)(B12+B13) -> C03, C12
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[5][13] = 1; W[6][13] = -1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 2, 2, 4
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # First, get the known rank-14 solution
    sol_rank14 = _strassen_2x2x4()
    best_solution = sol_rank14
    best_rank = 14
    
    # Try to find a rank-13 solution via local search
    # We will run several random restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Try rank 13
        U, V, W, err = _local_search(n, m, p, 13, max_iter=500, lr=0.05, deadline=deadline)
        
        # Check if error is close to zero
        if err < 1e-6:
            # Verify with higher precision
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
                best_solution = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = 13
                break
            else:
                # Try to refine if it's close
                pass

    # If we found a rank-13 solution, return it
    if best_rank < 14:
        return best_solution
    
    # Otherwise, return the rank-14 solution
    return sol_rank14
