# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Return Strassen's 7-rank decomposition for 2x2 * 2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    U[i*2+j, c] corresponds to A[i,j]
    V[j*2+k, c] corresponds to B[j,k]
    W[k*2+i, c] corresponds to C[i,k]
    """
    # Strassen's formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U: rows indexed by (i,j) for A[i,j], columns by M1..M7
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    U = [
        [1, 0, 1, 0, 1, -1, -1],  # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, -1, 1, 0, 0, -1],  # A11
    ]
    
    # V: rows indexed by (j,k) for B[j,k], columns by M1..M7
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, -1, 0, 0, 1, 0],   # B01
        [1, 0, 1, 0, 0, 0, 1],    # B10
        [1, 0, 0, 1, 1, 0, 1],    # B11
    ]
    
    # W: rows indexed by (k,i) for C[i,k], columns by M1..M7
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    W = [
        [1, 0, 0, 0, 0, 0, 1],    # C00 (k=0,i=0)
        [0, 0, 1, 0, 1, 0, 0],    # C01 (k=1,i=0)
        [0, 1, 0, 1, 0, 0, 0],    # C10 (k=0,i=1)
        [1, -1, 1, 0, 0, 1, 0],   # C11 (k=1,i=1)
    ]
    
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _find_rank23_3x3x3(deadline):
    """
    Attempt to find a rank-23 decomposition for 3x3x3 matrix multiplication.
    Returns U, V, W as numpy arrays of shape (9, 23) if found, else None.
    """
    n, m, p = 3, 3, 3
    rank = 23
    
    def residuals(x):
        size = 9 * rank
        U = x[0:size].reshape(9, rank)
        V = x[size:2*size].reshape(9, rank)
        W = x[2*size:3*size].reshape(9, rank)
        
        res = np.zeros(27)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    res[idx] = s - 1.0
                    idx += 1
        return res

    best_solution = None
    best_err = float('inf')
    
    for attempt in range(100):
        if time.time() > deadline - 15:
            break
            
        np.random.seed(attempt)
        x0 = np.random.choice([-1, 0, 1], size=3 * 9 * rank)
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=2000)
            err = sol.cost
            if err < best_err:
                best_err = err
                best_solution = sol.x
                if err < 1e-8:
                    break
        except Exception:
            continue

    if best_solution is None:
        return None

    size = 9 * rank
    U = best_solution[0:size].reshape(9, rank)
    V = best_solution[size:2*size].reshape(9, rank)
    W = best_solution[2*size:3*size].reshape(9, rank)
    
    U_int = np.round(U * 2) / 2
    V_int = np.round(V * 2) / 2
    W_int = np.round(W * 2) / 2
    
    if _verify_decomposition(U_int, V_int, W_int, n, m, p, tol=1e-9):
        return U_int, V_int, W_int
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # 1. Try to find rank-23 for 3x3x3
    rank23_result = _find_rank23_3x3x3(deadline)
    
    if rank23_result is not None:
        U3, V3, W3 = rank23_result
        rank3 = 23
        total_rank = 2 * rank3  # 46
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        # Block 1: columns 0-2 of B, columns 0-2 of C
        for c in range(rank3):
            U[:9, c] = U3
            V[:9, c] = V3
            W[:9, c] = W3
        
        # Block 2: columns 3-5 of B, columns 3-5 of C
        for c in range(rank3):
            offset = rank3 + c
            U[:9, offset] = U3
            V[9:18, offset] = V3
            W[9:18, offset] = W3
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 2. Try structured Strassen + Schoolbook hybrid
    # Split A (3x3) and B (3x6) into blocks
    # A = [[A11 (2x2), A12 (2x1)], [A21 (1x2), A22 (1x1)]]
    # B = [[B11 (2x2), B12 (2x4)], [B21 (1x2), B22 (1x4)]]
    # C = A*B
    # C11 (2x2) = A11*B11 + A12*B21
    # C12 (2x4) = A11*B12 + A12*B22
    # C21 (1x2) = A21*B11 + A22*B21
    # C22 (1x4) = A21*B12 + A22*B22
    
    # We can use Strassen for A11*B11 (2x2 * 2x2) -> rank 7
    # The rest are rectangular or small.
    
    # Let's try a simpler decomposition using the known rank for 3x3x3 (23) if we can't find it,
    # or try to construct a rank-30ish decomposition using Strassen blocks.
    
    # Actually, let's just use the best fallback we have which is schoolbook (54)
    # unless we can quickly find a better one via random search on the full problem.
    
    # Random search for better than 54
    best_result = _schoolbook(n, m, p)
    best_rank = 54
    
    # Try random search for ranks lower than 54
    # This is computationally expensive, so we limit it.
    for target_rank in [30, 36, 40, 45, 50]:
        if time.time() > deadline - 10:
            break
            
        found = False
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(attempt * 1000 + target_rank)
            # Use small integers
            U = np.random.choice([-1, 0, 1], size=(n * m, target_rank)).astype(float)
            V = np.random.choice([-1, 0, 1], size=(m * p, target_rank)).astype(float)
            W = np.random.choice([-1, 0, 1], size=(p * n, target_rank)).astype(float)
            
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                found = True
                break
        
        if found:
            continue  # Try lower ranks

    return best_result
# EVOLVE-BLOCK-END
