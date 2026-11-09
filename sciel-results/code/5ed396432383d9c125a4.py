"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = sum(Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c] for c in range(rank))
                if abs(actual - expected) > tol:
                    return False
    return True

def _als_decompose(max_rank, deadline, seed=42):
    """Try to find a decomposition of given rank using ALS."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    rank = max_rank
    if rank <= 0:
        return None
    
    # Initialize randomly
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(200):
        if time.time() > deadline - 5:
            break
        
        # Update U: for each (i,j), solve linear system
        for i in range(n):
            for j in range(m):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                # This is: U[i*m+j, :] @ (V[j*p:, :] * W[:, i].T).T = ones(p)
                # Let A[c,k] = V[j*p+k, c] * W[k*n+i, c]
                # Then sum_c U[i*m+j, c] * A[c,k] = 1
                # U[i*m+j, :] @ A.T = ones(p)
                A = np.zeros((rank, p))
                for k in range(p):
                    for c in range(rank):
                        A[c, k] = V[j*p+k, c] * W[k*n+i, c]
                # Solve U_row @ A.T = ones(p)
                # This is A @ U_row.T = ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), solve linear system
        for j in range(m):
            for k in range(p):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
                # Let B[c,i] = U[i*m+j, c] * W[k*n+i, c]
                # Then sum_c V[j*p+k, c] * B[c,i] = 1
                B = np.zeros((rank, n))
                for i in range(n):
                    for c in range(rank):
                        B[c, i] = U[i*m+j, c] * W[k*n+i, c]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), solve linear system
        for k in range(p):
            for i in range(n):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
                # Let C[c,j] = U[i*m+j, c] * V[j*p+k, c]
                # Then sum_c W[k*n+i, c] * C[c,j] = 1
                C = np.zeros((rank, m))
                for j in range(m):
                    for c in range(rank):
                        C[c, j] = U[i*m+j, c] * V[j*p+k, c]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    err += (actual - 1.0) ** 2
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-12:
            break
    
    # Check if we found an exact solution
    if best_err < 1e-9:
        # Convert to lists with rounding to nearest half-integer
        U_list = [[round(x * 2) / 2 for x in row] for row in best_U]
        V_list = [[round(x * 2) / 2 for x in row] for row in best_V]
        W_list = [[round(x * 2) / 2 for x in row] for row in best_W]
        
        # Verify the rounded solution
        if _verify(U_list, V_list, W_list, tol=1e-6):
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            # Try with tighter rounding
            U_list = [[round(x) for x in row] for row in best_U]
            V_list = [[round(x) for x in row] for row in best_V]
            W_list = [[round(x) for x in row] for row in best_W]
            if _verify(U_list, V_list, W_list, tol=1e-6):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Start with schoolbook
    best_result = _schoolbook()
    best_rank = sum(1 for row in best_result['U'] for x in row if x != 0)
    # Actually rank is just the number of columns
    best_rank = len(best_result['U'][0])
    
    # Try to find lower rank decompositions
    for target_rank in range(best_rank - 1, 0, -1):
        if time.time() > deadline - 10:
            break
        result = _als_decompose(target_rank, deadline)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
