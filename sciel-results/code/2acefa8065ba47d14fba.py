"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 if j==j (always true) and the indices match, else 0
    # Actually for matrix multiplication: T[i,j,k] should be 1 if the (i,j) of A and (j,k) of B
    # contribute to C[i,k], which is always the case for valid j.
    # The tensor for matrix multiplication has T[i,j,k] = 1 for all valid (i,j,k).
    # Wait, let me reconsider. The matrix multiplication tensor T_{ijk} = delta_{j,j} which is 1.
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-10:
                    return False, (i, j, k, s)
    return True, None

def _construct_rank_attempt(n, m, p, target_rank, max_time=100):
    """Try to find a decomposition of the given target rank."""
    start_time = time.time()
    deadline = start_time + max_time
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(n * m, target_rank)
    V = np.random.randn(m * p, target_rank)
    W = np.random.randn(p * n, target_rank)
    
    # The tensor we want: T[i,j,k] = 1 for all i,j,k
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Use alternating least squares / gradient descent approach
    # For each factor, fix the others and solve for the current one
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        # Update U: for each (i,j), solve for U[i*m+j, :]
        for i in range(n):
            for j in range(m):
                # We need sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k
                # Let A_k = V[j*p+k, :] * W[k*n+i, :] (element-wise product)
                # We need U[i*m+j, :] dot A_k = 1 for all k
                # This is a linear system: A @ U[i*m+j, :]^T = 1
                A = np.zeros((p, target_rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A @ x = ones(p)
                try:
                    x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m+j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k, :]
        for j in range(m):
            for k in range(p):
                # We need sum_c V[j*p+k, c] * (U[i*m+j, c] * W[k*n+i, c]) = 1 for all i
                A = np.zeros((n, target_rank))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[j*p+k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i, :]
        for k in range(p):
            for i in range(n):
                # We need sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1 for all j
                A = np.zeros((m, target_rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k*n+i, :] = x
                except:
                    pass
        
        # Compute loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    loss += (s - 1.0) ** 2
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            break
    
    # Check if we found a valid solution
    if best_loss < 1e-10:
        # Convert to lists with half-integer check
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        
        # Verify
        valid, err = _verify_decomposition(U_list, V_list, W_list, n, m, p)
        if valid:
            return U_list, V_list, W_list
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) for some cases
    # For 2x5x6: n*m=10, m*p=30, p*n=12, so lower bound is at least 30
    
    # Try ranks from a reasonable lower bound up to schoolbook
    schoolbook_rank = n * m * p  # 60
    
    # Try a range of ranks
    for target_rank in range(30, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        
        result = _construct_rank_attempt(n, m, p, target_rank, max_time=10)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END

