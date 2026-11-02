"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check reconstruction: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
    # if (i,j,k) corresponds to a multiplication term, else 0
    # Actually, the tensor T[i,j,k] = 1 for all valid (i,j,k) in the matmul tensor
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k])
                # W[k*n+i] should make this equal to 1
                # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                recon = 0.0
                for c in range(rank):
                    recon += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(recon - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _als_decompose(n, m, p, rank, max_iter=500, l1_lambda=0.01, tol=1e-10):
    """
    Use Alternating Least Squares with L1 regularization to find a rank-rank
    decomposition of the matrix multiplication tensor.
    
    The tensor T has entries T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p].
    We seek U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    """
    np.random.seed(42)
    
    # Initialize with random small values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        prev_loss = best_loss
        
        # Update U: for each row i*m+j, solve least squares
        for i in range(n):
            for j in range(m):
                # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # Let A[k,c] = V[j*p+k,c] * W[k*n+i,c]
                # Then U[i*m+j,:] @ A.T = ones(p)
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve: U_row @ A.T = ones(p), i.e., A @ U_row.T = ones(p)
                # With L1 regularization: minimize ||A @ u - 1||^2 + l1_lambda * ||u||_1
                # Use a simple approach: add small ridge for stability
                reg = 1e-8 * np.eye(rank)
                u, _, _, _ = np.linalg.lstsq(A + reg, np.ones(p), rcond=None)
                U[i * m + j, :] = u
        
        # Update V: for each row j*p+k, solve least squares
        for j in range(m):
            for k in range(p):
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                # Let B[i,c] = U[i*m+j,c] * W[k*n+i,c]
                # Then V[j*p+k,:] @ B.T = ones(n)
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                reg = 1e-8 * np.eye(rank)
                v, _, _, _ = np.linalg.lstsq(B + reg, np.ones(n), rcond=None)
                V[j * p + k, :] = v
        
        # Update W: for each row k*n+i, solve least squares
        for k in range(p):
            for i in range(n):
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                # Let C[j,c] = U[i*m+j,c] * V[j*p+k,c]
                # Then W[k*n+i,:] @ C.T = ones(m)
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                reg = 1e-8 * np.eye(rank)
                w, _, _, _ = np.linalg.lstsq(C + reg, np.ones(m), rcond=None)
                W[k * n + i, :] = w
        
        # Compute loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    loss += (recon - 1.0) ** 2
        
        # L1 penalty
        l1_penalty = l1_lambda * (np.abs(U).sum() + np.abs(V).sum() + np.abs(W).sum())
        total_loss = loss + l1_penalty
        
        if total_loss < best_loss:
            best_loss = total_loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if abs(prev_loss - total_loss) < tol:
            break
    
    return best_U, best_V, best_W, best_loss

def _try_rank(n, m, p, rank, time_budget):
    """Try to find a valid decomposition of given rank."""
    start_time = time.time()
    max_iter = 200
    
    # Try multiple random initializations
    for seed in range(10):
        if time.time() - start_time > time_budget:
            break
        
        np.random.seed(seed)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iter):
            if time.time() - start_time > time_budget:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    reg = 1e-8 * np.eye(rank)
                    u, _, _, _ = np.linalg.lstsq(A + reg, np.ones(p), rcond=None)
                    U[i * m + j, :] = u
            
            # Update V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, rank))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    reg = 1e-8 * np.eye(rank)
                    v, _, _, _ = np.linalg.lstsq(B + reg, np.ones(n), rcond=None)
                    V[j * p + k, :] = v
            
            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, rank))
                    for j in range(m):
                        C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    reg = 1e-8 * np.eye(rank)
                    w, _, _, _ = np.linalg.lstsq(C + reg, np.ones(m), rcond=None)
                    W[k * n + i, :] = w
            
            # Compute loss
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        loss += (recon - 1.0) ** 2
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-12:
                break
        
        if best_loss < 1e-8:
            # Found a valid solution!
            # Convert to list of lists with rounding
            U_list = [[round(float(x), 6) for x in row] for row in best_U]
            V_list = [[round(float(x), 6) for x in row] for row in best_V]
            W_list = [[round(float(x), 6) for x in row] for row in best_W]
            
            # Verify
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_result = _schoolbook(n, m, p)
    
    start_time = time.time()
    deadline = 130  # Leave 20s margin
    
    # Try ranks from lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Upper bound: n*m*p = 90
    min_rank = max(n * m, m * p, p * n)
    max_rank = n * m * p
    
    for rank in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break
        
        time_budget = min(10, deadline - time.time())
        if time_budget < 2:
            break
        
        result = _try_rank(n, m, p, rank, time_budget)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook_result
# EVOLVE-BLOCK-END
