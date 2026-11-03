"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 by 3x5 matrix.
    Uses a structured search over known low-rank algorithms and integer/half-integer rounding.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep margin

    # The multiplication tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For standard matrix mult, T[i,j,k] = 1 for all (i,j,k) in the range
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k

    # Schoolbook rank is n*m*p = 45
    # We know from literature that 3x3 matrix multiplication has rank 23 (Strassen-like)
    # For 3x3 * 3x5, we can potentially use rank 23 * 5 = 115? No, that's not right.
    # Actually, for A(3x3) * B(3x5), we can think of it as 5 independent 3x3 multiplications
    # but they share the same A. The tensor is different.
    
    # Let's try a direct approach: use known low-rank decompositions for small matrix multiplication.
    # For 3x3 * 3x3, the minimal rank is 23 (over reals, not proven but best known).
    # For 3x3 * 3x5, we can view it as computing 3x5 output = sum of 5 columns, each is 3x3 * 1x3 column.
    
    # Actually, let's just try to find a valid decomposition with rank < 45 using a search.
    # Given the time constraint, let's try a few known constructions.
    
    # Strategy 1: Try rank 30 (a heuristic reduction)
    # Strategy 2: Try to use the fact that we can factor out some structure
    
    # Let's implement a simple ALS-based approach with random initialization and try to
    # find a low-rank decomposition, then round to integers/half-integers.
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # T[i,j,k] should equal 1
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    max_err = max(max_err, abs(s - 1.0))
        return max_err < 1e-10

    def try_rank(rank, max_iters=200, seed=42):
        """Try to find a rank-r decomposition using ALS."""
        np.random.seed(seed)
        n, m, p = DIMENSIONS
        
        # Initialize
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Target tensor: T[i,j,k] = 1
        # We want sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), U[i*m+j, :] should satisfy
            # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
            # This is: U[i*m+j, :] @ diag(V[j*p+k,:] * W[k*n+i,:]) = 1 for all k
            # Actually, for fixed (i,j), we need:
            # For each k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # Let M_k[r] = V[j*p+k,r] * W[k*n+i,r]
            # Then U[i*m+j,:] @ M_k = 1 for all k
            # This is a linear system: U[i*m+j,:] @ M = 1 (vector of ones of length p)
            # M is p x rank
            
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve U[i*m+j, :] @ M = ones(p)
                    # M.T @ U[i*m+j, :].T = ones(p)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), V[j*p+k, :] should satisfy
            # sum_r V[j*p+k,r] * (U[i*m+j,r] * W[k*n+i,r]) = 1 for all i
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), W[k*n+i, :] should satisfy
            # sum_r W[k*n+i,r] * (U[i*m+j,r] * V[j*p+k,r]) = 1 for all j
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            if verify(U, V, W):
                return U, V, W
        
        return None

    # Try various ranks starting from a lower bound
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But for matrix multiplication, the rank is typically higher.
    # Let's try ranks from 20 to 44
    
    best_result = None
    
    for rank in range(20, 45):
        if time.time() > deadline - 10:
            break
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            result = try_rank(rank, max_iters=100, seed=seed)
            if result is not None:
                U, V, W = result
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded):
                    # Convert to lists
                    U_list = [[float(x) for x in row] for row in U_rounded]
                    V_list = [[float(x) for x in row] for row in V_rounded]
                    W_list = [[float(x) for x in row] for row in W_rounded]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return schoolbook decomposition
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
# EVOLVE-BLOCK-END
