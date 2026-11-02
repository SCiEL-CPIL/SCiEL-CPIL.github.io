"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _schoolbook_decomposition(n, m, p):
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_structured_decomposition(n, m, p, max_rank, deadline):
    """
    Try to find a decomposition of rank <= max_rank using structured approach.
    The key insight: for 2x3 x 3x6, we can exploit the block structure.
    
    The tensor T[i,j,k] = 1 for all i,j,k can be decomposed as:
    T[i,j,k] = sum_c u[i,j,c] * v[j,k,c] * w[k,i,c]
    
    We try small ranks with integer/half-integer entries.
    """
    start_time = time.time()
    
    # Lower bound: rank >= max(n, m, p) = 6, but actually rank >= max(n*p, m*p, n*m) is not right.
    # The tensor rank of matrix multiplication <n x m, m x p> is at least max(n*p, m*p, n*m) / something.
    # Actually, a simple lower bound: rank >= max(n, m, p) since fixing two indices gives a vector.
    # More carefully: the slice rank gives lower bounds.
    
    # Try ranks from a reasonable lower bound up to max_rank
    # Theoretical lower bound for 2x3 x 3x6: at least 6 (since p=6)
    # Actually, we need rank >= max(n, m, p) = 6
    
    lower_bound = max(n, m, p)
    
    for rank in range(lower_bound, max_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Try random structured decompositions
        # Strategy: use the fact that we can group terms
        # For each rank, try multiple random initializations
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random half-integer values
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Random initialization with small values
            for c in range(rank):
                for idx in range(n * m):
                    U[idx, c] = np.random.choice([-1, -0.5, 0, 0.5, 1])
                for idx in range(m * p):
                    V[idx, c] = np.random.choice([-1, -0.5, 0, 0.5, 1])
                for idx in range(p * n):
                    W[idx, c] = np.random.choice([-1, -0.5, 0, 0.5, 1])
            
            # Verify
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Also try a more systematic approach for small ranks
        if rank <= 12:
            # Try to find a decomposition where each "block" has structure
            # For rank r, think of it as r terms
            # Each term contributes to the tensor as outer product of three vectors
            # u_c is in R^{n*m}, v_c is in R^{m*p}, w_c is in R^{p*n}
            
            # Try a greedy/structured approach
            # First, try decompositions where U, V, W have sparse structure
            
            # For now, try a few more structured patterns
            patterns = 0
            while patterns < 20:
                if time.time() > deadline - 5:
                    break
                
                # Pattern: use Kronecker-like structure
                # Divide the rank into groups
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                
                # Try different structured patterns
                pattern_type = patterns % 4
                
                if pattern_type == 0:
                    # Block diagonal structure
                    for c in range(rank):
                        i = c % n
                        j = (c // n) % m
                        k = (c // (n*m)) % p
                        if i < n and j < m and k < p:
                            U[i*m+j, c] = 1
                            V[j*p+k, c] = 1
                            W[k*n+i, c] = 1
                elif pattern_type == 1:
                    # Sparse random
                    for c in range(rank):
                        for idx in range(n * m):
                            if np.random.random() < 0.5:
                                U[idx, c] = np.random.choice([-1, 1])
                        for idx in range(m * p):
                            if np.random.random() < 0.5:
                                V[idx, c] = np.random.choice([-1, 1])
                        for idx in range(p * n):
                            if np.random.random() < 0.5:
                                W[idx, c] = np.random.choice([-1, 1])
                elif pattern_type == 2:
                    # Try to match the schoolbook but with merging
                    # Use rank-1 updates
                    pass
                else:
                    # Random with sparsity
                    for c in range(rank):
                        for idx in range(n * m):
                            if np.random.random() < 0.3:
                                U[idx, c] = np.random.choice([-1, -0.5, 0.5, 1])
                        for idx in range(m * p):
                            if np.random.random() < 0.3:
                                V[idx, c] = np.random.choice([-1, -0.5, 0.5, 1])
                        for idx in range(p * n):
                            if np.random.random() < 0.3:
                                W[idx, c] = np.random.choice([-1, -0.5, 0.5, 1])
                
                if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                patterns += 1
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Get the schoolbook decomposition as baseline
    schoolbook = _schoolbook_decomposition(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Set deadline: leave 10 seconds margin from the 150s limit
    deadline = time.time() + 140
    
    # Try to find a lower rank decomposition
    # Start from rank 6 (theoretical minimum) up to schoolbook rank - 1
    for rank in range(max(n, m, p), best_rank):
        if time.time() > deadline - 10:
            break
        
        result = _try_structured_decomposition(n, m, p, rank, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            break  # Found the lowest possible rank
    
    return best_result
# EVOLVE-BLOCK-END
