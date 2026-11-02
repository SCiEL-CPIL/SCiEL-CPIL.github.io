"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p  # 48
    
    # Start with schoolbook as fallback
    best_result = None
    best_rank = schoolbook_rank
    
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # Try ranks from 24 up to schoolbook-1
    max_rank_to_try = min(schoolbook_rank - 1, 35)  # Limit search space
    min_rank_to_try = 24  # Theoretical lower bound
    
    for rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        success = _try_find_decomposition(n, m, p, rank)
        if success is not None:
            U, V, W = success
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
            # If we found a low rank, try to go even lower
            break
    
    if best_result is None:
        # Return schoolbook decomposition
        U = [[0] * schoolbook_rank for _ in range(nm)]
        V = [[0] * schoolbook_rank for _ in range(mp)]
        W = [[0] * schoolbook_rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
        best_rank = schoolbook_rank
    
    return best_result

def _try_find_decomposition(n, m, p, rank, max_attempts=50):
    """Try to find an exact rank-`rank` decomposition using structured random search."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Strategy: Use random half-integer matrices and verify
    # For matrix multiplication C = A*B, the tensor T[i,j,k] = 1 for all valid (i,j,k)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    np.random.seed(42)
    
    for attempt in range(max_attempts):
        if time.time() - START_TIME > TIME_LIMIT - 3:
            return None
        
        # Initialize with small random half-integers (multiply by 2 for integer arithmetic)
        scale = 3
        U = np.random.randint(-scale, scale + 1, (nm, rank))
        V = np.random.randint(-scale, scale + 1, (mp, rank))
        W = np.random.randint(-scale, scale + 1, (pn, rank))
        
        # Convert to half-integers
        U = U.astype(float) / 2.0
        V = V.astype(float) / 2.0
        W = W.astype(float) / 2.0
        
        # Check if this random guess works
        if _verify_decomposition(n, m, p, U, V, W):
            return (U.tolist(), V.tolist(), W.tolist())
        
        # Try a more structured initialization based on Strassen-like patterns
        if attempt == 10:
            result = _try_structured_decomposition(n, m, p, rank)
            if result is not None:
                return result
        
        # Try block-based approach
        if attempt == 20:
            result = _try_block_decomposition(n, m, p, rank)
            if result is not None:
                return result
    
    return None

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 for all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j, :]
                v_val = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                result = np.sum(u_val * v_val * w_val)
                if abs(result - 1.0) > 1e-9:
                    return False
    
    # Check half-integer constraint
    for arr in [U, V, W]:
        for row in arr:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    
    return True

def _try_structured_decomposition(n, m, p, rank):
    """Try a structured approach based on known low-rank constructions."""
    # For 2x4 * 4x6, we can try to exploit the small n=2 dimension
    # This is a placeholder for more sophisticated methods
    return None

def _try_block_decomposition(n, m, p, rank):
    """Try block-based decomposition exploiting the 2x4 * 4x6 structure."""
    # For C = A * B where A is 2x4 and B is 4x6:
    # C[0,:] = A[0,:] * B and C[1,:] = A[1,:] * B
    # We can write C[0,:] = sum_j A[0,j] * B[j,:]
    # C[1,:] = sum_j A[1,j] * B[j,:]
    
    # Try to find a decomposition where we group terms
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Initialize with zeros
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Try to fill in a pattern that might work
    # For rank 24 (lower bound), we might be able to achieve this
    # with a structured approach
    
    # This is a heuristic - try various patterns
    for pattern in range(5):
        if time.time() - START_TIME > TIME_LIMIT - 3:
            return None
        
        # Different initialization patterns
        if pattern == 0:
            # Diagonal-like pattern
            for r in range(min(rank, nm)):
                i, j = divmod(r, m)
                U[i * m + j, r] = 1
                for k in range(p):
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        elif pattern == 1:
            # Block pattern
            for r in range(min(rank, mp)):
                j, k = divmod(r, p)
                V[j * p + k, r] = 1
                for i in range(n):
                    U[i * m + j, r] = 1
                    W[k * n + i, r] = 1
        elif pattern == 2:
            # Another pattern
            for r in range(min(rank, pn)):
                k, i = divmod(r, n)
                W[k * n + i, r] = 1
                for j in range(m):
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
        
        if _verify_decomposition(n, m, p, U, V, W):
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None
# EVOLVE-BLOCK-END
