"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _schoolbook_decomp(n, m, p):
    """Return schoolbook decomposition for n x m * m x p -> n x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    R = U.shape[1]
    
    # For each (i, j, k), check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                val = np.sum(prod)
                errors.append(abs(val - 1.0))
    return max(errors) < tol

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is exact (within tolerance)."""
    return _verify_fast(U, V, W, n, m, p, tol)

def _round_to_half(x):
    """Round to nearest half-integer."""
    return np.round(x * 2) / 2.0

def _try_structured_reduction(n, m, p):
    """
    Try to find a structured reduction by splitting the inner dimension.
    For n x m x p, split m into m1 + m2, giving two subproblems n x m1 x p and n x m2 x p.
    If we can find rank-R1 and R2 decompositions, total rank is R1 + R2.
    """
    # Try splitting m into 2 + 2 for m=4
    if m == 4:
        m1, m2 = 2, 2
        # Schoolbook for each: n*m1*p and n*m2*p
        # Total: n*m1*p + n*m2*p = n*m*p (no improvement)
        # But maybe we can do better with shared structure?
        pass
    
    return None

def _construct_with_strassen_blocks(n, m, p):
    """
    Try to construct a decomposition using Strassen-like block techniques.
    For 3x4x8, we can view A as [A_left, A_right] where A_left is 3x2, A_right is 3x2.
    B is [B_top; B_bottom] where B_top is 2x8, B_bottom is 2x8.
    C = A_left * B_top + A_right * B_bottom.
    
    Each term is a 3x2 * 2x8 product.
    The rank of a 3x2x8 tensor is at most 3*2*8 = 48.
    But we need to compute the sum, so total rank is at most 48 + 48 = 96 (schoolbook).
    
    However, if we can find a rank-R decomposition for 3x2x8, then total is 2R.
    """
    # For now, return schoolbook as we don't have a known better decomposition for 3x2x8
    return None

def _greedy_rank_search(n, m, p, target_rank, num_trials=10, timeout=10):
    """
    Try to find a rank-target_rank decomposition using greedy/randomized approach.
    """
    start = time.time()
    R = target_rank
    
    for trial in range(num_trials):
        if time.time() - start > timeout:
            break
        
        # Initialize with random matrices
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        # Try a few ALS-like iterations
        for it in range(20):
            # Update W
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, R))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, R))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            if it % 5 == 0:
                err = _verify_fast(U, V, W, n, m, p, tol=1e-6)
                if err < 1e-6:
                    # Try to round
                    U_r = _round_to_half(U)
                    V_r = _round_to_half(V)
                    W_r = _round_to_half(W)
                    if _verify(U_r, V_r, W, n, m, p, tol=1e-6):
                        return {'U': U_r.tolist(), 'V': V_r.tolist(), 'W': W_r.tolist()}
        
        # Final check
        err = _verify_fast(U, V, W, n, m, p, tol=1e-6)
        if err < 1e-6:
            U_r = _round_to_half(U)
            V_r = _round_to_half(V)
            W_r = _round_to_half(W)
            if _verify(U_r, V_r, W, n, m, p, tol=1e-6):
                return {'U': U_r.tolist(), 'V': V_r.tolist(), 'W': W_r.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    schoolbook_rank = n * m * p  # 96
    
    # Try structured approaches first (fast)
    result = _construct_with_strassen_blocks(n, m, p)
    if result is not None:
        return result
    
    # Try greedy search for lower ranks
    # Start from lower ranks and work up
    for target_rank in range(50, schoolbook_rank):
        if time.time() > deadline:
            break
        
        remaining_time = deadline - time.time()
        if remaining_time < 3:
            break
        
        # Allocate time based on remaining budget
        timeout = min(5, remaining_time * 0.3)
        
        result = _greedy_rank_search(n, m, p, target_rank, 
                                     num_trials=5, 
                                     timeout=timeout)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook_decomp(n, m, p)
# EVOLVE-BLOCK-END
