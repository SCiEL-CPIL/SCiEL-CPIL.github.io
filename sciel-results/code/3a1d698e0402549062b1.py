"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s kill

# EVOLVE-BLOCK-START

def _schoolbook_decomposition(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build true tensor
    T_true = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Reconstruct
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check exactness
    if not np.allclose(T_recon, T_true, atol=tol):
        return False
    return True

def _strassen_2x2():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3: a00*(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4: a11*(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[3, 3] = 1
    
    # M5: (a00+a01)*b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1
    
    # M6: (a10-a00)*(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = 1; W[3, 5] = 1
    
    # M7: (a01-a11)*(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1
    
    return U, V, W

def _try_rank_reduction(n, m, p, deadline):
    """
    Try to find a decomposition with rank less than n*m*p.
    Uses block decomposition strategies.
    """
    schoolbook_rank = n * m * p
    best_U, best_V, best_W = _schoolbook_decomposition(n, m, p)
    best_rank = schoolbook_rank
    
    # For 3x4x8, we can try to exploit structure:
    # Split n=3 into n=2 and n=1
    # A = [A_top (2x4); A_bottom (1x4)]
    # B is 4x8
    # C_top = A_top * B (2x8), C_bottom = A_bottom * B (1x8)
    
    # For C_top (2x8), we can further split p=8 into four 2x2 blocks
    # This is complex, so let's try a simpler approach:
    # Use Strassen for the 2x2 sub-blocks where possible
    
    # Actually, let's try a more direct approach:
    # The tensor T[i,j,k] = 1 for i in 0..2, j in 0..3, k in 0..7
    # We can try to find a lower-rank decomposition using random search
    
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    lb = max(n*m, m*p, p*n)
    
    # Try ranks from lb up to schoolbook_rank
    # But this is a hard problem, so we'll limit the search
    
    # Strategy: Use gradient descent to minimize the error
    # T_recon - T_true, where T_recon = sum_r U[:,r] V[:,r] W[:,r]
    
    # For each candidate rank r, try to find U, V, W
    # such that the tensor equation holds
    
    # This is a non-convex problem, so we'll use multiple random restarts
    
    # Time budget: 100 seconds
    time_budget = deadline - time.time()
    
    # Try a few ranks
    for r in range(lb, min(lb + 50, schoolbook_rank + 1)):
        if time.time() > deadline:
            break
        
        # Try to find a rank-r decomposition
        success = _try_rank_r(r, n, m, p, time_budget / 50)
        if success:
            return success[0], success[1], success[2], r
    
    # If no improvement found, return schoolbook
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        raise RuntimeError("Schoolbook decomposition failed verification")
    
    return best_U, best_V, best_W, best_rank

def _try_rank_r(r, n, m, p, time_limit):
    """
    Try to find a rank-r decomposition using gradient descent.
    Returns (U, V, W) if successful, None otherwise.
    """
    if r < max(n*m, m*p, p*n):
        return None
    
    # Build true tensor
    T_true = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Try multiple random initializations
    for init in range(10):
        if time.time() > time_limit:
            break
        
        rng = np.random.RandomState(42 + init)
        
        # Initialize U, V, W
        U = rng.randn(n*m, r) * 0.1
        V = rng.randn(m*p, r) * 0.1
        W = rng.randn(p*n, r) * 0.1
        
        # Gradient descent
        eta = 0.01
        for it in range(1000):
            if time.time() > time_limit:
                break
            
            # Compute residual
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            R = T_true - T_recon
            
            # Compute gradients
            grad_U = np.einsum('abc,br,cr->ar', R, V, W)
            grad_V = np.einsum('abc,ar,cr->br', R, U, W)
            grad_W = np.einsum('abc,ar,br->cr', R, U, V)
            
            # Update
            U += eta * grad_U
            V += eta * grad_V
            W += eta * grad_W
            
            # Check convergence
            if it % 100 == 0:
                err = np.max(np.abs(T_recon - T_true))
                if err < 1e-10:
                    # Verify exactly
                    if _verify_decomposition(U, V, W, n, m, p):
                        return U, V, W
    
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.
    Returns dict with 'U', 'V', 'W' keys, each a 2D list.
    """
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    # Try to find an improved decomposition
    U, V, W, rank = _try_rank_reduction(n, m, p, deadline)
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
