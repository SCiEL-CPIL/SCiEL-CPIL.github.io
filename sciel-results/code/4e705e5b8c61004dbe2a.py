import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the n×m×p matrix multiplication tensor."""
    rank = U.shape[1]
    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    # Compute decomposition tensor
    T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_dec, T, atol=tol)

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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
    return U, V, W, rank

def _try_strassen_2x3x6():
    """
    Attempt to find a lower-rank decomposition for 2x3x6 using structured 
    approaches based on Strassen-like identities.
    
    Strategy: Split B (3x6) into three 3x2 blocks. Use a rank-8 decomposition
    for 2x3x2 and try to share the U factor across all three blocks.
    
    If we can find U (6x8), V_block (6x8), W_block (4x8) for 2x3x2, then
    for 2x3x6 we need U (6x24), V (18x24), W (12x24).
    
    Key insight: The U factor can be shared if we structure the decomposition
    so that the same intermediate computations are reused.
    """
    n, m, p = 2, 3, 2  # For the subproblem
    
    # Try to find a rank-8 decomposition for 2x3x2
    # Use a more robust optimization approach
    
    np.random.seed(42)
    R = 8
    
    # Initialize with structured random values
    U = np.random.randn(n * m, R) * 0.5
    V = np.random.randn(m * p, R) * 0.5
    W = np.random.randn(p * n, R) * 0.5
    
    # Define the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Use Adam-like optimization with adaptive learning rate
    lr = 0.05
    momentum_U = np.zeros_like(U)
    momentum_V = np.zeros_like(V)
    momentum_W = np.zeros_like(W)
    
    for it in range(10000):
        T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
        diff = T_dec - T
        loss = np.sum(diff ** 2)
        
        if loss < 1e-12:
            break
        
        # Compute gradients
        dU = np.einsum('ijk,jr,kr->ir', diff, V, W)
        dV = np.einsum('ijk,ir,kr->jr', diff, U, W)
        dW = np.einsum('ijk,ir,jr->kr', diff, U, V)
        
        # Update with momentum
        momentum_U = 0.9 * momentum_U + 0.1 * dU
        momentum_V = 0.9 * momentum_V + 0.1 * dV
        momentum_W = 0.9 * momentum_W + 0.1 * dW
        
        U -= lr * momentum_U
        V -= lr * momentum_V
        W -= lr * momentum_W
    
    # Check if we found a valid decomposition
    T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
    if np.allclose(T_dec, T, atol=1e-10):
        # Now use this to build the 2x3x6 decomposition
        n_full, m_full, p_full = 2, 3, 6
        R_full = 3 * R  # 24
        
        U_full = np.zeros((n_full * m_full, R_full))
        V_full = np.zeros((m_full * p_full, R_full))
        W_full = np.zeros((p_full * n_full, R_full))
        
        for block in range(3):
            offset = block * R
            # U is the same for all blocks
            U_full[:, offset:offset+R] = U
            
            # V: for block 'block', we use rows j*p_full + (block*2+k) for j in 0..2, k in 0..1
            for j in range(m_full):
                for k in range(2):
                    V_full[j * p_full + (block * 2 + k), offset:offset+R] = V[j * 2 + k]
            
            # W: for block 'block', we use rows (block*2+k)*n_full + i for k in 0..1, i in 0..1
            for k in range(2):
                for i in range(n_full):
                    W_full[(block * 2 + k) * n_full + i, offset:offset+R] = W[k * n_full + i]
        
        # Verify
        if _verify_decomposition(U_full, V_full, W_full, n_full, m_full, p_full):
            return U_full, V_full, W_full, R_full
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # First, try to find a structured lower-rank decomposition
    best_solution = None
    best_rank = n * m * p  # 36
    
    # Try Strassen-like approach
    if time.time() < deadline - 30:
        result = _try_strassen_2x3x6()
        if result is not None:
            U, V, W, R = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_solution = (U, V, W, R)
                best_rank = R
    
    # If no improvement found, fall back to schoolbook
    if best_solution is None:
        U, V, W, rank = _schoolbook(n, m, p)
        best_solution = (U, V, W, rank)
        best_rank = rank
    
    U, V, W, R = best_solution
    
    # Convert to lists for return
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
