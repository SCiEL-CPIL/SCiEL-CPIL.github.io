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
    
    The idea is to exploit the block structure: A is 2x3, B is 3x6.
    We can split B into three 3x2 blocks: B = [B1, B2, B3]
    Then C = A * B = [A*B1, A*B2, A*B3]
    
    For each 2x3x2 multiplication, we know that the minimal rank is 8 
    (this is a known result). So if we can find a rank-8 decomposition 
    for 2x3x2, we can use it three times to get rank 24 for 2x3x6.
    
    However, there might be a way to share computations across the three 
    blocks to reduce the total rank further. Let's first try to find a 
    rank-8 decomposition for 2x3x2.
    """
    n, m, p = 2, 3, 2  # For the subproblem
    
    # The tensor for 2x3x2 has dimension 6x6x4
    # We want to find U (6xR), V (6xR), W (4xR) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 
    # when (i,j,k) is a valid index triple, else 0
    
    # Known: rank of 2x3x2 matrix multiplication is 8
    # Let's try to construct it explicitly.
    
    # One approach: use the fact that 2x3x2 can be embedded in 3x3x3 
    # and use Strassen's algorithm. But let's try a direct construction.
    
    # Actually, let me try a different approach: use numerical optimization 
    # to find a rank-8 decomposition for 2x3x2.
    
    # Initialize with random values
    np.random.seed(42)
    R = 8
    U = np.random.randn(n * m, R) * 0.1
    V = np.random.randn(m * p, R) * 0.1
    W = np.random.randn(p * n, R) * 0.1
    
    # Define the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Use gradient descent to minimize ||T_dec - T||_F^2
    lr = 0.01
    for it in range(5000):
        T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
        diff = T_dec - T
        loss = np.sum(diff ** 2)
        
        if loss < 1e-10:
            break
        
        # Gradients
        dU = np.einsum('ijk,jr,kr->ir', diff, V, W)
        dV = np.einsum('ijk,ir,kr->jr', diff, U, W)
        dW = np.einsum('ijk,ir,jr->kr', diff, U, V)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Check if we found a valid decomposition
    T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
    if np.allclose(T_dec, T, atol=1e-8):
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
