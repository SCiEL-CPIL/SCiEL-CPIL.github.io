"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build the target tensor T as a 3D array of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try ranks from lower bound upward
    lower_bound = max(n*m, m*p, p*n)  # 35
    best_result = None
    best_rank = None
    
    # First, try Strassen-based construction which should give rank ~99
    strassen_result = _strassen_decompose(n, m, p)
    if strassen_result is not None:
        best_result = strassen_result
        best_rank = strassen_result['rank']
    
    # Try to improve with search
    if time.time() < deadline - 30:
        for rank in range(lower_bound, best_rank):
            if time.time() > deadline - 20:
                break
            found = _try_rank(T, n, m, p, rank, deadline)
            if found is not None:
                best_result = found
                best_rank = rank
                break
    
    if best_result is None:
        # Fallback to schoolbook
        best_result = _schoolbook(n, m, p)
    
    return best_result

def _strassen_decompose(n, m, p):
    """Use Strassen's algorithm on 2x2x2 blocks to reduce rank below schoolbook."""
    # Pad to even dimensions for Strassen
    n_pad = (n + 1) & ~1  # next even
    m_pad = (m + 1) & ~1
    p_pad = (p + 1) & ~1
    
    # For 3x5x7: n_pad=4, m_pad=6, p_pad=8
    # Strassen on 2x2 blocks: rank for 2x2x2 is 7 instead of 8
    # Number of 2x2x2 blocks: (n_pad/2) * (m_pad/2) * (p_pad/2)
    # Each block contributes 7 rank factors instead of 8
    # Total rank = 7 * (n_pad/2) * (m_pad/2) * (p_pad/2)
    
    blocks_n = n_pad // 2
    blocks_m = m_pad // 2
    blocks_p = p_pad // 2
    
    rank = 7 * blocks_n * blocks_m * blocks_p
    # For 4x6x8: 7 * 2 * 3 * 4 = 168... that's worse than schoolbook 105
    
    # Better approach: use Strassen only where beneficial
    # Schoolbook rank = n*m*p = 105
    # We need a smarter decomposition
    
    # Let's use a block Strassen approach:
    # Split A (3x5) into blocks, B (5x7) into blocks
    # Use Strassen multiplication on compatible block sizes
    
    # Actually, let me try a different approach:
    # Use the fact that for certain dimension combinations, we can find
    # decompositions with rank < n*m*p
    
    # Try to find a decomposition using bilinear form optimization
    # For now, return None and let the search handle it
    return None

def _try_rank(T, n, m, p, rank, deadline):
    """Try to find a rank-`rank` decomposition of T using random search."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    num_restarts = 3
    max_iter = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Random initialization with small values
        np.random.seed(42 + restart)
        U = np.random.randn(n_m, rank) * 0.1
        V = np.random.randn(m_p, rank) * 0.1
        W = np.random.randn(p_n, rank) * 0.1
        
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            # Compute current tensor approximation
            # T_approx[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
            # This is expensive for large tensors, so use gradient-based approach
            
            # Compute gradient with respect to U, V, W
            # dT/du[a,d] = sum_{b,c} T[a,b,c] * V[b,d] * W[c,d] - sum_{b,c} T_approx[a,b,c] * V[b,d] * W[c,d]
            # Simplify: grad_U[a,d] = (T * (V[d] ⊗ W[d]))_a - (T_approx * (V[d] ⊗ W[d]))_a
            
            # This is still expensive. Let's use a simpler approach:
            # Just check if we can find exact zeros in the residual
            
            err = _compute_error(U, V, W, T, n_m, m_p, p_n)
            if err < 1e-10:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": rank}
            
            if err < prev_err:
                prev_err = err
            else:
                break
            
            # Simple gradient step
            lr = 0.01
            grad_U = _grad_U(U, V, W, T, n_m, m_p, p_n)
            grad_V = _grad_V(U, V, W, T, n_m, m_p, p_n)
            grad_W = _grad_W(U, V, W, T, n_m, m_p, p_n)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
    
    return None

def _compute_error(U, V, W, T, n_m, m_p, p_n):
    """Compute ||T - sum_d U[:,d]⊗V[:,d]⊗W[:,d]||_F"""
    # T_approx[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
    # Use einsum for efficiency
    T_approx = np.einsum('ad,bd,cd->abc', U, V, W)
    return np.linalg.norm(T - T_approx)

def _grad_U(U, V, W, T, n_m, m_p, p_n):
    """Gradient of error w.r.t. U"""
    T_approx = np.einsum('ad,bd,cd->abc', U, V, W)
    residual = T_approx - T
    # grad_U[a,d] = sum_{b,c} residual[a,b,c] * V[b,d] * W[c,d]
    return np.einsum('abc,bd,cd->ad', residual, V, W)

def _grad_V(U, V, W, T, n_m, m_p, p_n):
    """Gradient of error w.r.t. V"""
    T_approx = np.einsum('ad,bd,cd->abc', U, V, W)
    residual = T_approx - T
    return np.einsum('abc,ad,cd->bd', residual, U, W)

def _grad_W(U, V, W, T, n_m, m_p, p_n):
    """Gradient of error w.r.t. W"""
    T_approx = np.einsum('ad,bd,cd->abc', U, V, W)
    residual = T_approx - T
    return np.einsum('abc,ad,bd->cd', residual, U, V)

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W, "rank": rank}
# EVOLVE-BLOCK-END
