"""Matrix multiplication 2x3 * 3x6: rank-minimal tensor decomposition."""
import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            s = 0.0
                            for t in range(rank):
                                s += U[a*m+b][t] * V[c*p+d][t] * W[k*n+i][t]
                            expected = (1.0 if (a==i and c==b and d==k) else 0.0)
                            if abs(s - expected) > 1e-9:
                                return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Construct the schoolbook decomposition of rank n*m*p."""
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
    
    return U, V, W, rank

def _rank_18_via_block_structure(n, m, p, deadline):
    """
    Try to find a rank-18 decomposition using a structured parameterization.
    
    The key insight: for 2x3 * 3x6, we can group the 6 rows of B (and 6 columns of C)
    into 3 groups of 2. This gives us a "block" structure where:
    - U has shape (6, 18) - indexed by (i, j) where i in {0,1}, j in {0,1,2}
    - V has shape (18, 18) - indexed by (j, k) where j in {0,1,2}, k in {0,...,5}
    - W has shape (12, 18) - indexed by (k, i) where k in {0,...,5}, i in {0,1}
    
    We use a parameterized form that respects the block structure and optimize.
    """
    rank = 18
    
    # We'll parameterize U, V, W with a structured form
    # U[i, j, t] for i in {0,1}, j in {0,1,2}, t in {0,...,17}
    # V[j, k, t] for j in {0,1,2}, k in {0,...,5}, t in {0,...,17}
    # W[k, i, t] for k in {0,...,5}, i in {0,1}, t in {0,...,17}
    
    # Flatten to vectors
    # U_flat: 6 * 18 = 108
    # V_flat: 18 * 18 = 324
    # W_flat: 12 * 18 = 216
    # Total: 648 parameters
    
    np.random.seed(42)
    
    # Initialize with small random values
    U_flat = np.random.randn(n * m, rank) * 0.5
    V_flat = np.random.randn(m * p, rank) * 0.5
    W_flat = np.random.randn(p * n, rank) * 0.5
    
    x = np.concatenate([U_flat, V_flat, W_flat])
    
    def _get_matrices(x_vec):
        U_v = x_vec[:n*m*rank].reshape(n*m, rank)
        V_v = x_vec[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_v = x_vec[n*m*rank + m*p*rank:].reshape(p*n, rank)
        return U_v, V_v, W_v
    
    def _compute_error(x_vec):
        U_v, V_v, W_v = _get_matrices(x_vec)
        
        error = 0.0
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for i in range(n):
                            for k in range(p):
                                s = 0.0
                                for t in range(rank):
                                    s += U_v[a*m+b, t] * V_v[c*p+d, t] * W_v[k*n+i, t]
                                expected = 1.0 if (a==i and c==b and d==k) else 0.0
                                error += (s - expected) ** 2
        return error
    
    def _compute_gradient(x_vec):
        U_v, V_v, W_v = _get_matrices(x_vec)
        
        grad_U = np.zeros_like(U_v)
        grad_V = np.zeros_like(V_v)
        grad_W = np.zeros_like(W_v)
        
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for i in range(n):
                            for k in range(p):
                                expected = 1.0 if (a==i and c==b and d==k) else 0.0
                                s = 0.0
                                for t in range(rank):
                                    s += U_v[a*m+b, t] * V_v[c*p+d, t] * W_v[k*n+i, t]
                                diff = s - expected
                                for t in range(rank):
                                    grad_U[a*m+b, t] += 2 * diff * V_v[c*p+d, t] * W_v[k*n+i, t]
                                    grad_V[c*p+d, t] += 2 * diff * U_v[a*m+b, t] * W_v[k*n+i, t]
                                    grad_W[k*n+i, t] += 2 * diff * U_v[a*m+b, t] * V_v[c*p+d, t]
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    # Adam optimizer for better convergence
    def _adam_step(x_vec, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        nonlocal m_adam, v_adam, t_adam
        if 'm_adam' not in globals():
            m_adam = np.zeros_like(x_vec)
            v_adam = np.zeros_like(x_vec)
            t_adam = 0
            globals()['m_adam'] = m_adam
            globals()['v_adam'] = v_adam
            globals()['t_adam'] = t_adam
        
        grad = _compute_gradient(x_vec)
        t_adam += 1
        m_adam = beta1 * m_adam + (1 - beta1) * grad
        v_adam = beta2 * v_adam + (1 - beta2) * grad**2
        m_hat = m_adam / (1 - beta1**t_adam)
        v_hat = v_adam / (1 - beta2**t_adam)
        x_new = x_vec - lr * m_hat / (np.sqrt(v_hat) + eps)
        return x_new, grad
    
    # Use Adam with adaptive learning rate
    lr = 0.05
    best_error = _compute_error(x)
    best_x = x.copy()
    
    max_iterations = 3000
    for it in range(max_iterations):
        if time.time() > deadline - 15:
            break
        
        err = _compute_error(x)
        if err < best_error:
            best_error = err
            best_x = x.copy()
        
        if best_error < 1e-12:
            break
        
        # Adam step
        x, grad = _adam_step(x, lr=lr)
        
        # Check for divergence
        new_err = _compute_error(x)
        if new_err > err * 1.1 or np.any(np.isnan(x)):
            lr *= 0.5
            x = best_x
        elif it % 50 == 0 and best_error < 1.0:
            lr *= 1.05
    
    # Extract best solution
    U_best = best_x[:n*m*rank].reshape(n*m, rank)
    V_best = best_x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W_best = best_x[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    # Convert to list of lists
    U_list = [[float(val) for val in row] for row in U_best]
    V_list = [[float(val) for val in row] for row in V_best]
    W_list = [[float(val) for val in row] for row in W_best]
    
    # Verify
    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
        return U_list, V_list, W_list, rank
    
    return None

def _try_rank_18(n, m, p, deadline):
    """
    Try to find a rank-18 decomposition using structured parameterization.
    """
    result = _rank_18_via_block_structure(n, m, p, deadline)
    if result is not None:
        return result
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    Convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    with W[k*n+i, t] indexing.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook_decomposition(n, m, p)
    
    # Verify schoolbook is correct
    if not _verify_decomposition(U, V, W, n, m, p):
        # Should never happen
        pass
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try rank 18
    if time.time() < deadline - 15:
        result = _try_rank_18(n, m, p, deadline)
        if result is not None:
            U18, V18, W18, rank18 = result
            if _verify_decomposition(U18, V18, W18, n, m, p):
                best_U, best_V, best_W, best_rank = U18, V18, W18, rank18
    
    # Return the best decomposition found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
