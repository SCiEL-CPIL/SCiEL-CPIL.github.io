"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment

def _sinkhorn(matrix, max_iter=100, tol=1e-8):
    """
    Sinkhorn-Knopp algorithm to project a positive matrix onto the Birkhoff polytope.
    """
    n = matrix.shape[0]
    # Ensure positivity
    matrix = np.maximum(matrix, 1e-10)
    
    u = np.ones(n)
    v = np.ones(n)
    
    for _ in range(max_iter):
        # Row normalization
        row_sums = matrix @ v
        u = 1.0 / row_sums
        # Column normalization
        col_sums = matrix.T @ u
        v = 1.0 / col_sums
        
        # Check convergence
        # Optional: check if u and v are stable
        if np.allclose(u, 1.0, atol=tol) and np.allclose(v, 1.0, atol=tol):
            break
            
    return (u[:, None]) * matrix * (v[None, :])

def solve(a, b):
    n = len(a)
    
    # 1. Continuous Relaxation via Gradient Descent on Birkhoff Polytope
    # Initialize X as uniform doubly stochastic matrix
    X = np.ones((n, n)) / n
    
    # Gradient step size
    # Scale based on matrix magnitudes
    scale_a = np.max(np.abs(a)) if n > 0 else 1.0
    scale_b = np.max(np.abs(b)) if n > 0 else 1.0
    eta = 0.01 / (scale_a * scale_b + 1e-6)
    
    # Optimization loop
    num_iters = 300
    for t in range(num_iters):
        # Compute Gradient
        # f(X) = tr(A X B X^T)
        # Grad = A^T X B^T + B X^T A
        # Assuming symmetric A, B: Grad = A X B + B X A
        G = a.T @ X @ b.T + b @ X.T @ a
        
        # Update
        X_raw = X - eta * G
        
        # Project back to Birkhoff Polytope using Sinkhorn
        X = _sinkhorn(X_raw)
        
        # Anneal learning rate
        eta *= 0.95
        
    # 2. Rounding to Discrete Permutation
    # Find permutation that maximizes sum_{i,j} X[i,j]
    # linear_sum_assignment minimizes cost, so we minimize -X
    row_ind, col_ind = linear_sum_assignment(-X)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    # 3. Local Search Refinement (2-Swap)
    # Compute initial cost
    def compute_cost(p):
        total = 0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total
        
    current_cost = compute_cost(p)
    
    # Simple 2-swap local search
    # Limit iterations to ensure time compliance
    improved = True
    max_passes = 5
    pass_count = 0
    
    while improved and pass_count < max_passes:
        improved = False
        pass_count += 1
        for i in range(n):
            for j in range(i + 1, n):
                old_pi, old_pj = p[i], p[j]
                p[i], p[j] = p[j], p[i]
                new_cost = compute_cost(p)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    p[i], p[j] = old_pi, old_pj
    
    return p.tolist()
# EVOLVE-BLOCK-END
