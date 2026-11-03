"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def solve(a, b):
    n = len(a)
    t0 = time.time()
    time_limit = 115.0  # leave 5s buffer for overhead
    
    # Convert to float for optimization
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    
    # Precompute: C = a ⊗ b (Kronecker product) would be too large (n^2 x n^2 = 22500 x 22500)
    # Instead, we work with the assignment matrix X (n x n) directly.
    # Cost = sum_{i,j} a[i,j] * b[X_i, X_j] where X is permutation.
    # In matrix form: cost = sum_{i,j,i',j'} a[i,j] * b[i',j'] * X[i,i'] * X[j,j']
    #                  = trace(X^T * (a ⊗ b) * X) but that's expensive.
    
    # Alternative: use the fact that for a permutation matrix P,
    # cost = sum_{i,j} a[i,j] * b[P[i], P[j]]
    #      = trace(A^T * B_P) where B_P has rows permuted by P
    #      Actually: cost = sum_{i,j} A[i,j] * B[P[i], P[j]]
    #      Let's define it as: for each facility i assigned to location p[i],
    #      the cost contribution involves flows from i to j and distances from p[i] to p[j].
    
    # Gradient-based approach on the Birkhoff polytope:
    # We maintain X (n x n) as a doubly stochastic matrix.
    # Objective: f(X) = sum_{i,j} a[i,j] * sum_{k,l} b[k,l] * X[i,k] * X[j,l]
    #           = sum_{k,l} b[k,l] * (sum_i X[i,k] * a_col_i ... )
    # Let's rewrite: f(X) = trace(X^T * M * X) where M is constructed appropriately?
    # Actually: f(X) = sum_{i,j,k,l} a[i,j] * b[k,l] * X[i,k] * X[j,l]
    #            = sum_{k,l} b[k,l] * (sum_i X[i,k] * (sum_j a[i,j] * X[j,l]))
    #            = sum_{k,l} b[k,l] * (X^T * (A * X))_{k,l}  ... let me check
    # Let Y = A * X, so Y[i,l] = sum_j A[i,j] * X[j,l]
    # Then f(X) = sum_{i,k} X[i,k] * Y[i,k] = sum_{i,k} X[i,k] * (A*X)[i,k] = trace(X^T * A * X)
    # Wait, that's not quite right because b is also involved.
    
    # Let me redo: f(X) = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # In relaxed form with X:
    # f(X) = sum_{i,j,k,l} a[i,j] * b[k,l] * X[i,k] * X[j,l]
    # Let's define S = X^T * B, so S[k,j] = sum_l X[l,k] * B[l,j] ... no
    # S[k,j] = sum_l X[l,k] * B[l,j]? No, X^T * B has [k,j] = sum_l X[l,k] * B[l,j]
    # Hmm, let's think differently.
    # f(X) = sum_{k,l} b[k,l] * (sum_{i,j} a[i,j] * X[i,k] * X[j,l])
    # Let C[k,l] = sum_{i,j} a[i,j] * X[i,k] * X[j,l]
    # C = X^T * A * X, so C[k,l] = sum_{i,j} X[i,k] * A[i,j] * X[j,l]
    # Yes! So f(X) = sum_{k,l} b[k,l] * C[k,l] = trace(B^T * C) = trace(B^T * X^T * A * X)
    #              = trace(X * B^T * X^T * A) = trace(A * X * B^T * X^T)
    #              = sum_{i,j,k,l} A[i,j] * X[i,k] * B[k,l] * X[j,l]  ✓
    
    # So f(X) = trace(A * X * B^T * X^T)
    
    # Gradient of f(X) w.r.t. X:
    # d/dX trace(A * X * B^T * X^T)
    # = A^T * X * B + B * X * A^T  ... let me derive carefully
    # f(X) = trace(A * X * B^T * X^T)
    # Let M = A * X * B^T, then f = trace(M * X^T) = sum M[i,j] * X[j,i]
    # df/dX[j,i] = M[i,j] = (A * X * B^T)[i,j]
    # So gradient = (A * X * B^T)^T = B * X^T * A^T
    # Wait: if f = sum_{i,j} M[i,j] * X[j,i], then df/dX[j,i] = M[i,j]
    # So gradient[j,i] = M[i,j] = (A*X*B^T)[i,j]
    # gradient = (A*X*B^T)^T = B * X^T * A^T
    
    # Let's verify with small case. Actually, let's just use:
    # grad = B @ X.T @ A.T + A.T @ X @ B  ... I need to be more careful.
    
    # f(X) = trace(A X B^T X^T)
    # Using matrix calculus: d trace(M X N X^T) = (M N^T + N M^T) X^T ... 
    # Let me just compute numerically or use a known formula.
    # 
    # Actually, f(X) = sum_{i,j,k,l} A[i,j] X[i,k] B[k,l] X[j,l]
    # df/dX[p,q] = sum_{j,l} A[p,j] B[q,l] X[j,l] + sum_{i,k} A[i,q] B[k,p] X[i,k]
    #            = (A @ X @ B.T)[p,q] + (B @ X.T @ A.T)[p,q]
    #            = (A @ X @ B.T + B @ X.T @ A.T)[p,q]
    
    # So grad = A @ X @ B.T + B @ X.T @ A.T
    
    # Initialize X as doubly stochastic (uniform or from rank matching)
    # Start with rank-matching permutation, then relax
    heavy = np.argsort(-a_f.sum(axis=1), kind="stable")
    central = np.argsort(b_f.sum(axis=1), kind="stable")
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy] = central
    
    # Create initial doubly stochastic matrix from permutation
    X = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        X[i, p0[i]] = 1.0
    
    # Projected gradient descent on Birkhoff polytope
    # Use Sinkhorn-Knopp for projection to doubly stochastic
    
    def sinkhorn(X, max_iter=100, tol=1e-6):
        """Project X to doubly stochastic matrix using Sinkhorn-Knopp."""
        X = np.maximum(X, 1e-10)
        for _ in range(max_iter):
            row_sums = X.sum(axis=1, keepdims=True)
            X = X / row_sums
            col_sums = X.sum(axis=0, keepdims=True)
            X = X / col_sums
            if np.max(np.abs(X.sum(axis=1) - 1.0)) < tol and np.max(np.abs(X.sum(axis=0) - 1.0)) < tol:
                break
        return X
    
    # Adaptive step size
    step_size = 1.0 / (n * max(a_f.max(), b_f.max(), 1.0))
    
    best_cost = np.inf
    best_X = X.copy()
    
    for iteration in range(5000):
        if time.time() - t0 > time_limit:
            break
        
        # Compute gradient
        grad = a_f @ X @ b_f.T + b_f @ X.T @ a_f.T
        
        # Compute current cost
        C = X.T @ a_f @ X
        cost = np.sum(b_f * C)
        
        if cost < best_cost:
            best_cost = cost
            best_X = X.copy()
        
        # Gradient descent step
        X = X - step_size * grad
        
        # Project to doubly stochastic
        X = sinkhorn(X, max_iter=50)
        
        # Adapt step size: if cost increased significantly, reduce step
        C_new = X.T @ a_f @ X
        cost_new = np.sum(b_f * C_new)
        if cost_new > cost * 1.01:
            step_size *= 0.5
        elif cost_new < cost * 0.99:
            step_size = min(step_size * 1.1, 1.0 / (n * max(a_f.max(), b_f.max(), 1.0) * 0.1))
    
    # Recover permutation from best_X using Hungarian algorithm
    # Minimize sum of costs: use -best_X as the cost matrix for linear_sum_assignment
    # (since we want to maximize the entries of best_X, which is equivalent to minimizing -best_X)
    row_ind, col_ind = linear_sum_assignment(-best_X)
    p = np.empty(n, dtype=np.int64)
    for i, j in zip(row_ind, col_ind):
        p[i] = j
    
    return p.tolist()
# EVOLVE-BLOCK-END
