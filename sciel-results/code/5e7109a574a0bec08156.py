import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def compute_cost(p):
        """Compute QAP cost for permutation p."""
        return np.sum(a * b[np.ix_(p, p)])
    
    # Step 1: Birkhoff polytope projection via iterative scaling (Sinkhorn)
    # We optimize a continuous matrix X over the Birkhoff polytope
    # to minimize trace(X^T B X A) or equivalent formulations.
    
    # Initialize X with rank-matching based permutation matrix
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), p_init] = 1.0
    
    # Step 2: Gradient descent on Birkhoff polytope
    # The objective is f(X) = sum_{i,j,k,l} A[i,j] X[i,k] B[k,l] X[j,l]
    # Gradient: grad[i,k] = sum_j A[i,j] * (B[k, p_j] + B[p_j, k]) where p_j is the location of facility j
    # But since X is continuous, we need the full gradient:
    # grad[i,k] = sum_j A[i,j] * sum_l B[k,l] X[j,l] + sum_j A[j,i] * sum_l X[j,k] B[l,k]
    #           = (A @ (X @ B.T))[i,k] + (A.T @ (X @ B))[i,k]
    
    # Actually, let me be more careful:
    # f(X) = sum_{i,j} A[i,j] * sum_{k,l} X[i,k] B[k,l] X[j,l]
    #      = sum_{i,j,k,l} A[i,j] X[i,k] B[k,l] X[j,l]
    
    # df/dX[i,k] = sum_{j,l} A[i,j] B[k,l] X[j,l] + sum_{j,l} A[j,i] X[j,k] B[l,k]
    #            = sum_j A[i,j] (B @ X.T)[k,j] + sum_j A[j,i] (X @ B)[k,j]
    #            = (A @ (B @ X.T))[i,k] + (A.T @ (X @ B))[i,k]
    
    # Wait, let me re-derive:
    # f(X) = Tr(X^T B X A) = Tr(B X A X^T)
    # df/dX = B X A^T + B^T X A
    # Since B is symmetric (distance matrix), B^T = B:
    # df/dX = B X A^T + B X A
    
    # Let's verify: f(X) = sum_{i,j,k,l} X[i,k] B[k,l] X[j,l] A[i,j]
    # df/dX[i,k] = sum_{j,l} B[k,l] X[j,l] A[i,j] + sum_{j,l} X[j,k] B[l,j] A[i,j]
    #            = sum_j A[i,j] sum_l B[k,l] X[j,l] + sum_j A[i,j] sum_l X[j,k] B[l,j]
    #            = sum_j A[i,j] (B @ X)[k,j] + sum_j A[i,j] (X @ B)[k,j]
    #            = (A @ (B @ X))[i,k] + (A @ (X @ B))[i,k]
    
    # Hmm, let me just use the formula: grad = B @ X @ A.T + B @ X @ A
    # Since A is symmetric in QAP typically, A.T = A, so grad = 2 * B @ X @ A
    
    # But to be safe, let's compute it properly:
    # grad[i,k] = sum_{j,l} A[i,j] * B[k,l] * X[j,l] + sum_{j,l} A[j,i] * X[j,k] * B[l,k]
    #           = (A @ (B @ X.T))[i,k] + (A.T @ (X @ B))[i,k]
    
    # Let me use a different formulation that's easier to verify:
    # f(X) = sum_{i,j} A[i,j] * (X B X^T)[i,j]
    # Let Y = X B X^T, then f = sum A[i,j] Y[i,j]
    # dY/dX[i,k] = B[k,:] X[:,i]^T + X[:,k] B[:,i]^T ... this is getting complex.
    
    # Let's use the standard result: for f(X) = Tr(X^T M X N) where M, N symmetric:
    # df/dX = M X N + M^T X N^T
    # In our case, f(X) = sum_{i,j,k,l} A[i,j] X[i,k] B[k,l] X[j,l]
    # This is Tr(X^T B X A) if we write it as sum_{i,k} X[i,k] (B X A)[k,i]
    # So M = B, N = A, and since both are symmetric:
    # grad = B @ X @ A + B @ X @ A = 2 * B @ X @ A
    
    # But wait, X is not necessarily symmetric, so let me be careful.
    # f(X) = sum_{i,j,k,l} A[i,j] X[i,k] B[k,l] X[j,l]
    # Let's compute df/dX[p,q]:
    # = sum_{i,j,k,l} A[i,j] (delta_{p,i} delta_{q,k} B[k,l] X[j,l] + X[i,k] B[k,l] delta_{p,j} delta_{q,l})
    # = sum_{j,l} A[p,j] B[q,l] X[j,l] + sum_{i,k} A[i,p] X[i,k] B[k,q]
    # = (A @ (B @ X.T))[p,q] + (A.T @ (X @ B))[p,q]
    
    # Since A and B are symmetric:
    # = (A @ (B @ X.T))[p,q] + (A @ (X @ B))[p,q]
    
    # Let's define the gradient computation:
    def compute_gradient(X):
        """Compute gradient of f(X) = sum_{i,j} A[i,j] * (X B X^T)[i,j]"""
        # grad[p,q] = sum_j A[p,j] sum_l B[q,l] X[j,l] + sum_i A[i,p] sum_k X[i,k] B[k,q]
        #           = (A @ (B @ X.T))[p,q] + (A.T @ (X @ B))[p,q]
        # Since A is symmetric: A.T = A
        term1 = a @ (b @ X.T)  # A @ (B @ X.T)
        term2 = a @ (X @ b)    # A @ (X @ B)  [since A.T = A]
        return term1 + term2
    
    # Sinkhorn scaling to project onto Birkhoff polytope
    def sinkhorn_project(X, max_iter=100, tol=1e-6):
        """Project matrix X onto Birkhoff polytope using iterative scaling."""
        X = X.copy()
        X = np.maximum(X, 1e-10)  # Ensure positivity
        
        for _ in range(max_iter):
            # Scale rows to sum to 1
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            X = X / row_sums
            
            # Scale columns to sum to 1
            col_sums = X.sum(axis=0, keepdims=True)
            col_sums = np.maximum(col_sums, 1e-10)
            X = X / col_sums
            
            # Check convergence
            row_sums = X.sum(axis=1)
            col_sums = X.sum(axis=0)
            if (np.max(np.abs(row_sums - 1.0)) < tol and 
                np.max(np.abs(col_sums - 1.0)) < tol):
                break
        
        return X
    
    # Step 3: Optimized gradient descent with Sinkhorn projection
    best_p = p_init.copy()
    best_cost = compute_cost(best_p)
    
    # Adaptive learning rate
    lr = 0.01
    no_improve = 0
    max_no_improve = 50
    
    for iteration in range(2000):  # Max iterations
        if time.time() > deadline:
            break
        
        # Compute gradient
        grad = compute_gradient(X)
        
        # Gradient descent step
        X_new = X - lr * grad
        
        # Project onto Birkhoff polytope
        X_new = sinkhorn_project(X_new)
        
        # Compute cost of the rounded solution
        # Use Hungarian algorithm to get permutation from X_new
        # Minimize cost = sum X_new[i,j] * C[i,j] where C[i,j] is the cost of assigning i to j
        # But we need the QAP cost, not just linear.
        
        # For rounding, use the spectral method:
        # The relaxed solution X gives us a soft assignment.
        # We can round by finding the permutation that maximizes sum X[i,p[i]]
        # This is the linear assignment problem on -X.
        
        row_ind, col_ind = linear_sum_assignment(-X_new)
        p_rounded = col_ind.copy()
        
        cost_rounded = compute_cost(p_rounded)
        
        if cost_rounded < best_cost:
            best_cost = cost_rounded
            best_p = p_rounded.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        X = X_new
        
        # Adaptive learning rate
        if no_improve >= max_no_improve:
            # Reduce learning rate and reset
            lr *= 0.5
            no_improve = 0
            
            # Also try a perturbation: mix with random permutation
            if lr < 0.001:
                # Reset X to a new random initialization
                p_rand = np.random.permutation(n)
                X = np.zeros((n, n), dtype=np.float64)
                X[np.arange(n), p_rand] = 1.0
                X = sinkhorn_project(X + 0.1 * np.random.randn(n, n))
                lr = 0.01
    
    # Step 4: Local search refinement from the best solution found
    current_p = best_p.copy()
    current_cost = best_cost
    
    # Use vectorized delta computation for 2-opt
    # Precompute B_perm = B[np.ix_(p, p)]
    B_perm = b[np.ix_(current_p, current_p)]
    
    # Row and column contributions
    row_contrib = a @ B_perm  # row_contrib[i] = sum_j a[i,j] * B_perm[i,j]
    col_contrib = a.T @ B_perm  # col_contrib[j] = sum_i a[i,j] * B_perm[i,j]
    
    # Note: current_cost = sum(row_contrib) = sum(col_contrib)
    # But row_contrib[i] = sum_j a[i,j] B_perm[i,j]
    # And the total cost = sum_i row_contrib[i]
    
    # For swapping p[i] and p[j] (let x = p[i], y = p[j]):
    # New B_perm rows/cols change.
    # Let's compute the delta directly.
    
    # Old contributions involving i or j:
    # From row i: sum_k a[i,k] B_perm[i,k] = row_contrib[i]
    # From col i: sum_k a[k,i] B_perm[k,i] = col_contrib[i]
    # From row j: sum_k a[j,k] B_perm[j,k] = row_contrib[j]
    # From col j: sum_k a[k,j] B_perm[k,j] = col_contrib[j]
    # But the terms a[i,i]B_perm[i,i], a[i,j]B_perm[i,j], a[j,i]B_perm[j,i], a[j,j]B_perm[j,j] 
    # are counted in both row and col contributions.
    
    # This is getting complex. Let's use a simpler approach: 
    # Just do random swaps with full cost re-evaluation, but limit to a reasonable number.
    
    no_improve_local = 0
    max_no_improve_local = 200
    
    while time.time() < deadline:
        if no_improve_local > max_no_improve_local:
            # Perturb: swap 3 random pairs
            for _ in range(3):
                i, j = np.random.choice(n, 2, replace=False)
                current_p[i], current_p[j] = current_p[j], current_p[i]
            current_cost = compute_cost(current_p)
            no_improve_local = 0
            continue
        
        i, j = np.random.choice(n, 2, replace=False)
        
        # Swap
        current_p[i], current_p[j] = current_p[j], current_p[i]
        
        new_cost = compute_cost(current_p)
        
        if new_cost < current_cost:
            current_cost = new_cost
            no_improve_local = 0
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = current_p.copy()
        else:
            # Revert
            current_p[i], current_p[j] = current_p[j], current_p[i]
            no_improve_local += 1
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
