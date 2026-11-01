import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def _pairwise_distances_squared(X):
    """Compute squared pairwise distances for N points in D dimensions."""
    # X is (N, D)
    # Use broadcasting to compute squared distances
    # d^2 = |x_i|^2 + |x_j|^2 - 2 x_i . x_j
    x2 = np.sum(X**2, axis=1)
    dist_sq = x2[:, None] + x2[None, :] - 2 * X @ X.T
    # Ensure non-negative due to floating point errors
    np.maximum(dist_sq, 0, out=dist_sq)
    return dist_sq

def _objective_and_grad(X, tau):
    """
    Compute the smooth objective function and its gradient.
    Objective: log(sum(exp(d^2/tau))) - log(sum(exp(-d^2/tau)))
    which approximates log(max_d) - log(min_d).
    """
    dist_sq = _pairwise_distances_squared(X)
    
    # Mask diagonal (self-distances are 0, we ignore them)
    mask = ~np.eye(N, dtype=bool)
    d2 = dist_sq[mask]
    
    # Soft max term: log(sum(exp(d2/tau)))
    # For numerical stability, subtract max
    max_d2 = np.max(d2)
    exp_max = np.exp(d2 / tau - max_d2 / tau)
    log_sum_max = np.log(np.sum(exp_max)) + max_d2 / tau
    
    # Soft min term: -log(sum(exp(-d2/tau)))
    # This is equivalent to log(1 / sum(exp(-d2/tau)))
    # Let's compute log_sum_min = log(sum(exp(-d2/tau)))
    # To avoid underflow, we can just use the formula directly if tau is not too small,
    # or use the same trick.
    # Actually, softmin(d2) = -softmax(-d2).
    # log(sum(exp(-d2/tau)))
    min_d2 = np.min(d2)
    exp_min = np.exp(-d2 / tau + min_d2 / tau)
    log_sum_min = np.log(np.sum(exp_min)) - min_d2 / tau
    
    obj = log_sum_max - log_sum_min
    
    # Gradient
    # d_obj/dX = (1/tau) * ( W_max - W_min ) @ (X - X) structure?
    # Let's derive carefully.
    # F = log(sum_{i<j} exp(d_{ij}^2/tau)) - log(sum_{i<j} exp(-d_{ij}^2/tau))
    # dF/dx_k = (1/tau) [ sum_{j != k} W^max_{kj} * d(d_{kj}^2)/dx_k - sum_{j != k} W^min_{kj} * d(d_{kj}^2)/dx_k ]
    # where W^max_{kj} = exp(d_{kj}^2/tau) / sum_{i<j} exp(d_{ij}^2/tau)
    # and W^min_{kj} = exp(-d_{kj}^2/tau) / sum_{i<j} exp(-d_{ij}^2/tau)
    # d(d_{kj}^2)/dx_k = 2(x_k - x_j)
    
    # Reconstruct weights matrices
    # exp_max_full is not directly available, we have exp_max for the flattened vector.
    # We need the full matrix to sum over j for each k.
    
    # Recalculate exp terms for full matrix
    # d2_full = dist_sq
    # We need to be careful with the diagonal.
    
    # Let's compute the gradient via the full matrix approach for clarity and correctness.
    # exp_max_full = exp(d2_full / tau - max_d2/tau)
    # sum_exp_max = sum(exp_max_full[mask])
    # W_max = exp_max_full / sum_exp_max
    # W_max[diag] = 0
    
    # exp_min_full = exp(-d2_full / tau + min_d2/tau)
    # sum_exp_min = sum(exp_min_full[mask])
    # W_min = exp_min_full / sum_exp_min
    # W_min[diag] = 0
    
    # grad = (2/tau) * ( W_max - W_min ) @ (X - X) ? No.
    # grad_k = (2/tau) * sum_j (W_max[k,j] - W_min[k,j]) * (x_k - x_j)
    # This can be written as:
    # grad = (2/tau) * ( (W_max - W_min) @ X - (W_max - W_min) @ X? No.)
    # Let A = W_max - W_min. A is symmetric.
    # grad_k = (2/tau) * sum_j A[k,j] (x_k - x_j)
    # grad_k = (2/tau) * ( x_k * sum_j A[k,j] - sum_j A[k,j] x_j )
    # Let row_sum = A.sum(axis=1)
    # grad = (2/tau) * ( row_sum[:, None] * X - A @ X )
    
    exp_max_full = np.exp(dist_sq / tau - max_d2 / tau)
    sum_exp_max = np.sum(exp_max_full[mask])
    W_max = exp_max_full / sum_exp_max
    W_max[~mask] = 0 # Zero out diagonal
    
    exp_min_full = np.exp(-dist_sq / tau + min_d2 / tau)
    sum_exp_min = np.sum(exp_min_full[mask])
    W_min = exp_min_full / sum_exp_min
    W_min[~mask] = 0 # Zero out diagonal
    
    A = W_max - W_min
    row_sum = A.sum(axis=1)
    grad = (2.0 / tau) * (row_sum[:, None] * X - A @ X)
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Initial guess: Random points on a sphere
    # Or a regular polytope if possible
    # For N=14, random sphere is a good start
    np.random.seed(42)
    X0 = np.random.randn(N, D)
    X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
    
    best_X = X0
    best_score = np.inf
    
    # Annealing schedule for tau
    # Start with large tau (smooth) and decrease to small tau (sharp)
    taus = [1.0, 0.5, 0.1, 0.01, 0.001]
    
    for tau in taus:
        if time.time() > deadline:
            break
            
        try:
            # Use L-BFGS-B
            # We need to flatten X for scipy
            x_flat = X0.flatten()
            
            def fun_grad_flat(xf):
                X = xf.reshape(N, D)
                obj, grad = _objective_and_grad(X, tau)
                return obj, grad.flatten()
                
            res = minimize(fun_grad_flat, x_flat, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < best_score:
                X_new = res.x.reshape(N, D)
                # Calculate actual score for tracking
                dists = np.sqrt(_pairwise_distances_squared(X_new))
                mask = ~np.eye(N, dtype=bool)
                d_pairs = dists[mask]
                if len(d_pairs) > 0:
                    min_d = np.min(d_pairs)
                    max_d = np.max(d_pairs)
                    if min_d > 1e-10:
                        score = (max_d / min_d) ** 2
                        if score < best_score:
                            best_score = score
                            best_X = X_new
                # Update X0 for next iteration
                X0 = X_new
        except Exception as e:
            print(f"Optimization error: {e}")
            # Continue with next tau if time permits
    
    # Final check and return
    # Ensure distinct points (they should be, but just in case)
    return best_X
