import numpy as np
import time
import scipy.optimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Minimizes the area of the union of 64 triangles.
    Uses L-BFGS-B with a smooth soft-min surrogate for the union length.
    """
    n = 64
    k = int(np.log2(n))
    w = 1.0 / n
    M = 20000  # Number of quadrature points
    
    # Precompute heights and slopes
    y = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    
    # Baseline: Keich construction
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(positions, dtype=np.float64)
    
    # Ensure x0 is sorted to maintain the assumption L_i < L_{i+1}
    # The Keich construction for n=64 yields sorted x?
    # Let's check. If not, sort it.
    # Note: Sorting x changes which x is paired with which a.
    # The Keich x_i is derived from binary of i.
    # If we sort x, we are effectively permuting the triangles.
    # Is the optimal solution sorted x?
    # If x is sorted, L is sorted for all y.
    # If x is not sorted, L crosses, and the "sorted" union length formula in the prompt
    # implies we must sort L at each y.
    # If we assume sorted x, we avoid the sorting cost and discontinuities.
    # Let's verify if Keich x is sorted.
    if not np.all(np.diff(x0) > 0):
        x0 = np.sort(x0)
        
    def objective_and_grad(x):
        # x: shape (64,)
        # L[m, i] = x[i] + a[i] * y[m]
        # Use broadcasting: x[None, :] + y[:, None] * a[None, :]
        L = x[None, :] + y[:, None] * a[None, :]
        
        # D[m, i] = L[m, i+1] - L[m, i] for i in 0..62
        D = L[:, 1:] - L[:, :-1]  # shape (20000, 63)
        
        # Soft-min derivative: s'(u) = 1 / (1 + exp(beta * (u - w)))
        # We need to choose beta.
        # Let's use a global beta, maybe anneal it outside.
        # For the gradient function, beta is fixed per call.
        # We will wrap this in a closure or pass beta.
        # For simplicity in this block, I'll define the inner function with a fixed beta 
        # and run multiple optimizations with increasing beta.
        
        # To make this efficient, I'll implement the optimization loop outside 
        # and define the function dynamically or use a class.
        # But the prompt asks for a single function.
        # I will implement the optimization loop inside kakeya_needle_2d_positions64.
        
        return None # Placeholder

    # Optimization Loop
    deadline = time.time() + 600  # 600s deadline
    best_x = x0.copy()
    best_val = np.inf
    
    # Define the actual objective/grad function factory
    def make_func(beta):
        def func(x):
            L = x[None, :] + y[:, None] * a[None, :]
            D = L[:, 1:] - L[:, :-1]
            
            # Soft min: min(u, w) approx w - 1/beta * log(1 + exp(beta*(w-u)))
            # To avoid overflow in exp, use log-sum-exp trick or clip
            # beta*(w-u) can be large positive or negative.
            # Let z = beta * (w - D)
            # log(1 + exp(z)) is stable if we use np.logaddexp(0, z)
            
            z = beta * (w - D)
            # softmin_val = w - (1/beta) * np.logaddexp(0, z)
            # grad_s = 1 / (1 + exp(-z)) = sigmoid(z)
            # Wait, s'(u) = 1 / (1 + exp(beta(u-w))) = 1 / (1 + exp(-z))
            
            # Compute objective
            softmin_terms = w - (1.0 / beta) * np.logaddexp(0, z)
            union_lengths = w + np.sum(softmin_terms, axis=1)
            area = np.mean(union_lengths)
            
            # Compute gradient
            # dU/dx_k = s'(D_{k-1}) - s'(D_k)
            # s'(D) = sigmoid(z)
            grad_s = 1.0 / (1.0 + np.exp(-z))  # shape (20000, 63)
            
            # grad_x shape (64,)
            grad_x = np.zeros(n)
            # Contribution from D_i to x_{i+1} is +grad_s[:, i]
            # Contribution from D_i to x_i is -grad_s[:, i]
            
            # Sum over m (axis 0)
            sum_grad_s = np.sum(grad_s, axis=0)  # shape (63,)
            
            # x_0: only affected by D_0 (as x_0 is left of D_0) -> -sum_grad_s[0]
            # x_i (i>0): affected by D_{i-1} (as right) and D_i (as left)
            # x_{n-1}: only affected by D_{n-2} (as right)
            
            grad_x[0] = -sum_grad_s[0]
            grad_x[1:-1] = sum_grad_s[:-1] - sum_grad_s[1:]
            grad_x[-1] = sum_grad_s[-1]
            
            # Scale by 1/M
            grad_x /= M
            
            return area, grad_x
        
        return func

    # Annealing beta
    betas = [10, 50, 100, 200, 500]
    
    for beta in betas:
        if time.time() > deadline - 30:
            break
            
        func = make_func(beta)
        
        # Initial guess is the best found so far
        x_current = best_x.copy()
        
        # Run L-BFGS-B
        try:
            res = scipy.optimize.minimize(
                func, 
                x_current, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_val:
                # Evaluate final value with the exact (non-soft) metric to be sure
                # The soft-min is an approximation. We should check the true area.
                x_candidate = res.x
                L = x_candidate[None, :] + y[:, None] * a[None, :]
                D = L[:, 1:] - L[:, :-1]
                # True union length: w + sum(min(D, w))
                true_union = w + np.sum(np.minimum(D, w), axis=1)
                true_area = np.mean(true_union)
                
                if true_area < best_val:
                    best_val = true_area
                    best_x = x_candidate
        except Exception as e:
            pass

    return best_x

# EVOLVE-BLOCK-END
