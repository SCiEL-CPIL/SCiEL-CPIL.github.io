import numpy as np
from scipy.optimize import minimize

def construct():
    N = 400
    # Initial guess: random vector normalized to sum 1
    np.random.seed(42)
    h = np.random.randn(N)
    h = h - np.mean(h) # Center it? No, we want sum 1.
    h = h / np.sum(h) if np.sum(h) != 0 else h / np.std(h)
    # Ensure sum is 1
    h = h / np.sum(h)
    
    def objective(x):
        conv = np.convolve(x, x)
        # We want to minimize max(conv)
        # To make it smooth, we can use a soft-max or just max
        # Max is non-smooth, but scipy can handle it with SLSQP or COBYLA
        # However, gradient of max is tricky.
        # Let's try minimizing the sum of squares of (conv - mean(conv)) + penalty on max?
        # Or just minimize max(conv) directly.
        return np.max(conv)
        
    def gradient(x):
        conv = np.convolve(x, x)
        max_val = np.max(conv)
        # Gradient of max is the gradient of the argmax term
        # If multiple max, average gradients
        idxs = np.where(conv == max_val)[0]
        # grad of conv_k wrt x is 2 * (convolve(x, e_k) ??)
        # d/dx_i (sum_j x_j x_{k-j}) = x_{k-i} + x_{k-i} = 2 x_{k-i} (if i != k-i)
        # Actually, grad_k = 2 * np.roll? No.
        # Let G_k be the matrix such that conv_k = x^T G_k x.
        # G_k is a symmetric matrix with 1s at (i, k-i).
        # grad_k = 2 G_k x.
        # So grad of max is sum_{k in idxs} 2 G_k x.
        
        # Efficient computation:
        # We need to add 2 * x_{k-i} for all i.
        # This is equivalent to 2 * (x * x) at specific lags?
        # Let's just compute the gradient for the active max indices.
        grad = np.zeros(N)
        for k in idxs:
            # conv_k = sum_{i=0}^{N-1} h[i] h[k-i] (with bounds)
            # d(conv_k)/d(h[j]) = h[k-j] + h[k-j] (if k-j valid)
            # Basically, for each j, add h[k-j] if k-j is in [0, N-1].
            # This is like a cross-correlation?
            # Let's use the property: grad_k = 2 * np.convolve(x, np.eye(N)[k])? No.
            # Simpler: 
            # h[k-j] is non-zero only if 0 <= k-j < N => k-N+1 <= j <= k.
            # So for j in range(max(0, k-N+1), min(N, k+1)):
            #    grad[j] += 2 * x[k-j]
            
            # Vectorized:
            # Create a vector v where v[j] = x[k-j] if valid.
            # This is just a reversed segment of x.
            start = max(0, k - N + 1)
            end = min(N, k + 1)
            # Indices j in [start, end)
            # Corresponding x indices: k - j.
            # j=start => k-start. j=end-1 => k-end+1.
            # So we take x[k-(end-1) : k-start+1] reversed?
            # Let's just loop for simplicity in the gradient step, N=400 is small.
            for j in range(start, end):
                grad[j] += 2.0 * x[k - j]
                
        return grad

    # Constraint: sum(x) = 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones(N)}]
    
    # Use SLSQP
    res = minimize(objective, h, method='SLSQP', jac=gradient, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-6})
    
    return res.x
