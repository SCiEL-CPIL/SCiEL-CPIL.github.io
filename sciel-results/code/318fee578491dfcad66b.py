import numpy as np
from scipy.optimize import minimize

N = 100

def _objective_and_grad(x):
    """
    Objective: maximize the minimum of C_h(k) = sum_i x_i * x_{i+k} for k=1..N-1,
    subject to 0 <= x_i <= 1 and sum x_i = N/2.
    We minimize the negative of the minimum, i.e., maximize min C_h(k).
    """
    # Compute autocorrelations for lags 1 to N-1
    # C_k = sum_{i=0}^{N-1-k} x[i] * x[i+k]
    C = np.zeros(N - 1)
    for k in range(1, N):
        C[k - 1] = np.dot(x[:N - k], x[k:])
    
    min_C = np.min(C)
    
    # Gradient: d/dx_j of min_C
    # For the lag k* that achieves the minimum, gradient w.r.t. x_j:
    # d C_{k*}/d x_j = x_{j+k*} (if j < N-k*) + x_{j-k*} (if j >= k*)
    k_star = np.argmin(C) + 1
    grad = np.zeros(N)
    # Contribution from x[j] * x[j+k*]
    idx1 = np.arange(N - k_star)
    grad[idx1] += x[idx1 + k_star]
    # Contribution from x[j-k*] * x[j] (when j >= k*)
    idx2 = np.arange(k_star, N)
    grad[idx2] += x[idx2 - k_star]
    
    return -min_C, grad

def construct():
    # Start from a non-trivial initial guess to escape potential symmetry
    # Use a sinusoidal perturbation around 0.5
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = 0.5 + 0.1 * np.sin(t)
    
    # Enforce sum constraint: adjust by a constant
    x0 = x0 - np.mean(x0) + 0.5
    
    # Bound constraints
    bounds = [(0.0, 1.0)] * N
    
    # Equality constraint: sum x_i = N/2
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}]
    
    # Use L-BFGS-B with gradient
    res = minimize(
        _objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    x_opt = res.x
    
    # Final projection to ensure exact sum and bounds
    # Use a simple water-filling / clipping approach for exact sum
    for _ in range(100):
        # Clip to [0, 1]
        x_opt = np.clip(x_opt, 0.0, 1.0)
        # Adjust to meet sum constraint
        diff = N / 2.0 - np.sum(x_opt)
        if abs(diff) < 1e-12:
            break
        # Add/subtract uniformly, then re-clip
        x_opt += diff / N
    x_opt = np.clip(x_opt, 0.0, 1.0)
    
    # Final fine-tune: distribute remaining error to non-saturated elements
    diff = N / 2.0 - np.sum(x_opt)
    if abs(diff) > 1e-12:
        non_sat = (x_opt > 1e-6) & (x_opt < 1.0 - 1e-6)
        if np.any(non_sat):
            x_opt[non_sat] += diff / np.sum(non_sat)
        else:
            # Fallback: distribute to all
            x_opt += diff / N
            x_opt = np.clip(x_opt, 0.0, 1.0)
    
    return x_opt
