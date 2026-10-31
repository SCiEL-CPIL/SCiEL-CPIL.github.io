import numpy as np
from scipy.optimize import minimize

N = 100

def _overlap_at_shift(h, k):
    """Compute sum_i h[i] * (1 - h[(i+k) % N]) for a given shift k."""
    shifted = np.roll(h, -k)
    return np.sum(h * (1.0 - shifted))

def _max_overlap(h):
    """Compute the maximum overlap over all non-zero shifts."""
    h = np.asarray(h)
    n = len(h)
    # Normalize to ensure sum is n/2 (defensive)
    s = np.sum(h)
    if abs(s - n / 2.0) > 1e-10:
        h = h * (n / 2.0) / s
    max_val = 0.0
    for k in range(1, n):
        val = _overlap_at_shift(h, k)
        if val > max_val:
            max_val = val
    return max_val

def _objective(x):
    """Objective function for the optimizer: maximize the minimum of h(1-h_shifted).
    We want to MINIMIZE the maximum overlap, which is equivalent to
    MINIMIZING max_k sum h_i(1-h_{i+k}).
    
    Alternatively, we can maximize the minimum of the "good" part.
    Let's stick to minimizing the max overlap directly.
    """
    h = np.clip(x, 0.0, 1.0)
    return _max_overlap(h)

def _objective_grad(x):
    """Gradient of the objective function.
    Since _max_overlap is a max of linear functions, it is convex.
    We can use subgradient, but for smoothness in optimization,
    we can use a softmax approximation or just use the subgradient of the active set.
    For simplicity and robustness, let's use numerical gradient or a smooth approximation.
    Actually, scipy's L-BFGS-B doesn't support non-smooth functions well.
    Let's use a smooth approximation: max_k f_k(x) ~ log-sum-exp.
    """
    h = np.clip(x, 0.0, 1.0)
    n = len(h)
    vals = np.array([_overlap_at_shift(h, k) for k in range(1, n)])
    # Softmax smoothing
    beta = 50.0  # Higher beta makes it closer to max
    max_val = np.max(vals)
    exp_vals = np.exp(beta * (vals - max_val))
    obj = (max_val + np.log(np.sum(exp_vals)) / beta)
    
    # Gradient computation (numerical for simplicity, or analytical)
    # Let's compute gradient analytically for the smoothed objective.
    # d/dx_i [ log sum_k exp(beta * f_k(x)) / beta + max_val ]
    # = (1/beta) * sum_k [ beta * exp(beta f_k) / sum exp(beta f_k) * df_k/dx_i ]
    # = sum_k [ w_k * df_k/dx_i ]
    # where w_k = exp(beta f_k) / sum exp(beta f_k)
    
    weights = exp_vals / np.sum(exp_vals)
    grad = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1]
        if w > 1e-8:
            shifted = np.roll(h, -k)
            # f_k = sum_i h_i (1 - h_{i+k})
            # df_k/dx_i = (1 - h_{i+k}) - h_{i-k} (due to symmetry in sum)
            # Wait, f_k = sum_i h_i - h_i h_{i+k}
            # df_k/dx_i = 1 - (h_{i+k} + h_i * 0) ... no.
            # d/dx_i [ h_i (1 - h_{i+k}) ] = (1 - h_{i+k})
            # d/dx_i [ - h_j (1 - h_{j+k}) ] for j != i is 0.
            # But h_{i-k} appears as h_j where j+k = i => j = i-k.
            # So term -h_{i-k}(1-h_i) also depends on x_i.
            # df_k/dx_i = (1 - h_{i+k}) - h_{i-k}
            
            d_h = (1.0 - shifted) - np.roll(h, k)
            grad += w * d_h
    return obj, grad

def construct():
    # Start with a symmetric initial guess
    # The golden ratio split is a good start
    c = (np.sqrt(5) - 1) / 2
    h0 = np.zeros(N)
    h0[:N//2] = c
    h0[N//2:] = 1 - c
    
    # Ensure sum is exactly N/2
    h0 = h0 * (N / 2.0) / np.sum(h0)
    
    # Optimize
    res = minimize(
        _objective_grad,
        h0,
        method='L-BFGS-B',
        bounds=[(0.0, 1.0)] * N,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
    )
    
    h_opt = res.x
    # Clip and renormalize
    h_opt = np.clip(h_opt, 0.0, 1.0)
    s = np.sum(h_opt)
    if s > 0:
        h_opt = h_opt * (N / 2.0) / s
    
    return h_opt
