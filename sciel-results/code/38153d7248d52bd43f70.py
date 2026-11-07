import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_value(h):
    """
    Computes the value for a given height vector h.
    Returns (value, gradient).
    h is assumed to be non-negative.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        h = np.clip(h, 0, None)
    
    h_sum = np.sum(h)
    if h_sum == 0:
        return 0.0, np.zeros_like(h)
        
    g = np.convolve(h, h, mode='full')
    m = len(g)
    
    # y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    # Spacing w
    w = 1.0 / (m + 1)
    
    # l2 = sum_{i=0}^m (w/3) * (y[i]^2 + y[i]y[i+1] + y[i+1]^2)
    y_curr = y[:m+1]
    y_next = y[1:m+2]
    terms = y_curr**2 + y_curr * y_next + y_next**2
    S = np.sum(terms)
    l2 = (w / 3.0) * S
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return 0.0, np.zeros_like(h)
        
    value = l2 / (norm_1 * norm_inf)
    
    # Analytic Gradient Calculation
    # Objective: F(h) = L2(h) / (N1(h) * Ninf(h))
    # Note: Since we are optimizing on the simplex (sum h = 1) using log-transform,
    # we need the gradient of the log-objective or just the raw gradient projected.
    # Let's compute dF/dh.
    # F = L2 * (N1 * Ninf)^-1
    # dF = dL2 * (N1 Ninf)^-1 - L2 * (N1 Ninf)^-2 * d(N1 Ninf)
    # d(N1 Ninf) = Ninf dN1 + N1 dNinf
    
    # 1. Gradient of L2
    # L2 = (w/3) * sum_i (y_i^2 + y_i y_{i+1} + y_{i+1}^2)
    # Let Q(y) = sum_i (y_i^2 + y_i y_{i+1} + y_{i+1}^2)
    # dQ/dy_i = 2y_i + y_{i-1} + y_{i+1} (with y_{-1}=0, y_{m+1}=0)
    # y = P g, where P is the padding matrix. P is linear.
    # dL2/dg = P^T (2y + shift(y) + shift_back(y)) * (w/3)
    # Let's compute dQ/dy vectorially.
    # y has length m+2. g has length m.
    # y[0]=0, y[m+1]=0.
    # dQ/dy[0] = 2y[0] + y[1] = y[1]
    # dQ/dy[m+1] = 2y[m+1] + y[m] = y[m]
    # dQ/dy[i] for 0<i<m+1 = 2y[i] + y[i-1] + y[i+1]
    
    dQ_dy = np.zeros_like(y)
    dQ_dy[1:-1] = 2 * y[1:-1] + y[:-2] + y[2:]
    dQ_dy[0] = y[1]
    dQ_dy[-1] = y[-2]
    
    # dL2/dy = (w/3) * dQ_dy
    # dL2/dg = dL2/dy[1:-1] because y[1:-1] = g
    dL2_dg = (w / 3.0) * dQ_dy[1:-1]
    
    # 2. Gradient of N1
    # N1 = sum(|g|)/(m+1). Since h>=0, g>=0.
    # dN1/dg = ones(m) / (m+1)
    dN1_dg = np.ones(m) / (m + 1)
    
    # 3. Gradient of Ninf
    # Ninf = max(g).
    # Subgradient is 1 at argmax, 0 elsewhere.
    # For smooth optimization, we can use a soft-max approximation or just the hard max.
    # Hard max is non-differentiable but works okay with L-BFGS-B if we handle it carefully,
    # or we can use the index of the max.
    idx_max = np.argmax(g)
    dNinf_dg = np.zeros(m)
    dNinf_dg[idx_max] = 1.0
    
    # Combine gradients w.r.t g
    # dF/dg = dL2_dg / (N1 Ninf) - L2 * (Ninf dN1_dg + N1 dNinf_dg) / (N1 Ninf)^2
    denom = N1 * Ninf
    dF_dg = dL2_dg / denom - value * (Ninf * dN1_dg + N1 * dNinf_dg) / denom
    
    # 4. Gradient w.r.t h
    # g = conv(h, h)
    # g_k = sum_j h_j h_{k-j}
    # d g_k / d h_i = h_{k-i} + h_{i-k} (if indices valid, else 0)
    # This is equivalent to d g / d h_i = conv(h, e_i) + conv(e_i, h) = h shifted + h shifted back.
    # Specifically, if we define the operator such that g = h * h (convolution),
    # then dF/dh_i = sum_k (dF/dg_k) * (d g_k / d h_i)
    # d g_k / d h_i is non-zero only if k-i is in [0, N-1] or i-k is in [0, N-1].
    # Let's use the property: sum_k (dF/dg_k) h_{k-i} = conv(dF_dg, h)_i
    # And sum_k (dF/dg_k) h_{i-k} = conv(dF_dg, h)_{i} ??
    # Let's check indices.
    # g_k = sum_{j=0}^{N-1} h_j h_{k-j}
    # d g_k / d h_i = h_{k-i} (if 0<=k-i<N) + h_{i-k} (if 0<=i-k<N, i!=k)
    # Actually, since h is symmetric in the convolution product term h_j h_{k-j},
    # d/dh_i (h_i h_{k-i} + h_{k-i} h_i) = 2 h_{k-i} if i != k-i.
    # If i == k-i (k=2i), term is h_i^2, deriv is 2 h_i.
    # So effectively d g_k / d h_i = 2 * h_{k-i} (with appropriate boundary handling).
    # Wait, convolution is commutative.
    # Let's just compute the gradient via matrix-free multiplication or explicit loop if N is small.
    # N=50 is small enough for a simple loop or vectorized trick.
    # dF_dh_i = sum_k dF_dg_k * (h_{k-i} + h_{i-k})
    # This is 2 * conv(dF_dg, h) but careful with boundaries?
    # conv(dF_dg, h)[i] = sum_k dF_dg[k] h[i-k].
    # We want sum_k dF_dg[k] h[k-i]. This is cross-correlation or flipped convolution.
    # Let's just do it explicitly to be safe, N=50, m=99. 50*100 = 5000 ops. Trivial.
    
    dF_dh = np.zeros(N)
    for i in range(N):
        grad_i = 0.0
        for k in range(m):
            # Term 1: h_{k-i}
            if 0 <= k - i < N:
                grad_i += dF_dg[k] * h[k - i]
            # Term 2: h_{i-k}
            if 0 <= i - k < N:
                grad_i += dF_dg[k] * h[i - k]
        dF_dh[i] = grad_i
        
    return value, dF_dh

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate diverse initial candidates on the simplex
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N) / N)
    
    # Gaussians
    for width in [1, 2, 5, 10, 20]:
        for center in [5, 15, 25, 35, 45]:
            x = np.arange(N)
            h = np.exp(-((x - center) ** 2) / (2 * width ** 2))
            h_sum = np.sum(h)
            if h_sum > 0:
                candidates.append(h / h_sum)
                
    # Boxcars
    for start in range(0, N, 5):
        for length in [5, 10, 20, 50]:
            h = np.zeros(N)
            h[start:min(start+length, N)] = 1.0
            h_sum = np.sum(h)
            if h_sum > 0:
                candidates.append(h / h_sum)
                
    # Sine waves
    for freq in [1, 2, 3]:
        x = np.linspace(0, 2*np.pi, N)
        h = np.abs(np.sin(freq * x))
        h_sum = np.sum(h)
        if h_sum > 0:
            candidates.append(h / h_sum)
            
    # Bimodal
    for d in [5, 10, 20]:
        x = np.arange(N)
        h = np.exp(-((x - 10) ** 2)) + np.exp(-((x - 40) ** 2))
        h_sum = np.sum(h)
        if h_sum > 0:
            candidates.append(h / h_sum)
            
    # Random
    for _ in range(10):
        h = np.random.rand(N)
        h_sum = np.sum(h)
        if h_sum > 0:
            candidates.append(h / h_sum)

    best_h = None
    best_score = -np.inf
    
    # 2. Optimize each candidate
    # We optimize in the log-space of the simplex to ensure h > 0 and sum h = 1.
    # Let h_i = exp(u_i) / sum(exp(u_j)).
    # We minimize -log(value) or just -value?
    # Value is not log-concave necessarily. Let's minimize -value.
    # However, L-BFGS-B works well with smooth functions.
    # The log-transform ensures positivity.
    
    for h_init in candidates:
        if time.time() > deadline:
            break
            
        # Initialize u such that exp(u)/sum(exp(u)) = h_init
        # u_i = log(h_i). Since h_i > 0, this is valid.
        # To avoid issues with log(0), we ensured h_init > 0 by construction or add tiny epsilon.
        h_safe = np.maximum(h_init, 1e-10)
        u_init = np.log(h_safe)
        
        # Objective function for optimizer
        def obj(u):
            if time.time() > deadline:
                return 0.0
            # Softmax to get h
            u_max = np.max(u)
            exp_u = np.exp(u - u_max)
            h = exp_u / np.sum(exp_u)
            
            # Compute value and gradient
            val, grad_val = compute_value(h)
            
            # Chain rule: dval/du_i = dval/dh_i * dh_i/du_i
            # h_i = exp(u_i) / S, where S = sum exp(u_j)
            # dh_i/du_i = h_i - h_i^2 = h_i (1 - h_i)
            # dh_i/du_j = -h_i h_j for i != j
            # grad_u_i = sum_j (dval/dh_j) * (dh_j/du_i)
            # = dval/dh_i * (1 - h_i) - h_i * sum_j dval/dh_j
            # Let G = sum_j dval/dh_j
            # grad_u_i = dval/dh_i - G * h_i
            
            G = np.sum(grad_val)
            grad_u = grad_val - G * h
            
            # We want to maximize value, so minimize -value
            return -val, -grad_u

        # Run L-BFGS-B
        try:
            res = minimize(obj, u_init, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            # Extract final h
            u_final = res.x
            u_max = np.max(u_final)
            exp_u = np.exp(u_final - u_max)
            h_final = exp_u / np.sum(exp_u)
            
            val_final, _ = compute_value(h_final)
            
            if val_final > best_score:
                best_score = val_final
                best_h = h_final.copy()
                
        except Exception:
            continue

    # Fallback if optimization failed or didn't find anything
    if best_h is None:
        best_h = np.ones(N) / N
        best_score, _ = compute_value(best_h)

    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
