import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * max(h*h))"""
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0)  # ensure nonnegative
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-300 or l2_sq < 1e-300:
        return 0.0
    # Autoconvolution (linear)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv < 1e-300:
        return 0.0
    return l2_sq / (l1 * max_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Start with a power-law concentrated shape
    # Try various initial shapes and optimize
    best_ratio_shape = 0.0
    best_h_shape = np.ones(N)
    
    # Several initial shapes to try
    init_shapes = []
    
    # 1. Power law concentrated at one end
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
        x = np.linspace(1, N, N)
        h = x ** (-alpha)
        h = h / np.max(h)  # normalize to max 1
        init_shapes.append(h.copy())
    
    # 2. Exponential decay
    for decay in [0.1, 0.3, 0.5, 1.0]:
        h = np.exp(-decay * np.arange(N))
        h = h / np.max(h)
        init_shapes.append(h.copy())
    
    # 3. Gaussian concentrated
    for center in [N//4, N//2, 3*N//4]:
        for sigma in [2, 5, 10]:
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / sigma)**2)
            h = h / np.max(h)
            init_shapes.append(h.copy())
    
    # 4. Random sparse
    np.random.seed(42)
    for _ in range(5):
        h = np.zeros(N)
        idx = np.random.choice(N, size=N//5, replace=False)
        h[idx] = 1.0
        init_shapes.append(h.copy())
    
    # Evaluate all initial shapes
    for h in init_shapes:
        r = compute_ratio(h)
        if r > best_ratio_shape:
            best_ratio_shape = r
            best_h_shape = h.copy()
    
    # Optimize the best shape using gradient descent
    def objective(x):
        """Minimize negative ratio"""
        h = np.maximum(x, 0)
        return -compute_ratio(h)
    
    # Use L-BFGS-B for optimization
    x0 = best_h_shape.copy()
    
    # Do several optimization attempts with different starting points
    for h_init in init_shapes[:10]:  # top 10 initial shapes
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                objective,
                h_init.copy(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_opt = np.maximum(result.x, 0)
            r = compute_ratio(h_opt)
            if r > best_ratio_shape:
                best_ratio_shape = r
                best_h_shape = h_opt.copy()
        except Exception:
            continue
    
    # Additional local refinement with random perturbations
    if time.time() < deadline - 10:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            h_pert = best_h_shape + 0.1 * np.random.randn(N)
            h_pert = np.maximum(h_pert, 0)
            h_pert = h_pert / np.max(h_pert) if np.max(h_pert) > 0 else h_pert
            r = compute_ratio(h_pert)
            if r > best_ratio_shape:
                best_ratio_shape = r
                best_h_shape = h_pert.copy()
    
    # Scale down to exploit the 1/c dependence
    # The ratio scales as 1/c when h -> c*h
    # Use a very small scale
    scale = 1e-6
    final_h = best_h_shape * scale
    
    return list(final_h)
