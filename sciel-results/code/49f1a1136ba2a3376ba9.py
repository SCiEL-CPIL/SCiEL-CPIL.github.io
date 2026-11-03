import numpy as np
import time
from scipy.optimize import minimize

N = 600
TIME_LIMIT = 135  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use FFT for faster convolution calculation during optimization
    # Pad to next power of 2 for efficiency if needed, or just use full
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def construct():
    start_time = time.time()
    
    # 1. Define the parametric model
    # We use a cosine series: f(x) = c0 + sum_{k=1}^{K} ck * cos(2*pi*k*x)
    # Domain is [-1/4, 1/4], so x = (i + 0.5)/N - 0.25
    # This ensures f is symmetric and smooth, which often yields lower peak autoconvolution
    # than arbitrary step functions.
    
    K = 15  # Number of cosine terms (frequency components)
    
    # Grid points (centers of the intervals)
    x = (np.arange(N) + 0.5) / N - 0.25
    
    # Precompute the cosine basis matrix
    # B[i, j] = cos(2*pi*j*x_i) for j=1..K
    # B[:, 0] = 1
    freqs = np.arange(1, K+1)
    angles = 2 * np.pi * np.outer(x, freqs)
    B = np.ones((N, K+1))
    B[:, 1:] = np.cos(angles)
    
    # Initial coefficients: uniform distribution corresponds to c0=1, others 0
    # But we can start with a slightly "peaked" or "flat" distribution to help optimization
    # Let's start with uniform (c0=1) and let the optimizer find the shape.
    # Actually, a slightly non-uniform start might help escape the flat minimum if there is one,
    # but uniform is a strong candidate. Let's try a small perturbation.
    c0 = 1.0
    coeffs = np.zeros(K+1)
    coeffs[0] = c0
    
    # Objective function
    def objective(c):
        # Reconstruct f
        f = B @ c
        # Enforce non-negativity for score calculation (projection)
        f_pos = np.maximum(f, 0.0)
        s = np.sum(f_pos)
        if s < 1e-15:
            return 1e10
        # Compute autoconvolution max
        # For speed during optimization, we can use FFT
        # np.convolve is O(N^2) which is 360,000 ops, very fast.
        # FFT is O(N log N). Let's stick to np.convolve for simplicity and exactness,
        # it's fast enough for N=600.
        conv = np.convolve(f_pos, f_pos, mode='full')
        peak = np.max(conv)
        return peak / (s * s)
    
    # Gradient approximation (numerical) or analytic?
    # Analytic gradient is complex due to the max and projection.
    # L-BFGS-B can use finite differences or we can provide a numerical gradient.
    # Given N=600 and K=15, finite differences on 16 variables is cheap.
    
    # Optimization
    # We want to minimize objective(c)
    # Constraints: f(x) >= 0 is handled by projection in objective, 
    # but L-BFGS-B works best with explicit bounds or smooth objectives.
    # The projection makes the objective non-smooth.
    # Alternative: Use SLSQP or just L-BFGS-B with the projected objective.
    # Since we are minimizing a convex-ish peak metric, L-BFGS-B should be robust.
    
    # Let's run L-BFGS-B
    # We don't have explicit bounds on c, but f must be non-negative.
    # We'll rely on the projection in the objective.
    
    best_c = coeffs.copy()
    best_score = objective(best_c)
    
    # Try a few different initializations to avoid local minima
    inits = [
        np.array([1.0] + [0.0]*K),
        np.array([1.0] + [0.1]*K),
        np.array([1.0] + [0.01]*K),
        np.array([0.9] + [0.05]*K),
    ]
    
    for init in inits:
        if time.time() - start_time > TIME_LIMIT * 0.8:
            break
            
        try:
            res = minimize(objective, init, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_c = res.x
        except Exception:
            pass

    # Final reconstruction and refinement
    f = B @ best_c
    f = np.maximum(f, 0.0)
    
    # Optional: A few steps of coordinate descent on the raw vector to polish
    # This can fix any slight asymmetries or numerical issues from the parametric fit.
    for _ in range(50):
        if time.time() - start_time > TIME_LIMIT * 0.95:
            break
        improved = False
        for i in range(N):
            orig = f[i]
            # Try small adjustments
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                f[i] = max(0.0, orig + delta)
                s = np.sum(f)
                if s < 1e-15:
                    f[i] = orig
                    continue
                conv = np.convolve(f, f, mode='full')
                score = np.max(conv) / (s*s)
                if score < best_score - 1e-12:
                    best_score = score
                    improved = True
                else:
                    f[i] = orig
        if not improved:
            break
            
    return [float(x) for x in f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
