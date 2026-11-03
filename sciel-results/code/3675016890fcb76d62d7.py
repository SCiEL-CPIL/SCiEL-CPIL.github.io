import numpy as np
from scipy.optimize import minimize
import time

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)

    # 2. Generate a dense set of target points to approximate the sphere
    # Use a Fibonacci spiral with more points for better resolution
    M = 5000
    j = np.arange(M) + 0.5
    z_t = 1.0 - 2.0 * j / M
    phi_t = np.pi * (1.0 + np.sqrt(5.0)) * j
    r_t = np.sqrt(np.maximum(0, 1.0 - z_t * z_t))
    targets = np.column_stack([r_t * np.cos(phi_t), r_t * np.sin(phi_t), z_t])
    targets = targets / np.linalg.norm(targets, axis=1, keepdims=True)

    # Objective function: Sum of squared angular distances from each target to its nearest point
    # We minimize this to spread points out and cover the sphere
    def objective(x):
        # x is a flat array of shape (N*3,)
        P = x.reshape(N, 3)
        # Normalize to project back to sphere (helps with numerical stability)
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        P = P / norms
        
        # Compute dot products: (M, N)
        dots = targets @ P.T
        # Clip for numerical safety
        dots = np.clip(dots, -1.0, 1.0)
        
        # Max dot product corresponds to min angular distance
        max_dots = np.max(dots, axis=1)
        
        # Squared angular distance approx: (arccos(max_dot))^2
        # arccos is expensive, but M=5000 is manageable.
        # Alternatively, use (1 - max_dot) as a proxy for distance squared (small angle approx)
        # But for larger distances, arccos is more accurate for "covering".
        # Let's use the proxy (1 - max_dot) for speed, as it is monotonic with angle for [0, pi]
        # Actually, covering radius is angle. Minimizing sum of (1-cos(theta)) is a good proxy.
        dists_proxy = 1.0 - max_dots
        return np.sum(dists_proxy ** 2)

    # Gradient is not provided, so we use numerical gradient (slow) or a method that doesn't need it.
    # Nelder-Mead is derivative-free but can be slow in high dimensions.
    # L-BFGS-B with numerical gradient might be okay for 390 dims?
    # Let's try a simple custom gradient descent or use L-BFGS-B with '2-point' gradient.
    
    # To speed up, we can use a smaller target set if time is tight, or reduce iterations.
    # Let's try L-BFGS-B.
    
    x0 = pts.flatten()
    
    def time_check():
        return time.time() > deadline

    try:
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
        )
        if time_check():
            return pts # Return initial if we ran out of time during optimization? No, res might be valid.
        
        x_opt = res.x
        pts_opt = x_opt.reshape(N, 3)
        pts_opt = pts_opt / np.linalg.norm(pts_opt, axis=1, keepdims=True)
        
        # Check if the optimized points are valid (non-zero, finite)
        if not np.all(np.isfinite(pts_opt)):
            return pts
            
        # Calculate the actual covering radius of the optimized points to compare with initial
        # We need a fast way to estimate covering radius for comparison.
        # The objective value is a proxy. We can just return the optimized points.
        # The evaluator will measure the true covering radius.
        
        return pts_opt

    except Exception as e:
        # If optimization fails, return the initial Fibonacci points
        return pts

# Helper to ensure we don't exceed time in the wrapper if needed, but minimize handles its own iterations.
# We added a check inside, but minimize might block. 
# To be safe, we should probably run minimize in a way that we can interrupt, 
# but standard scipy minimize doesn't support callbacks for stopping based on wall-clock easily without threading.
# However, 200 iterations of L-BFGS-B on 390 vars with a 5000x130 matmul objective should be fast (< 10s).
