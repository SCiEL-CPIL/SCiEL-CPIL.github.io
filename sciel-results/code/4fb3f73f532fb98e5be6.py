"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Constructs 8 points in the plane maximizing the product of pairwise distances
    with diameter 2.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    # Function to generate points for two concentric squares
    def generate_points(r1, r2, alpha):
        # Square 1 angles: 0, pi/2, pi, 3pi/2
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        # Square 2 angles: alpha, alpha + pi/2, ...
        angles2 = angles1 + alpha
        
        pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
        pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
        
        return np.vstack((pts1, pts2))

    # Function to compute the log of the product of ordered pair distances
    # scaled to diameter 2
    def log_score(pts):
        # Compute pairwise distances
        # pts is 8x2
        # dist[i, j] = || pts[i] - pts[j] ||
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        
        # Diameter is max distance
        D = np.max(dists)
        
        if D == 0:
            return -np.inf
            
        # Scale to diameter 2
        scale = 2.0 / D
        dists_scaled = dists * scale
        
        # Product over ordered pairs (i != j)
        # Mask diagonal to avoid log(0)
        mask = ~np.eye(8, dtype=bool)
        log_prod = np.sum(np.log(dists_scaled[mask]))
        
        return log_prod

    # Objective function to minimize (negative log score)
    def objective(params):
        r1, r2, alpha = params
        # Ensure positive radii
        if r1 < 1e-6 or r2 < 1e-6:
            return 1e10
        pts = generate_points(r1, r2, alpha)
        score = log_score(pts)
        return -score

    best_score = -np.inf
    best_params = None
    
    # Multi-start optimization
    # Parameters: r1, r2, alpha
    # Bounds: r1, r2 in [0.1, 3.0], alpha in [0, pi/2]
    # Note: The relative scale is what matters, but absolute scale is normalized in log_score.
    # So we can search in a reasonable box.
    
    starts = [
        [1.0, 1.0, np.pi/4],       # Regular octagon
        [1.2, 0.8, np.pi/4],       # Perturbed radii
        [0.8, 1.2, np.pi/4],       # Inverse perturbed radii
        [1.0, 1.0, np.pi/3],       # Perturbed angle
        [1.0, 1.0, np.pi/6],       # Perturbed angle
        [1.5, 0.5, np.pi/4],       # Strong radius perturbation
        [0.5, 1.5, np.pi/4],       # Strong radius perturbation
        [1.1, 0.9, np.pi/5],       # Mixed perturbation
    ]

    for start in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, start, method='L-BFGS-B', bounds=[(0.1, 3.0), (0.1, 3.0), (0.0, np.pi/2)])
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x
        except Exception:
            pass

    # If no valid optimization found, fall back to regular octagon
    if best_params is None:
        best_params = [1.0, 1.0, np.pi/4]

    # Generate final points
    pts = generate_points(*best_params)
    
    # Final scaling to ensure diameter is exactly 2 (handling floating point errors)
    dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    D = np.max(dists)
    if D > 0:
        pts = pts * (2.0 / D)
        
    return pts.tolist()

# EVOLVE-BLOCK-END
