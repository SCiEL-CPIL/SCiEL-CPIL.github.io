# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin for 600s limit

    def normalize(points):
        """
        Center and scale points to prevent numerical drift.
        This keeps the configuration in a standard frame, which helps the optimizer.
        """
        pts = points.copy()
        # Center the configuration
        pts -= np.mean(pts, axis=0)
        # Scale to unit norm (RMS or L2 norm of the flattened array)
        norm = np.linalg.norm(pts)
        if norm > 1e-12:
            pts /= norm
        return pts

    def get_score(points):
        """Calculate the score R = (d_min / d_max)^2"""
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x):
        """
        Objective function for minimization.
        Minimizes log(d_max) - log(d_min).
        """
        points = x.reshape(n, d)
        # Normalize to keep search space compact and stable
        points = normalize(points)
        dists = pdist(points)
        if len(dists) == 0:
            return 1e10
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min < 1e-12 or d_max < 1e-12:
            return 1e10
        return np.log(d_max) - np.log(d_min)

    def local_search(points, max_iters=5000):
        """
        Local search with adaptive step size.
        """
        best_points = normalize(points.copy())
        best_score = get_score(best_points)
        step_size = 0.05
        stagnant = 0
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            # Pick a random point to perturb
            j = np.random.randint(n)
            
            # Try several perturbations for this point
            for _ in range(10):
                perturbation = np.random.normal(0, step_size, size=d)
                candidate = best_points.copy()
                candidate[j] += perturbation
                candidate = normalize(candidate)
                
                new_score = get_score(candidate)
                if new_score > best_score:
                    best_score = new_score
                    best_points = candidate
                    improved = True
                    break # Move to next iteration
            
            if not improved:
                stagnant += 1
                if stagnant > 20:
                    step_size *= 0.9
                    stagnant = 0
                    if step_size < 1e-8:
                        break
            else:
                stagnant = 0
                
        return best_points, best_score

    # Generate diverse starting points
    starts = []
    
    # 1. 4x4 Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    starts.append(np.column_stack([xx.ravel(), yy.ravel()]))
    
    # 2. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    starts.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 3. Hexagonal cluster approximation (16 closest to origin)
    hex_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x_h = i + 0.5 * j
            y_h = np.sqrt(3) / 2 * j
            hex_points.append([x_h, y_h])
    hex_points = np.array(hex_points)
    dists_center = np.sum(hex_points**2, axis=1)
    indices = np.argsort(dists_center)
    starts.append(hex_points[indices[:n]])
    
    # 4. Random starts
    np.random.seed(42)
    for _ in range(5):
        starts.append(np.random.randn(n, d))
        
    # 5. Two rings (8 and 8)
    angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    p1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    angles2 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    p2 = np.column_stack([1.5 * np.cos(angles2), 1.5 * np.sin(angles2)])
    starts.append(np.vstack([p1, p2]))
    
    # 6. Concentric squares (4 inner, 12 outer? No, 4+8+4=16)
    # Inner: (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)
    p_inner = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])
    # Outer: Corners (1.5, 1.5) etc and midpoints (1.5, 0) etc
    p_outer = np.array([
        [1.5, 1.5], [-1.5, 1.5], [-1.5, -1.5], [1.5, -1.5], # 4 corners
        [1.5, 0], [-1.5, 0], [0, 1.5], [0, -1.5], # 4 midpoints
        [0.5, 1.5], [-0.5, 1.5], [-0.5, -1.5], [0.5, -1.5], # 4 more?
        [1.5, 0.5], [-1.5, 0.5], [1.5, -0.5], [-1.5, -0.5]  # 4 more?
    ])
    # This is 4 + 12 = 16.
    starts.append(np.vstack([p_inner, p_outer]))

    best_score = -1.0
    best_points = starts[0].copy()

    for i, start in enumerate(starts):
        if time.time() > deadline - 30:
            break
            
        try:
            # Phase 1: Powell optimization (robust for non-smooth)
            res = minimize(objective, start.ravel(), method='Powell', 
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            opt_points = res.x.reshape(n, d)
            opt_points = normalize(opt_points)
            
            s = get_score(opt_points)
            if s > best_score:
                best_score = s
                best_points = opt_points.copy()
                
            # Phase 2: Local search refinement
            if time.time() < deadline - 10:
                np.random.seed(1000 + i)
                refined_points, refined_score = local_search(opt_points, max_iters=3000)
                if refined_score > best_score:
                    best_score = refined_score
                    best_points = refined_points.copy()
                    
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
