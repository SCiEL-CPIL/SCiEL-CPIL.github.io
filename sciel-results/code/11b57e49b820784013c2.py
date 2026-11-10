# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    def get_score(points):
        """Calculate the metric R = (d_min / d_max)^2"""
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
        Objective function to minimize.
        We want to maximize d_min, so we minimize -d_min.
        x is a flattened array of shape (32,).
        """
        points = x.reshape(-1, 2)
        dists = pdist(points)
        # If points coincide, dist is 0, score is bad.
        # We want to maximize min dist.
        return -np.min(dists)

    def constraints(x):
        """
        Constraints: d_max <= 1.
        We express this as 1 - d_max >= 0.
        """
        points = x.reshape(-1, 2)
        dists = pdist(points)
        d_max = np.max(dists)
        return 1.0 - d_max

    # We use SLSQP which supports inequality constraints
    # Constraint type 'ineq' means func(x) >= 0
    
    def run_optimization(initial_points):
        """
        Run optimization from a specific starting point.
        """
        x0 = initial_points.flatten()
        
        # Normalize initial points to have max dist 1 to satisfy constraint initially or close to it
        d_max_init = np.max(pdist(initial_points))
        if d_max_init > 0:
            x0 = x0 / d_max_init
        else:
            # If all points are the same, perturb them
            x0 = np.random.randn(n, 2).flatten()
            d_max_init = np.max(pdist(x0.reshape(-1, 2)))
            if d_max_init > 0:
                x0 = x0 / d_max_init

        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                constraints=[{'type': 'ineq', 'fun': constraints}],
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if res.success:
                best_points = res.x.reshape(-1, 2)
                # The optimizer maximizes d_min subject to d_max <= 1.
                # The resulting score is (d_min / 1)^2 = d_min^2.
                # However, we should re-normalize to ensure d_max is exactly 1 for the score calculation 
                # or just calculate the score directly. The score is invariant to scaling.
                return best_points
            else:
                return None
        except Exception:
            return None

    # Generate multiple initial configurations
    inits = []
    
    # 1. 4x4 Grid
    x = np.arange(4)
    y = np.arange(4)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    # Center it
    grid_points -= np.mean(grid_points, axis=0)
    inits.append(grid_points)

    # 2. Hexagonal Lattice (approx)
    # Create a hex lattice and take the 16 closest to center
    hex_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            # Hexagonal coordinates to Cartesian
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            hex_points.append([x, y])
    hex_points = np.array(hex_points)
    hex_points -= np.mean(hex_points, axis=0)
    # Sort by distance to center and take top 16
    dists_center = np.linalg.norm(hex_points, axis=1)
    indices = np.argsort(dists_center)[:16]
    hex_init = hex_points[indices]
    inits.append(hex_init)

    # 3. Random points in unit circle
    np.random.seed(42)
    for _ in range(5):
        r = np.sqrt(np.random.rand(n))
        theta = np.random.rand(n) * 2 * np.pi
        rand_points = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        inits.append(rand_points)

    # 4. Points on a circle (regular polygon) with some radial jitter
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.ones(n)
    circle_points = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    inits.append(circle_points)
    
    # 5. Two concentric circles
    theta1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    theta2 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    r1 = 0.5
    r2 = 1.0
    c1 = np.column_stack((r1 * np.cos(theta1), r1 * np.sin(theta1)))
    c2 = np.column_stack((r2 * np.cos(theta2), r2 * np.sin(theta2)))
    concentric_points = np.vstack((c1, c2))
    inits.append(concentric_points)

    best_points = None
    best_score = -1.0

    for init in inits:
        if time.time() > deadline:
            break
            
        # Try optimization from this start
        res_points = run_optimization(init)
        
        if res_points is not None:
            score = get_score(res_points)
            if score > best_score:
                best_score = score
                best_points = res_points

    # Fallback: if optimization failed or didn't improve over a simple grid, use grid
    if best_points is None or best_score < get_score(inits[0]):
        best_points = inits[0]
        best_score = get_score(best_points)

    # Final check and return
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        best_points = inits[0]
        
    return best_points

# EVOLVE-BLOCK-END
