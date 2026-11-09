# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Treat the problem as a smooth optimization of the pairwise distance spread.
    2. Objective: Minimize the variance of pairwise distances (or CV^2).
       Minimizing the variance encourages all distances to be equal, which maximizes the min/max ratio.
       Since the metric is scale-invariant, we can fix the mean distance to 1 (or just let the optimizer handle scale, 
       but fixing it helps stability). Actually, minimizing variance alone might collapse points to 0.
       We must prevent collapse. 
       
       A better objective for "equal distances" is minimizing the sum of squared deviations from the mean distance.
       However, if all points coincide, variance is 0. We need to ensure points are distinct.
       The "coefficient of variation" (std/mean) is scale invariant. Minimizing CV^2 = (std/mean)^2 
       drives the distribution to be as tight as possible relative to its magnitude.
       If points collapse, mean->0, CV->0? No, if points collapse to 0, mean=0, undefined.
       
       Let's look at the metric: (d_min / d_max)^2.
       This is maximized when d_min is as large as possible relative to d_max.
       Equivalently, we want to minimize d_max / d_min.
       
       A common smooth surrogate for minimizing the spread is minimizing the variance of the distances.
       To avoid collapse, we can constrain the mean distance to be 1, or simply normalize the points 
       such that the RMS distance is 1 during the objective evaluation, and minimize the variance.
       
       Let's define the objective:
       1. Compute pairwise distances.
       2. Normalize points so that the mean distance is 1? Or just compute the ratio.
       
       Actually, a very effective objective for Tammes/spherical code problems is to minimize the 
       potential energy (sum of 1/d^k), but that maximizes min distance, not the ratio.
       
       The specific goal is to equalize distances.
       Let's minimize: sum( (d_i - d_mean)^2 ) / d_mean^2  (which is n * CV^2)
       This is scale invariant.
       
       To prevent the trivial solution where all points are at the same location (d_mean=0), 
       we must ensure the points are spread out. 
       However, if we start from a good initialization (like icosahedron), the optimizer will 
       likely stay in the basin of the non-trivial solution.
       
       We will use multiple initializations:
       1. Icosahedron + 2 Poles (normalized)
       2. Two Hexagonal Rings (normalized)
       3. Random points on a sphere
       
       For each, we run Nelder-Mead or L-BFGS-B (if we have gradients, but we don't easily).
       Nelder-Mead is robust for 42D.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580

    def objective(params):
        """
        Minimize the squared coefficient of variation of pairwise distances.
        CV^2 = Var(dist) / Mean(dist)^2
        This is scale invariant.
        """
        pts = params.reshape((n, d))
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        
        mean_d = np.mean(dists)
        if mean_d < 1e-12:
            # Collapse penalty
            return 1e10
        
        var_d = np.var(dists)
        return var_d / (mean_d * mean_d)

    def generate_init_ico():
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, -1, phi],
            [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0],
            [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1],
            [phi, 0, -1], [-phi, 0, -1]
        ])
        ico = ico / np.linalg.norm(ico, axis=1)[:, np.newaxis]
        # Add 2 poles
        pole1 = np.array([0, 0, 1.2]) # Slightly outside to encourage separation? 
        pole2 = np.array([0, 0, -1.2])
        pts = np.vstack([ico, pole1, pole2])
        # Center and normalize
        pts -= np.mean(pts, axis=0)
        return pts.flatten()

    def generate_init_hex():
        pts = []
        # Ring 1
        for i in range(6):
            a = 2 * np.pi * i / 6
            pts.append([np.cos(a), np.sin(a), 0.5])
        # Ring 2
        for i in range(6):
            a = 2 * np.pi * i / 6 + np.pi/6
            pts.append([np.cos(a), np.sin(a), -0.5])
        # Poles
        pts.append([0, 0, 1.5])
        pts.append([0, 0, -1.5])
        pts = np.array(pts)
        pts -= np.mean(pts, axis=0)
        return pts.flatten()

    def generate_init_random():
        # Random points on sphere
        rng = np.random.default_rng(42)
        pts = rng.normal(size=(n, d))
        pts = pts / np.linalg.norm(pts, axis=1)[:, np.newaxis]
        return pts.flatten()

    def generate_init_cube_like():
        # 8 corners of a cube + 6 face centers + 2 poles? 
        # Cube vertices
        verts = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
        # Face centers
        centers = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])
        # Need 14 points. 8 + 6 = 14. Perfect.
        pts = np.vstack([verts, centers])
        pts -= np.mean(pts, axis=0)
        return pts.flatten()

    inits = [
        generate_init_ico(),
        generate_init_hex(),
        generate_init_random(),
        generate_init_cube_like()
    ]
    
    # Add some perturbed random inits
    rng = np.random.default_rng(123)
    for _ in range(4):
        pts = rng.normal(size=(n, d))
        pts = pts / np.linalg.norm(pts, axis=1)[:, np.newaxis]
        inits.append(pts.flatten())

    best_params = None
    best_obj = np.inf

    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        try:
            res = minimize(
                objective, 
                init, 
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
                
        except Exception as e:
            pass

    if best_params is None:
        # Fallback to ico
        best_params = inits[0]

    points = best_params.reshape((n, d))
    
    # Final cleanup: center and normalize to unit max distance? 
    # The metric is scale invariant, so scaling doesn't change the score.
    # But we must return finite points.
    points -= np.mean(points, axis=0)
    
    # Check for degeneracy
    dists = pdist(points)
    if np.max(dists) < 1e-9:
        # If collapsed, use fallback
        points = np.array(inits[0]).reshape((n, d))
        points -= np.mean(points, axis=0)

    return points

# EVOLVE-BLOCK-END
