import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    start_time = time.time()
    time_limit = 580.0  # Safety margin
    
    def score(points):
        """Calculate the score R = (d_min / d_max)^2"""
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(params):
        """Objective function for minimization (we minimize -score)"""
        points = params.reshape(n, 2)
        s = score(points)
        return -s

    def generate_hex_cluster():
        """Generate a hexagonal cluster of 16 points"""
        # Rows: 1, 3, 5, 7 points
        points = []
        # Center at (0,0) roughly
        # Row 0 (1 point)
        points.append([0, 0])
        # Row 1 (3 points)
        y1 = np.sqrt(3) / 2
        points.extend([[-1, y1], [0, y1], [1, y1]])
        # Row 2 (5 points)
        y2 = np.sqrt(3)
        points.extend([[-1.5, y2], [-0.5, y2], [0.5, y2], [1.5, y2], [2.5, y2]])
        # Wait, standard hex packing:
        # Let's use a proper hex grid generator
        pts = []
        # Center
        pts.append((0,0))
        # Ring 1 (6 points)
        for i in range(6):
            angle = i * np.pi / 3
            pts.append((np.cos(angle), np.sin(angle)))
        # Ring 2 (12 points, we need 9 more to make 16 total? 1+6+9=16)
        # Actually, just take a patch of the lattice.
        
        # Let's just define the 1+3+5+7 structure explicitly with correct coordinates
        # Spacing 1
        s3 = np.sqrt(3)
        # Row 0: y=0, x=0
        p0 = [(0,0)]
        # Row 1: y=s3/2, x=-0.5, 0.5, 1.5 (shifted by 0.5)
        p1 = [(-0.5, s3/2), (0.5, s3/2), (1.5, s3/2)]
        # Row 2: y=s3, x=-1, 0, 1, 2, 3
        p2 = [(-1, s3), (0, s3), (1, s3), (2, s3), (3, s3)]
        # Row 3: y=3*s3/2, x=-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5
        p3 = [(-1.5, 3*s3/2), (-0.5, 3*s3/2), (0.5, 3*s3/2), (1.5, 3*s3/2), (2.5, 3*s3/2), (3.5, 3*s3/2), (4.5, 3*s3/2)]
        
        all_pts = p0 + p1 + p2 + p3
        arr = np.array(all_pts)
        # Center the points
        center = arr.mean(axis=0)
        arr -= center
        return arr

    def generate_square_grid():
        """Generate a 4x4 square grid"""
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def generate_rings(n_inner, n_outer):
        """Generate ring configuration"""
        p_center = np.array([0.0, 0.0])
        r1 = 0.5
        angles1 = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
        p_inner = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        r2 = 1.0
        angles2 = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
        angles2 += np.pi / n_outer
        p_outer = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        return np.vstack([p_center, p_inner, p_outer])

    best_points = None
    best_score = -1.0
    
    # List of initial configurations
    inits = [
        generate_hex_cluster(),
        generate_square_grid(),
        generate_rings(6, 9),
        generate_rings(5, 10),
        generate_rings(7, 8),
    ]
    
    # Add some random inits
    for _ in range(5):
        rand_pts = np.random.rand(n, 2) * 2 - 1
        inits.append(rand_pts)

    # Optimization loop
    for i, init_pts in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
            
        # Try to optimize from this init
        # Use Powell as it is robust for non-smooth functions
        try:
            res = minimize(objective, init_pts.flatten(), method='Powell', 
                           options={'maxiter': 500, 'maxfev': 5000})
            opt_pts = res.x.reshape(n, 2)
            s = score(opt_pts)
            
            if s > best_score:
                best_score = s
                best_points = opt_pts
        except Exception:
            continue

        # If we have time, try basin hopping: perturb best and re-optimize
        if best_points is not None and time.time() - start_time < time_limit - 60:
            # Perturb the best points found so far
            for _ in range(5): # 5 perturbations
                if time.time() - start_time > time_limit:
                    break
                noise = np.random.normal(0, 0.1, best_points.shape)
                perturbed = best_points + noise
                try:
                    res2 = minimize(objective, perturbed.flatten(), method='Powell', 
                                   options={'maxiter': 200, 'maxfev': 2000})
                    opt_pts2 = res2.x.reshape(n, 2)
                    s2 = score(opt_pts2)
                    if s2 > best_score:
                        best_score = s2
                        best_points = opt_pts2
                except Exception:
                    continue

    # If no valid points were found (should not happen), return a default
    if best_points is None:
        best_points = generate_hex_cluster()

    return best_points
