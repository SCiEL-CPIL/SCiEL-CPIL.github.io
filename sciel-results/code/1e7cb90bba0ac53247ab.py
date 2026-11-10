import numpy as np
import time
import scipy.optimize
import scipy.spatial.distance

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600 - 30  # 570 seconds to be safe

    def get_distances(points):
        """Calculate pairwise distances for a (N, 2) array."""
        return scipy.spatial.distance.pdist(points)

    def raw_objective(x):
        """
        Objective function to minimize.
        We want to maximize R = (d_min / d_max)^2.
        So we minimize -d_min/d_max.
        """
        points = x.reshape(n, d)
        dists = get_distances(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return 0.0
        return -(d_min / d_max)

    def smoothed_objective(x, beta=20.0):
        """
        Smoothed surrogate objective.
        Uses softmin and softmax to approximate min and max in a differentiable way.
        This allows gradient-based methods to work.
        """
        points = x.reshape(n, d)
        dists = get_distances(points)
        
        # Softmin: -1/beta * log(sum(exp(-beta * dists)))
        # Softmax: 1/beta * log(sum(exp(beta * dists)))
        
        # For numerical stability:
        max_d = np.max(dists)
        
        # Softmin calculation
        # min(d) ~ max_d - (1/beta) * log(sum(exp(-beta * (d - max_d))))
        term_min = np.exp(-beta * (dists - max_d))
        sum_min = np.sum(term_min)
        d_min_soft = max_d - (1.0 / beta) * np.log(sum_min)
        
        # Softmax calculation
        # max(d) ~ max_d + (1/beta) * log(sum(exp(beta * (d - max_d))))
        term_max = np.exp(beta * (dists - max_d))
        sum_max = np.sum(term_max)
        d_max_soft = max_d + (1.0 / beta) * np.log(sum_max)
        
        if d_max_soft == 0:
            return 0.0
            
        ratio = d_min_soft / d_max_soft
        return -ratio

    def get_score(points):
        """Calculate the actual score R = (d_min/d_max)^2"""
        dists = get_distances(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Generate diverse initial candidates
    # 1. Hexagonal-like lattice (good for packing)
    # 2. Circle (regular polygon)
    # 3. Random (seeded for reproducibility)
    # 4. Two clusters
    
    candidates = []
    
    # Candidate 1: 4x4 Grid (centered)
    grid_x, grid_y = np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4))
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    candidates.append(grid_pts)
    
    # Candidate 2: Regular 16-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates.append(circle_pts)
    
    # Candidate 3: Random Gaussian
    np.random.seed(42)
    rand_pts = np.random.randn(n, d)
    candidates.append(rand_pts)
    
    # Candidate 4: Hexagonal packing approximation
    # 3 rings: 1 center, 6 inner, 9 outer? No, 1+6+12=19. 
    # Let's try a distorted grid or specific hexagonal subset.
    # A 4x4 grid is actually quite good for 16 points.
    # Let's try a "superlattice" or just perturb the grid slightly.
    hex_pts = grid_pts.copy()
    # Shift alternating rows
    for i in range(1, 4, 2):
        hex_pts[i*4:(i+1)*4, 0] += 0.5
    candidates.append(hex_pts)
    
    best_points = None
    best_score = -np.inf
    
    # Optimization Strategy:
    # Use L-BFGS-B on the smoothed objective.
    # Start with a moderate beta (e.g., 10) to navigate the landscape,
    # then refine with higher beta (e.g., 50, 100) to sharpen the min/max.
    
    for i, init_pts in enumerate(candidates):
        if time.time() > deadline:
            break
            
        current_x = init_pts.flatten().copy()
        
        # Check initial score
        init_score = get_score(init_pts)
        if init_score > best_score:
            best_score = init_score
            best_points = init_pts.copy()
            
        # Multi-stage refinement with L-BFGS-B
        # Stage 1: Coarse search (lower beta)
        # Stage 2: Fine search (higher beta)
        
        betas = [5.0, 10.0, 20.0, 50.0]
        
        for beta in betas:
            if time.time() > deadline:
                break
                
            try:
                res = scipy.optimize.minimize(
                    smoothed_objective, 
                    current_x, 
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
                )
                
                if res.success or res.fun < smoothed_objective(current_x, beta=beta):
                    current_x = res.x
                    # Update best if this stage improved the true score
                    pts = current_x.reshape(n, d)
                    score = get_score(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
                        
            except Exception:
                pass
                
    # Final check and return
    if best_points is None:
        best_points = candidates[0]
        
    return best_points
