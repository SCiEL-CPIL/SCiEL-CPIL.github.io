import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    """
    Maximize minimum triangle area / convex hull area for 14 points.
    Uses log-sum-exp smoothing with explicit normalization (centering and scaling)
    to handle scale invariance and remove translational degrees of freedom.
    """
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)
    
    # Precompute triangle indices for vectorized area computation
    tri_i = np.array([t[0] for t in triples])
    tri_j = np.array([t[1] for t in triples])
    tri_k = np.array([t[2] for t in triples])
    
    def get_points_normalized(x):
        """
        x has 2*N elements.
        Returns points centered at origin and scaled such that sum of squared distances to origin is 1.
        """
        pts = x.reshape((N, 2))
        # Center
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        # Scale
        norm_sq = np.sum(pts**2)
        if norm_sq < 1e-15:
            # Fallback to regular polygon if collapsed
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            pts = np.column_stack([np.cos(angles), np.sin(angles)])
            norm_sq = np.sum(pts**2)
            pts = pts / np.sqrt(norm_sq)
        else:
            pts = pts / np.sqrt(norm_sq)
        return pts
    
    def compute_areas_vectorized(pts):
        """Compute all triangle areas using vectorized operations"""
        pi = pts[tri_i]  # (n_triples, 2)
        pj = pts[tri_j]  # (n_triples, 2)
        pk = pts[tri_k]  # (n_triples, 2)
        
        v1 = pj - pi
        v2 = pk - pi
        dets = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(dets)
        return areas
    
    def compute_hull_area(pts):
        """Compute convex hull area"""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1e-10
    
    def objective(x, beta):
        """
        Smoothed objective: -min_area / hull_area using log-sum-exp.
        min(a_i) ≈ -1/beta * log(sum(exp(-beta * a_i)))
        """
        pts = get_points_normalized(x)
        areas = compute_areas_vectorized(pts)
        
        # Numerically stable log-sum-exp
        a_min = np.min(areas)
        if a_min < 1e-15:
            return 1.0
        
        # Use the smaller areas for numerical stability
        a_ref = a_min
        exp_terms = np.exp(-beta * (areas - a_ref))
        log_sum = -beta * a_ref + np.log(np.sum(exp_terms))
        smooth_min = -log_sum / beta
        
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        
        return -(smooth_min / hull_area)

    def generate_initials():
        initials = []
        
        # 1. Regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        initials.append(reg_pts.flatten())
        
        # 2. Random points
        for _ in range(10):
            initials.append(np.random.randn(N, 2).flatten())
            
        # 3. Grid-like
        # Create a 4x4 grid and pick 14
        x_coords = np.linspace(-1, 1, 4)
        y_coords = np.linspace(-1, 1, 4)
        grid = []
        for x in x_coords:
            for y in y_coords:
                grid.append([x, y])
        # Shuffle and take first 14
        np.random.shuffle(grid)
        grid_pts = np.array(grid[:14])
        initials.append(grid_pts.flatten())
        
        # 4. Elliptical
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        for rx, ry in [(1.0, 0.5), (0.5, 1.0), (1.0, 1.0), (1.5, 0.5)]:
            ell_pts = np.column_stack([rx * np.cos(angles), ry * np.sin(angles)])
            initials.append(ell_pts.flatten())
            
        return initials

    # Time management
    start_time = time.time()
    time_limit = 115.0
    
    best_score = -np.inf
    best_x = None
    
    dim = 2 * N
    bounds = [(-5.0, 5.0)] * dim
    
    # Phase 1: Coarse to fine beta annealing with multiple starts
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    
    initials = generate_initials()
    
    for beta in betas:
        if time.time() - start_time > time_limit - 10.0:
            break
            
        # Update initials: keep best, add perturbations, keep some diversity
        new_initials = []
        if best_x is not None:
            new_initials.append(best_x.copy())
            # Add perturbations around best
            for sigma in [0.1, 0.5, 1.0]:
                for _ in range(3):
                    jitter = np.random.normal(0, sigma, size=dim)
                    new_initials.append(best_x + jitter)
        
        # Add original initials for diversity
        new_initials.extend(initials[:5])
        
        # Shuffle to ensure diversity
        np.random.shuffle(new_initials)
        
        for x0 in new_initials:
            if time.time() - start_time > time_limit - 10.0:
                break
                
            try:
                res = minimize(
                    objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
                )
                
                # Evaluate raw score to be sure
                pts = get_points_normalized(res.x)
                areas = compute_areas_vectorized(pts)
                min_area = np.min(areas)
                hull_area = compute_hull_area(pts)
                if hull_area > 1e-10:
                    current_score = min_area / hull_area
                    if current_score > best_score:
                        best_score = current_score
                        best_x = res.x.copy()
                        
            except Exception:
                pass
        
        # Update initials for next iteration
        if best_x is not None:
             initials = [best_x.copy()] + initials[:5]
        else:
             initials = new_initials

    # Final polish with Nelder-Mead if time permits
    if time.time() - start_time < time_limit - 5.0 and best_x is not None:
        try:
            # Use a high beta for the final objective to approximate raw min
            res_nm = minimize(
                objective,
                best_x,
                args=(5000.0,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            pts = get_points_normalized(res_nm.x)
            areas = compute_areas_vectorized(pts)
            min_area = np.min(areas)
            hull_area = compute_hull_area(pts)
            if hull_area > 1e-10:
                final_score = min_area / hull_area
                if final_score > best_score:
                    best_score = final_score
                    best_x = res_nm.x.copy()
        except Exception:
            pass

    # Fallback
    if best_x is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_x = pts.flatten()
        
    return get_points_normalized(best_x)
