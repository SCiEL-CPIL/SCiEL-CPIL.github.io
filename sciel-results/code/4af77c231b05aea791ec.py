import numpy as np
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    
    # Fourier series parameters
    # We use modes 1 to 5.
    # theta_i = 2*pi*i/n + sum( a_m * cos(2*pi*m*i/n) + b_m * sin(2*pi*m*i/n) )
    # We optimize the coefficients.
    
    def generate_polygon(coeffs):
        # coeffs: [a1, b1, a2, b2, a3, b3, a4, b4, a5, b5]
        i = np.arange(n)
        base_angles = 2.0 * np.pi * i / n
        
        angles = base_angles.copy()
        for m in range(1, 6):
            a = coeffs[2*(m-1)]
            b = coeffs[2*(m-1)+1]
            angles += a * np.cos(2.0 * np.pi * m * i / n) + b * np.sin(2.0 * np.pi * m * i / n)
            
        x = np.cos(angles)
        y = np.sin(angles)
        pts = np.column_stack((x, y))
        
        # Normalize
        centroid = np.mean(pts, axis=0)
        pts -= centroid
        dists = np.linalg.norm(pts, axis=1)
        mean_dist = np.mean(dists)
        if mean_dist < 0.01:
            return None
        pts /= mean_dist
        
        return pts

    def score_polygon(pts):
        if pts is None:
            return -1e9
            
        # Check convexity
        # Calculate cross products of consecutive edges
        edges = np.roll(pts, -1, axis=0) - pts
        cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
        
        # All cross products must have the same sign and be non-zero
        if np.any(cross == 0):
            return -1e9
        if np.any(cross > 0) and np.any(cross < 0):
            return -1e9
            
        # Calculate vertex scores
        min_score = 1.0
        for i in range(n):
            v = pts[i]
            # Distances to all other vertices
            diffs = pts - v
            dists = np.linalg.norm(diffs, axis=1)
            # Exclude self
            dists = np.delete(dists, i)
            
            # Sort
            dists_sorted = np.sort(dists)
            
            # Find k: first index minimizing d[k+1] - d[k]
            gaps = np.diff(dists_sorted)
            min_gap = np.min(gaps)
            k = np.argmin(gaps)
            
            # Check if any dist < 0.05
            if dists_sorted[0] < 0.05:
                continue # Score 0 for this vertex, min_score becomes 0
            
            # Determine which 4 distances to use
            if k == 0:
                d_subset = dists_sorted[0:4]
            elif k == 17: # 19-2 = 17
                d_subset = dists_sorted[15:19]
            else:
                d_subset = dists_sorted[k-1:k+3]
                
            D = np.mean(d_subset)
            if D < 0.1:
                continue # Score 0
                
            # raw = mean of max(d/D, D/d)
            ratios = np.maximum(d_subset / D, D / d_subset)
            raw = np.mean(ratios)
            v_score = min(1.0, 1.0 / raw)
            
            min_score = min(min_score, v_score)
            
        return min_score

    def objective(coeffs):
        pts = generate_polygon(coeffs)
        s = score_polygon(pts)
        return -s # Minimize negative score

    # Initial guess: Regular polygon (all coeffs 0)
    # We need to perturb to escape the local minimum?
    # The regular polygon is a symmetric point. The score function is symmetric.
    # Small perturbations might not change the score much if the symmetry is preserved.
    # However, the "gap" logic depends on the specific ordering.
    # Let's try a small random perturbation to break symmetry and allow the optimizer to find a better asymmetric (or lower symmetry) shape.
    
    np.random.seed(42)
    x0 = np.random.normal(0, 0.1, size=10)
    
    # Run optimization
    # Nelder-Mead is good for this
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_coeffs = res.x
    best_pts = generate_polygon(best_coeffs)
    
    # It's possible the optimizer didn't improve or went invalid.
    # We should compare with the regular polygon score.
    reg_pts = generate_polygon(np.zeros(10))
    reg_score = score_polygon(reg_pts)
    opt_score = score_polygon(best_pts)
    
    if opt_score > reg_score:
        return best_pts
    else:
        return reg_pts
