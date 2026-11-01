import numpy as np
import scipy.spatial as spat
import time

N = 14
# Symmetry group size: 14 points arranged in 7 pairs of antipodal points
# This implies rotational symmetry of order 7 (or effectively, we fix angles and optimize radii)
# The optimal configuration for maximizing min triangle area in a convex hull for even N
# often has high symmetry. For N=14, a 7-fold symmetric configuration (7 radii, 7 fixed angles)
# is a strong candidate.

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    """
    try:
        hull_area = _hull_area(points)
        if hull_area < 1e-10:
            return 0.0
        
        n = len(points)
        # Precompute all triangle areas
        # C(14,3) = 364 triangles. 
        # Vectorized computation is fast enough for 1D optimization steps.
        
        # Generate indices for all unique triples (i, j, k) with i < j < k
        i_idx, j_idx, k_idx = np.indices((n, n, n)).reshape(3, -1)
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        k_idx = k_idx[mask]
        
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]
        
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        return min_area / hull_area
    except Exception:
        return 0.0

def _points_from_radii(radii):
    """
    Construct 14 points from 7 radii.
    The points are placed at angles k * (2*pi/14) = k * pi/7 for k=0..13.
    Due to the symmetry of the problem (convex hull, min triangle area),
    it is highly likely that the optimal solution has 14-fold or 7-fold symmetry.
    Let's assume 14-fold symmetry is too restrictive (all radii equal -> regular polygon).
    Let's assume 7-fold symmetry: r_k = r_{k+7}.
    So we have 7 unique radii r_0, ..., r_6.
    Angles are fixed at theta_k = k * pi / 7.
    """
    angles = np.arange(N) * (np.pi / 7.0)
    # Map index k to radius index k % 7
    r_idx = np.arange(N) % 7
    radii_full = radii[r_idx]
    
    x = radii_full * np.cos(angles)
    y = radii_full * np.sin(angles)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Initial guess: Regular 14-gon (all radii = 1)
    # But we can start with slight perturbations or just uniform.
    # Actually, the regular polygon is a local minimum for min triangle area?
    # No, for N=4, square is better than rhombus? 
    # For maximizing min triangle area, we want to "spread" points.
    # The regular polygon is a strong starting point.
    
    # However, we suspect the optimal might not be regular.
    # Let's start with regular and optimize radii.
    
    best_radii = np.ones(7)
    best_points = _points_from_radii(best_radii)
    best_score = _score(best_points)
    
    current_radii = best_radii.copy()
    current_score = best_score
    
    # Coordinate Descent with Simulated Annealing / Random Restarts
    # We optimize 7 variables (radii).
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Coordinate Descent
    # Iterate through each radius, find best value via 1D search.
    
    max_iter = 5000
    step_size = 0.1
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        improved = False
        for i in range(7):
            # 1D search for radius[i]
            # We can use a golden section search or just gradient-free local search
            # Given the non-smoothness of min() and hull area, simple random search around current value
            # or a coarse grid search is robust.
            
            # Let's try a local random search for this coordinate
            base_val = current_radii[i]
            best_local_val = base_val
            best_local_score = current_score
            
            # Try a few perturbations
            for _ in range(20):
                # Gaussian perturbation
                delta = rng.normal(0, step_size)
                new_val = base_val + delta
                if new_val < 0.1: # Keep radius positive
                    new_val = 0.1 + rng.random() * 0.1
                    delta = new_val - base_val
                elif new_val > 2.0: # Cap radius to avoid huge hulls
                    new_val = 2.0 - rng.random() * 0.1
                    delta = new_val - base_val
                
                trial_radii = current_radii.copy()
                trial_radii[i] = new_val
                trial_points = _points_from_radii(trial_radii)
                trial_score = _score(trial_points)
                
                if trial_score > best_local_score:
                    best_local_score = trial_score
                    best_local_val = new_val
            
            if best_local_val != base_val:
                current_radii[i] = best_local_val
                current_score = best_local_score
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_radii = current_radii.copy()
                    best_points = _points_from_radii(best_radii)
        
        if not improved:
            # If no improvement in a full pass, try a larger step or restart
            # Reduce step size if stuck
            step_size *= 0.9
            if step_size < 1e-6:
                break

    # Phase 2: Full Space Refinement
    # Take the best symmetric solution and refine in full 2D space using local search
    # This breaks the symmetry constraint if it was suboptimal, but usually
    # symmetry is preserved or slightly broken for marginal gain.
    
    current_points = best_points.copy()
    current_score = best_score
    
    # Simple local search in 2D
    T = 0.05
    for _ in range(2000):
        if time.time() > deadline:
            break
            
        noise = rng.normal(0, T, size=(N, 2))
        new_points = current_points + noise
        new_score = _score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Accept worse with small probability to escape local minima
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(-abs(delta) / T)
                if rng.random() < prob:
                    current_points = new_points
                    current_score = new_score
                    
        T *= 0.999
        
    return best_points

if __name__ == "__main__":
    # Quick self-test
    import time
    start = time.time()
    pts = construct()
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    print(f"Score: {_score(pts)}")
    print(pts)
