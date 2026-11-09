# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    Uses a D3h symmetric parametrization (2 poles + 2 hexagonal rings)
    combined with a generic 42-D local refinement.
    """
    n = 14
    deadline = time.time() + 580.0  # Safety margin before 630s limit

    def score(pts):
        if pts.shape != (14, 3):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        d = pdist(pts)
        dmin = np.min(d)
        dmax = np.max(d)
        if dmax < 1e-12:
            return 0.0
        return (dmin / dmax) ** 2

    def neg_score(pts):
        return -score(pts)

    # --- Stage 1: D3h Symmetric Search ---
    # Points:
    # 0: (0, 0, 1)
    # 1: (0, 0, -1)
    # 2-7: Ring 1 (upper) at z, radius sqrt(1-z^2), 6 points
    # 8-13: Ring 2 (lower) at -z, radius sqrt(1-z^2), 6 points with phase shift phi
    
    def build_d3h(params):
        z, phi = params
        if z < 0: z = 0
        if z > 1: z = 1
        # Clamp z to [1e-5, 1-1e-5] to avoid singularities if needed, though poles are distinct
        z = np.clip(z, 1e-5, 1 - 1e-5)
        r = np.sqrt(1 - z*z)
        
        pts = np.zeros((14, 3))
        pts[0, 2] = 1.0
        pts[1, 2] = -1.0
        
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        
        # Ring 1 (Upper)
        for k in range(6):
            a = angles[k]
            pts[2+k, 0] = r * np.cos(a)
            pts[2+k, 1] = r * np.sin(a)
            pts[2+k, 2] = z
            
        # Ring 2 (Lower)
        for k in range(6):
            a = angles[k] + phi
            pts[8+k, 0] = r * np.cos(a)
            pts[8+k, 1] = r * np.sin(a)
            pts[8+k, 2] = -z
            
        return pts

    def neg_score_d3h(params):
        pts = build_d3h(params)
        return neg_score(pts)

    best_score = 0.0
    best_pts = None
    
    # Grid search over z and phi
    # z in (0, 1), phi in [0, pi/3) due to symmetry of hexagon? 
    # Actually, relative rotation phi and phi+pi/3 gives same set of distances?
    # The hexagon has 6-fold symmetry. Rotating ring 2 by phi or phi + pi/3 results in same relative distances.
    # So phi in [0, pi/3] is sufficient.
    
    n_z = 50
    n_phi = 50
    
    z_vals = np.linspace(0.05, 0.95, n_z)
    phi_vals = np.linspace(0.0, np.pi/3, n_phi)
    
    for z in z_vals:
        for phi in phi_vals:
            if time.time() > deadline - 10:
                break
            pts = build_d3h([z, phi])
            s = score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
                
    if best_pts is None:
        # Fallback: Fibonacci sphere
        phi_i = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
        theta_i = np.pi * (1 + 5**0.5) * np.arange(n)
        best_pts = np.column_stack((
            np.sin(phi_i) * np.cos(theta_i),
            np.sin(phi_i) * np.sin(theta_i),
            np.cos(phi_i)
        ))
        best_score = score(best_pts)

    # Local refinement on D3h parameters
    if best_pts is not None:
        # We need to extract z and phi from best_pts? 
        # It's easier to just restart from the grid best parameters.
        # Let's store the best parameters too.
        # Wait, I didn't store best_params in the grid loop. Let's re-run or track it.
        # I'll just run a few random restarts of local optimization on the 2D parameter space.
        
        # Actually, let's just take the current best_pts and use it to initialize a 42D optimizer.
        # But first, let's try to optimize the 2D parameters more precisely using the current best as a seed for local search.
        # Since we didn't track best_params, let's just do a few random restarts in the 2D space using minimize.
        
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            
            if restart == 0:
                # Use the center of the grid or a random point
                x0 = [0.5, 0.0]
            else:
                x0 = [np.random.uniform(0.1, 0.9), np.random.uniform(0, np.pi/3)]
                
            res = minimize(neg_score_d3h, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            pts = build_d3h(res.x)
            s = score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()

    # --- Stage 2: Generic 42-D Local Refinement ---
    # Start from the best D3h configuration and allow all coordinates to move.
    # This can break symmetry if a better asymmetric configuration exists (though unlikely for N=14, it's safe).
    
    if time.time() < deadline - 5:
        x0 = best_pts.flatten()
        # Use L-BFGS-B for efficiency, but we need to handle the fact that the objective is non-smooth (min/max).
        # Nelder-Mead is robust for non-smooth objectives.
        # Since 42D is large for NM, we might be slow. 
        # However, we are starting from a very good point, so a few iterations might suffice or it might stagnate.
        # Let's try a limited number of iterations.
        
        # Alternative: Use the derivative-free optimizer with a small step size.
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < -best_score:
                new_pts = res.x.reshape(14, 3)
                if np.all(np.isfinite(new_pts)):
                    new_score = score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts
        except Exception:
            pass

    # Final validation
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        # Fallback
        phi_i = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
        theta_i = np.pi * (1 + 5**0.5) * np.arange(n)
        best_pts = np.column_stack((
            np.sin(phi_i) * np.cos(theta_i),
            np.sin(phi_i) * np.sin(theta_i),
            np.cos(phi_i)
        ))

    return best_pts

# EVOLVE-BLOCK-END
