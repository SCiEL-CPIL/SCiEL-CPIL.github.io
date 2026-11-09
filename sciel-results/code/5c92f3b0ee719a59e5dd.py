import numpy as np
import time
import math
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = 615.0  # Safety margin before 630s kill
    
    # Grid for evaluation, matching the evaluator
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    P = np.column_stack((X.ravel(), Y.ravel()))
    n_points = P.shape[0]
    
    # Precompute base corner checks? No, the base is fixed in local coords.
    
    def evaluate_path_fast(path):
        """
        Calculate the area of the intersection of the corridor placements.
        path: array of shape (n, 3)
        Returns: (score, valid_mask)
        """
        if not isinstance(path, np.ndarray):
            path = np.array(path)
            
        # Prepend and append fixed endpoints
        # The evaluator puts (0, 0, 0) before and (-2, 0, 90) after.
        full_path = np.vstack([np.array([0.0, 0.0, 0.0]), path, np.array([-2.0, 0.0, 90.0])])
        
        # Densify path exactly as evaluator does
        dense_path = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            if m == 1:
                dense_path.append(p2)
            else:
                for k in range(1, m+1):
                    t = k / m
                    interp = p1 + t * (p2 - p1)
                    dense_path.append(interp)
                    
        # Convert to numpy array for vectorization if needed, but loop is fine for ~90 steps
        dense_path = np.array(dense_path)
        
        valid = np.ones(n_points, dtype=bool)
        
        # Check each placement
        for px, py, pa in dense_path:
            a_rad = np.deg2rad(pa)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = P[:, 0] - px
            dy = P[:, 1] - py
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            # Base corridor:
            # Arm 1: 0 <= u <= 1, -3 <= v <= 1
            # Arm 2: -3 <= u <= 1, 0 <= v <= 1
            in_base = ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
                      ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))
            
            valid = valid & in_base
            
            # Early exit if no points left
            if not np.any(valid):
                return 0.0, valid
                
        score = np.sum(valid) * 0.0001
        return score, valid

    # Strategy:
    # Represent the path as a smooth function of angle.
    # Angles are integer degrees from 1 to 89.
    # We model x(a) and y(a) as polynomials in t = a/90.
    # Let's use cubic polynomials (4 coefficients each) for flexibility.
    # x(t) = c0 + c1*t + c2*t^2 + c3*t^3
    # y(t) = c4 + c5*t + c6*t^2 + c7*t^3
    
    # Constraints/Initial guess:
    # At t=0 (a=0), x=0, y=0.
    # At t=1 (a=90), x=-2, y=0.
    # This implies:
    # x(0) = c0 = 0
    # x(1) = c0 + c1 + c2 + c3 = -2  => c1 + c2 + c3 = -2
    # y(0) = c4 = 0
    # y(1) = c4 + c5 + c6 + c7 = 0   => c5 + c6 + c7 = 0
    
    # We can reduce the number of free parameters by enforcing these boundary conditions.
    # Let c0=0, c4=0.
    # c1 = -2 - c2 - c3
    # c5 = -c6 - c7
    
    # Free parameters: c2, c3, c6, c7 (4 parameters).
    # Or we can just optimize all 8 and let the optimizer handle it, 
    # but fixing boundaries ensures we start with a valid path and reduces search space.
    
    # Let's optimize the 4 free parameters: [c2, c3, c6, c7]
    # Initial guess:
    # A common "smooth" turn is roughly circular or parabolic.
    # If we assume a circular arc of radius R, center at (x_c, y_c).
    # For 90 deg turn from (0,0) to (-2,0), center could be at (-1, 1) or (-1, -1)?
    # Vector from start to end is (-2, 0). Midpoint (-1, 0).
    # Perpendicular bisector is x = -1.
    # If center is (-1, 1), radius is sqrt(1^2 + 1^2) = sqrt(2).
    # Check: dist from (-1,1) to (0,0) is sqrt(2). dist to (-2,0) is sqrt(2).
    # Angle at (0,0) relative to center: vector (1, -1). Angle -45 deg.
    # Angle at (-2,0) relative to center: vector (-1, -1). Angle -135 deg.
    # Total rotation 90 deg.
    # This fits perfectly.
    # So the path is a circular arc.
    # Parametric eq for circle center (-1, 1), r=sqrt(2), angle theta from -45 to -135?
    # Wait, the sofa orientation changes by 90 deg.
    # The center of rotation of the sofa corner is not necessarily the center of the path circle.
    # However, for a constant width corridor, the path of the inner corner for a smooth turn
    # is often close to a circular arc or a polynomial.
    
    # Let's just pick reasonable initial values for the polynomial coefficients.
    # Linear interpolation: x = -2t, y=0. (c2=c3=0, c6=c7=0). This is the "straight line" path.
    # This path is valid?
    # If y=0, the sofa moves horizontally.
    # At a=0, sofa is vertical-ish (arm up, arm left).
    # At a=90, sofa is horizontal-ish (arm left, arm down).
    # Moving horizontally might cause the sofa to hit the "outer" wall of the corridor?
    # The corridor is fixed in the world? No, the corridor moves with the sofa?
    # No, "Moving sofa from a corridor path". The corridor is the space the sofa occupies.
    # The "base corridor" is the shape of the space available at any given time?
    # "Row (x, y, a) places the width-1 L-shaped corridor with its inner corner at (x, y)".
    # So the CORRIDOR moves. The SOFA is fixed in the world?
    # "The sofa is the set of grid points ... that are inside every placement".
    # Yes, the sofa is fixed in the world (grid points). The corridor (the "hole" or "space") moves.
    # So we are moving a hole around a fixed object?
    # Or moving the object through a fixed hole?
    # "places the ... corridor ... at (x,y)".
    # "w is inside placement ... if R(-a)(w - (x,y)) lies in the base".
    # This defines the region of space occupied by the corridor at that step.
    # The sofa points must be inside ALL these regions.
    # So the intersection of all corridor placements must cover the sofa.
    # We want to maximize the area of the intersection of the corridor placements.
    # This is equivalent to finding the largest fixed shape (sofa) that fits in the moving corridor.
    # Since the sofa is defined by the grid points in the intersection, maximizing the count is correct.
    
    # Initial guess for polynomial:
    # Let's try a simple curve that bulges out.
    # x(t) = -2t
    # y(t) = h * sin(pi * t) ? Or just a parabola y = 4h t(1-t).
    # If we use polynomial:
    # y(t) = c5 t + c6 t^2 + c7 t^3.
    # Symmetric parabola: c5 = c7, c6 = -2 c5.
    # y(t) = c5(t - 2t^2 + t^3) = c5 t(1-t)^2.
    # Let's try c5 = 1.0, c7 = 1.0, c6 = -2.0.
    # x(t) = -2t. c1=-2, c2=0, c3=0.
    
    def generate_path_from_coeffs(coeffs):
        c2, c3, c6, c7 = coeffs
        # Fixed: c0=0, c4=0
        # c1 = -2 - c2 - c3
        # c5 = -c6 - c7
        
        c1 = -2.0 - c2 - c3
        c5 = -c6 - c7
        
        angles = np.arange(1, 90)
        t = angles / 90.0
        
        x = c0_val + c1*t + c2*t**2 + c3*t**3
        y = c4_val + c5*t + c6*t**2 + c7*t**3
        
        return np.column_stack([x, y, angles])
    
    c0_val = 0.0
    c4_val = 0.0
    
    def objective(coeffs):
        path = generate_path_from_coeffs(coeffs)
        score, _ = evaluate_path_fast(path)
        return -score
    
    # Initial guess
    # Try a few different initial guesses and pick the best one to start Nelder-Mead.
    # 1. Straight line (0,0,0,0)
    # 2. Parabolic bulge (0,0, -2, 1) -> y = t - 2t^2 + t^3
    # 3. Larger bulge
    
    initial_guesses = [
        [0.0, 0.0, 0.0, 0.0],       # Straight
        [0.0, 0.0, -2.0, 1.0],      # Parabolic y = t(1-t)^2
        [0.0, 0.0, -4.0, 2.0],      # Larger parabolic
        [0.0, 0.0, 2.0, -1.0],      # Opposite bulge
        [1.0, -1.0, 0.0, 0.0],      # x-curve
        [-1.0, 1.0, 0.0, 0.0],      # x-curve opposite
    ]
    
    best_score = 0.0
    best_coeffs = None
    
    for guess in initial_guesses:
        if time.time() > deadline - 5:
            break
            
        res = minimize(objective, guess, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if res.fun < -best_score: # Minimizing negative score
            best_score = -res.fun
            best_coeffs = res.x
            
    # If we didn't find anything, use the first guess
    if best_coeffs is None:
        best_coeffs = initial_guesses[0]
    
    # Final evaluation to get the exact mask and score for the best found
    final_path = generate_path_from_coeffs(best_coeffs)
    final_score, _ = evaluate_path_fast(final_path)
    
    # Optional: Try a local refinement with coordinate descent on the polynomial coeffs?
    # Nelder-Mead should be good enough for 4 variables.
    
    # Ensure output format
    return final_path

if __name__ == "__main__":
    moving_sofa_2d()
