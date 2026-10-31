"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def _min_area(pts):
    """Compute the minimum area of any triangle formed by three of the points."""
    n = len(pts)
    min_a = float('inf')
    # Pre-allocate or use list for speed if necessary, but for N=11 (165 triangles) simple loop is fine
    for i in range(n):
        ax, ay = pts[i]
        for j in range(i + 1, n):
            bx, by = pts[j]
            for k in range(j + 1, n):
                cx, cy = pts[k]
                # Area = 0.5 * | (bx-ax)(cy-ay) - (by-ay)(cx-ax) |
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_a:
                    min_a = area
                    if min_a < 1e-15:
                        return 0.0
    return min_a

def _clamp_to_triangle(p):
    """
    Clamp a point to be inside the equilateral triangle using barycentric coordinates.
    This is more robust than simple projection for the interior.
    """
    x, y = p
    # Barycentric coordinates (a, b, c) such that p = a*V0 + b*V1 + c*V2
    # V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    # x = b + 0.5*c
    # y = (sqrt(3)/2)*c  => c = 2*y/sqrt(3)
    # b = x - 0.5*c
    # a = 1 - b - c
    
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c
    
    # Clamp to [0, 1]
    # If any coordinate is negative, it's outside.
    # We project back to the feasible region a,b,c >= 0, a+b+c=1.
    
    # Simple clamping logic:
    # If a < 0, set a=0, renormalize b,c
    # If b < 0, set b=0, renormalize a,c
    # If c < 0, set c=0, renormalize a,b
    # If multiple are negative, it's more complex, but for small deviations, 
    # clipping the most negative one and renormalizing usually works.
    # A robust way is to find the closest point in the simplex.
    
    # For efficiency and robustness in optimization, we can use a simple iterative clip:
    for _ in range(3):
        if a < 0:
            a = 0
            s = b + c
            if s > 0:
                b /= s
                c /= s
            else:
                a = 1; b = 0; c = 0
        if b < 0:
            b = 0
            s = a + c
            if s > 0:
                a /= s
                c /= s
            else:
                a = 0; b = 1; c = 0
        if c < 0:
            c = 0
            s = a + b
            if s > 0:
                a /= s
                b /= s
            else:
                a = 0; b = 0; c = 1
        
        # Check if sum is 1 (it should be if we started with a+b+c=1 and only set one to 0 and normalized others)
        # But if we just set to 0 without normalizing, sum < 1.
        # Let's restart the loop if we changed anything to ensure stability
        # Actually, the standard "project to simplex" algorithm is better.
        
    # Proper projection to simplex a,b,c >= 0, sum=1
    # Sort a,b,c
    vals = np.array([a, b, c])
    # Find the projection
    # This is a known problem: project vector onto probability simplex.
    # Algorithm:
    # 1. Sort v in decreasing order
    # 2. Find rho such that sum_{i<=rho} (v_i - tau) = 1, where tau = (sum_{i<=rho} v_i - 1) / rho
    # 3. v_i = max(0, v_i - tau)
    
    # For 3 dimensions, we can do this analytically or with a small loop.
    # Given the points are usually close to the triangle during optimization,
    # the simple clamping above is often sufficient if we normalize at the end.
    # Let's use a simpler, faster approach that works for small violations:
    
    a = max(0, a)
    b = max(0, b)
    c = max(0, c)
    s = a + b + c
    if s < 1e-10:
        # Fallback to center
        return V0 + (V1 + V2 - 2*V0)/3
    a /= s
    b /= s
    c /= s
    
    return a * V0 + b * V1 + c * V2

def _initial_placement_lattice():
    """
    Create a structured initial placement using a triangular lattice.
    For N=11, a good pattern is a 3x3 grid plus center adjustments, or specific rows.
    Rows: 4, 4, 3 points.
    """
    pts = []
    # Row 1 (bottom): 4 points
    y1 = 0.1
    for i in range(4):
        x = 0.1 + i * 0.25
        pts.append([x, y1])
        
    # Row 2 (middle): 4 points
    y2 = 0.35
    for i in range(4):
        x = 0.2 + i * 0.2
        pts.append([x, y2])
        
    # Row 3 (top): 3 points
    y3 = 0.6
    for i in range(3):
        x = 0.3 + i * 0.2
        pts.append([x, y3])
        
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    rng = np.random.default_rng(12345)
    
    best_points = None
    best_score = -1
    
    # Strategy 1: Structured start + L-BFGS-B
    init_pts = _initial_placement_lattice()
    
    # Define objective for L-BFGS-B
    # We want to maximize min_area, so minimize -min_area
    # We also need to ensure points stay in the triangle. 
    # L-BFGS-B doesn't natively support "clamp inside triangle" as a constraint easily 
    # without defining gradients of the clamp function.
    # Instead, we use a penalty method or rely on the clamp inside the objective.
    # However, differentiability of clamp is an issue.
    # Alternative: Optimize in barycentric coordinates?
    # Let's stick to Cartesian with a penalty for being outside, 
    # but since we clamp inside the objective, the optimizer might struggle with the kinks.
    
    # Better approach: Use Nelder-Mead which is derivative-free and robust to non-smoothness,
    # or use L-BFGS-B with a smooth penalty for boundary violation.
    
    # Let's try a hybrid:
    # 1. Nelder-Mead on the initial lattice (robust to non-smooth min area)
    # 2. L-BFGS-B on the result (fast polish if landscape is smooth enough locally)
    # 3. Simulated Annealing for global refinement
    
    def objective_nm(x):
        pts = x.reshape(N, 2)
        # Clamp to ensure validity for area calculation
        pts_c = np.array([_clamp_to_triangle(p) for p in pts])
        min_a = _min_area(pts_c)
        # Add a small penalty if any point is significantly outside (should be 0 due to clamp, but just in case)
        return -min_a
    
    # Run Nelder-Mead from the lattice start
    try:
        res = minimize(objective_nm, init_pts.flatten(), method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        final_pts = res.x.reshape(N, 2)
        final_pts = np.array([_clamp_to_triangle(p) for p in final_pts])
        score = _min_area(final_pts)
        if score > best_score:
            best_score = score
            best_points = final_pts.copy()
    except Exception:
        pass
    
    # Try a few random restarts with Nelder-Mead if time permits
    n_restarts = 10
    for i in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Random start
        pts = []
        for _ in range(N):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u * 1.0 + v * 0.5
            y = v * np.sqrt(3) / 2
            pts.append([x, y])
        pts = np.array(pts)
        
        try:
            res = minimize(objective_nm, pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            final_pts = res.x.reshape(N, 2)
            final_pts = np.array([_clamp_to_triangle(p) for p in final_pts])
            score = _min_area(final_pts)
            if score > best_score:
                best_score = score
                best_points = final_pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        return _initial_placement_lattice()
    
    current_best = best_points.copy()
    current_score = best_score
    
    # Strategy 2: Simulated Annealing / Local Search Refinement
    # This helps escape local minima of the min-area function.
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        T = 0.01  # Initial temperature
        T_min = 1e-6
        cooling_rate = 0.995
        
        # Estimate number of iterations we can afford
        # _min_area is O(N^3) = 11^3 ~ 1331 operations. Very fast.
        # We can run many iterations.
        
        iter_count = 0
        while time.time() < deadline - 2 and T > T_min:
            iter_count += 1
            idx = rng.integers(0, N)
            # Perturbation size proportional to T
            delta = rng.normal(0, T, 2)
            new_pts = current_best.copy()
            new_pts[idx] = _clamp_to_triangle(current_best[idx] + delta)
            
            new_score = _min_area(new_pts)
            
            # Acceptance probability
            if new_score > current_score:
                current_best = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_best.copy()
            else:
                # Boltzmann factor
                # dE = new_score - current_score (negative)
                # Prob = exp(dE / T)
                # Note: We are maximizing, so dE is negative if worse.
                try:
                    prob = np.exp((new_score - current_score) / max(T, 1e-10))
                except OverflowError:
                    prob = 0.0
                
                if rng.random() < prob:
                    current_best = new_pts
                    current_score = new_score
            
            T *= cooling_rate
            
            # Adaptive cooling: if stuck, cool faster? Or keep going?
            # Standard SA is fine.
            
    return best_points

# EVOLVE-BLOCK-END
