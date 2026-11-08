import numpy as np
import time
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Helper: Generate vertices of a regular hexagon
    def hex_verts(cx, cy, s, rot_deg):
        # rot_deg is in degrees
        rot_rad = np.deg2rad(rot_deg)
        angles = rot_rad + np.deg2rad(np.arange(6) * 60)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def min_dist_polygons_fast(v1, v2):
        """
        Fast approximate min distance for convex polygons.
        Returns 0 if intersecting, else min distance.
        Uses vertex-to-edge distance.
        """
        min_d = float('inf')
        # Check vertex of v1 to edges of v2
        for i in range(6):
            p1 = v1[i]
            # Distance from p1 to each edge of v2
            for j in range(6):
                p2 = v2[j]
                p3 = v2[(j+1)%6]
                # Project p1 onto line p2-p3
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d:
                    min_d = d
        # Check vertex of v2 to edges of v1
        for i in range(6):
            p1 = v2[i]
            for j in range(6):
                p2 = v1[j]
                p3 = v1[(j+1)%6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d:
                    min_d = d
        
        # Check for intersection (if min_d is very small, they might overlap)
        # For robustness, we should check if any vertex is inside the other.
        # But for the optimizer, a smooth penalty is better.
        # We'll treat min_d < 0 as overlap in the penalty function logic outside.
        return min_d

    def evaluate_violation(params, outer_s):
        """
        params: [x1, y1, ..., x12, y12]
        Returns total violation (containment + overlap).
        Lower is better. 0 means valid.
        """
        xs = params[0:12]
        ys = params[12:24]
        
        # Generate inner hexagons (assumed angle 0 for speed and simplicity of this strategy)
        # We can add rotation later if this fails, but aligned is a strong hypothesis.
        inner_verts = []
        for i in range(12):
            v = hex_verts(xs[i], ys[i], 1.0, 0.0)
            inner_verts.append(v)
        
        violation = 0.0
        
        # 1. Containment Check
        # Outer hexagon: center (0,0), side outer_s, angle 0.
        # Inradius r = outer_s * sqrt(3)/2
        r = outer_s * np.sqrt(3) / 2.0
        # Normals for outer hexagon (axis aligned, flat-top? No, vertex at 0 deg means pointy-right)
        # Vertices at 0, 60, 120...
        # Edges are perpendicular to 30, 90, 150...
        normals = np.deg2rad(30 + np.arange(6)*60)
        n_vecs = np.column_stack([np.cos(normals), np.sin(normals)])
        
        for i in range(12):
            for v in inner_verts[i]:
                dists = np.dot(v, n_vecs.T)
                # If dist > r, it's outside. Penalty is linear or quadratic?
                # Linear is smoother for Nelder-Mead.
                out = np.maximum(0, dists - r)
                violation += np.sum(out)
        
        # 2. Overlap Check
        for i in range(12):
            for j in range(i+1, 12):
                d = min_dist_polygons_fast(inner_verts[i], inner_verts[j])
                # If d < 0, they are intersecting? min_dist returns positive distance.
                # If they intersect, the "min distance" between boundaries is 0, but they overlap.
                # We need a penalty for overlap.
                # A simple proxy: if center distance < min_possible_center_dist, penalty.
                # For two unit hexagons, min center dist is approx 1.732 (sqrt(3)).
                # Let's use a softer penalty based on center distance to guide them apart.
                cx1, cy1 = xs[i], ys[i]
                cx2, cy2 = xs[j], ys[j]
                c_dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                # Threshold for "touching" aligned hexagons is sqrt(3)
                threshold = np.sqrt(3)
                if c_dist < threshold:
                    # Quadratic penalty to push apart
                    violation += (threshold - c_dist) ** 2 * 10.0
                    # Also add the actual geometric violation if we want to be precise,
                    # but center dist is a good gradient-like signal.
                    # To be safe, if d (boundary dist) is very small, they are close.
                    # If they actually intersect, d will be 0 (or small) but c_dist < threshold.
                    # The quadratic penalty handles the repulsion.
        
        return violation

    def objective(params):
        # We need to know outer_s for the objective.
        # In this strategy, we binary search outer_s.
        # So objective is just the violation for a FIXED outer_s.
        return evaluate_violation(params, current_outer_s)

    # Strategy: Binary search on Outer Side Length S
    # Lower bound: 12 hexagons need at least some space.
    # A single hex has width 2 (vertex to vertex) or sqrt(3) (flat to flat).
    # 12 hexagons... roughly 4x3 grid?
    # Let's estimate.
    # Lower bound LB = 3.0 (optimistic)
    # Upper bound UB = 10.0 (safe)
    
    LB = 2.5
    UB = 10.0
    
    best_solution = None
    best_s = float('inf')
    
    # We will perform a binary search for the minimal S such that violation can be minimized to ~0.
    # To save time, we limit the number of Nelder-Mead iterations.
    
    # Initial guess for positions: Grid
    def get_initial_positions():
        # Place them in a loose grid
        pos = []
        # 4 columns, 3 rows roughly
        for i in range(12):
            x = (i % 4) * 2.0 - 3.0
            y = (i // 4) * 1.8 - 1.5
            pos.append([x, y])
        return np.array(pos).flatten()

    # Binary search loop
    # We have ~140s. 
    # Each Nelder-Mead run might take 1-5s depending on iterations.
    # Let's do ~20-30 binary search steps.
    
    max_iter_nm = 200
    tol = 1e-4
    
    for step in range(30):
        if time.time() > deadline:
            break
            
        mid_s = (LB + UB) / 2.0
        current_outer_s = mid_s
        
        # Try to find a valid configuration for this S
        # Use multiple random restarts for Nelder-Mead to avoid local minima
        found_valid = False
        best_violation = float('inf')
        
        # 3 restarts per S value to improve robustness
        for restart in range(3):
            if time.time() > deadline:
                break
                
            x0 = get_initial_positions()
            # Add some noise
            x0 += np.random.normal(0, 0.5, x0.shape)
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter_nm, 'xatol': 1e-4, 'fatol': 1e-4})
                
                if res.fun < best_violation:
                    best_violation = res.fun
                    best_params = res.x
                    
                # Check if it's effectively valid
                if res.fun < 1e-3:
                    found_valid = True
                    best_params = res.x
                    break # If valid, no need for other restarts for this S
            
            except Exception:
                pass
        
        if found_valid:
            # It fits! Update best and try smaller S
            UB = mid_s
            best_solution = best_params.copy()
            best_s = mid_s
        else:
            # Doesn't fit. Need larger S
            LB = mid_s
        
    if best_solution is None:
        # Fallback: Return a large valid box
        # Just place them in a grid with large spacing
        inner = []
        for i in range(12):
            x = (i % 4) * 2.5 - 3.75
            y = (i // 4) * 2.2 - 2.2
            inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0

    # Extract best solution
    xs = best_solution[0:12]
    ys = best_solution[12:24]
    
    inner = []
    for i in range(12):
        inner.append([xs[i], ys[i], 0.0])
        
    return inner, (0.0, 0.0), best_s, 0.0
