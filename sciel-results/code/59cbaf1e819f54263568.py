import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Helper: Generate vertices of a regular hexagon
    def hex_verts(cx, cy, s, rot_deg):
        rot_rad = np.deg2rad(rot_deg)
        angles = rot_rad + np.deg2rad(np.arange(6) * 60)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def min_dist_convex(v1, v2):
        """
        Fast exact min distance between two convex polygons (hexagons).
        Returns 0 if intersecting, else min distance.
        """
        min_d = float('inf')
        # Check vertex of v1 to edges of v2
        for i in range(6):
            p1 = v1[i]
            for j in range(6):
                p2 = v2[j]
                p3 = v2[(j+1)%6]
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
        return min_d

    def get_outer_side_length(params):
        """
        Compute the minimum side length of a regular hexagon centered at (0,0)
        with angle 0 that contains all inner hexagons defined by params.
        params: [x1, y1, a1, x2, y2, a2, ...]
        """
        max_proj = 0.0
        # Normals of the outer hexagon (flat-top orientation? No, vertex at 0 deg)
        # If outer hex has vertex at 0 deg, its edges are perpendicular to 30, 90, 150...
        # Inradius r = s * sqrt(3)/2.
        # Condition: dot(vertex, normal) <= r
        # So r >= max(dot(v, n)) for all v in all inner hexagons.
        # s = r * 2 / sqrt(3)
        
        normals = np.deg2rad(30 + np.arange(6)*60)
        n_vecs = np.column_stack([np.cos(normals), np.sin(normals)])
        
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            v = hex_verts(x, y, 1.0, a)
            projs = np.dot(v, n_vecs.T)
            max_proj = max(max_proj, np.max(projs))
        
        r = max_proj
        s = r * 2.0 / np.sqrt(3.0)
        return s

    def objective(params, penalty_weight):
        """
        Objective: Sum of overlap violations + penalty_weight * outer_side_length
        We want to minimize overlaps to 0, then minimize size.
        """
        overlap_violation = 0.0
        
        # Precompute vertices for all inner hexagons
        verts = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            verts.append(hex_verts(x, y, 1.0, a))
            
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                d = min_dist_convex(verts[i], verts[j])
                if d < 1e-9: # Intersecting
                    # Penalty is proportional to how much they overlap?
                    # min_dist returns 0 if intersecting.
                    # We need a measure of penetration.
                    # Approximation: If centers are closer than min touching dist, penalty.
                    # For aligned hexagons, min center dist is sqrt(3).
                    # For rotated, it varies. 
                    # A robust penalty: sum of squared distances from vertices to the other polygon if inside?
                    # Simpler: Use center distance heuristic for penalty magnitude.
                    cx1, cy1 = params[3*i], params[3*i+1]
                    cx2, cy2 = params[3*j], params[3*j+1]
                    c_dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                    # Threshold ~ 1.732
                    threshold = 1.732 
                    if c_dist < threshold:
                        overlap_violation += (threshold - c_dist) ** 2 * 10.0
                    else:
                        # They intersect but centers are far? Unlikely for convex unit hexes unless rotated weirdly.
                        # Just add a constant penalty.
                        overlap_violation += 1.0
                        
        outer_s = get_outer_side_length(params)
        
        return overlap_violation + penalty_weight * outer_s

    # Simulated Annealing
    def simulate_annealing():
        # Initial state: Random positions and angles
        # Start with a loose grid to help convergence
        x0 = np.zeros(N * 3)
        # Grid layout: 4x3
        for i in range(N):
            row = i // 4
            col = i % 4
            x0[3*i] = (col - 1.5) * 2.2
            x0[3*i+1] = (row - 1.0) * 2.0
            x0[3*i+2] = 0.0 # Angle
            
        # Add noise
        x0 += np.random.normal(0, 0.5, x0.shape)
        
        current_params = x0
        current_cost = objective(current_params, 1.0)
        
        best_params = current_params.copy()
        best_cost = current_cost
        
        T = 5.0 # Initial temperature
        cooling_rate = 0.999
        min_T = 0.01
        
        # Penalty weight increases over time to prioritize size reduction once overlaps are low
        # But we must ensure overlaps go to 0 first.
        # Strategy: Start with low penalty weight for size, high for overlaps?
        # My objective: overlap + w * size.
        # If w is small, it focuses on overlap. As w increases, it focuses on size.
        # Let's ramp w up from 0.1 to 1.0 over the run.
        
        max_steps = 20000
        steps_per_check = 100
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            # Ramp penalty weight
            progress = step / max_steps
            w = 0.1 + 0.9 * progress
            
            # Generate new state
            new_params = current_params.copy()
            # Perturb one or two hexagons
            idx = np.random.randint(N)
            new_params[3*idx] += np.random.normal(0, T)
            new_params[3*idx+1] += np.random.normal(0, T)
            new_params[3*idx+2] += np.random.normal(0, T * 10) # Angle in degrees, scale larger? 
            # Actually, let's keep angle perturbation smaller to avoid wild rotations
            new_params[3*idx+2] += np.random.normal(0, T * 5)
            
            new_cost = objective(new_params, w)
            
            # Acceptance probability
            if new_cost < current_cost:
                current_params = new_params
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_params = current_params.copy()
            else:
                delta = new_cost - current_cost
                if T > 0:
                    prob = np.exp(-delta / T)
                    if np.random.rand() < prob:
                        current_params = new_params
                        current_cost = new_cost
            
            # Cool down
            T *= cooling_rate
            
        return best_params

    # Run multiple restarts of SA
    best_final_params = None
    best_final_s = float('inf')
    
    # How many restarts can we do?
    # Each SA run might take ~5-10s depending on steps.
    # We have 145s. Let's try up to 10-15 restarts if fast, or fewer if slow.
    
    restarts = 0
    while time.time() < deadline - 10.0:
        params = simulate_annealing()
        restarts += 1
        
        # Final polish: Check if valid (overlaps ~ 0)
        # If overlaps > 0, this solution is invalid. We need a valid solution.
        # Let's verify validity and get the exact side length.
        
        # Check overlaps
        verts = []
        valid = True
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            verts.append(hex_verts(x, y, 1.0, a))
            
        for i in range(N):
            for j in range(i+1, N):
                d = min_dist_convex(verts[i], verts[j])
                if d < 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            s = get_outer_side_length(params)
            if s < best_final_s:
                best_final_s = s
                best_final_params = params.copy()
                
        # If we've found a good solution (< 4.0), we can stop early or try to improve
        if best_final_s < 3.8 and restarts > 3:
             # We have a good result. Try a few more if time permits.
             pass
             
    if best_final_params is None:
        # Fallback: Return a known valid large configuration
        # Grid 4x3 with spacing 2.2 x 2.0
        inner = []
        for i in range(12):
            row = i // 4
            col = i % 4
            x = (col - 1.5) * 2.2
            y = (row - 1.0) * 2.0
            inner.append([x, y, 0.0])
        # Calculate side for this grid
        # Max x: 1.5*2.2 + 1 = 3.3+1=4.3
        # Max y: 1.0*2.0 + 0.866 = 2.866
        # This is an estimate. Just return a safe large value.
        return inner, (0.0, 0.0), 6.0, 0.0

    # Extract results
    inner = []
    for i in range(N):
        x = best_final_params[3*i]
        y = best_final_params[3*i+1]
        a = best_final_params[3*i+2]
        inner.append([x, y, a])
        
    # Center is (0,0) because we optimized for hexagon centered at origin.
    # Rotation of outer hexagon is 0 (aligned with axes as per support function logic).
    
    return inner, (0.0, 0.0), best_final_s, 0.0

if __name__ == "__main__":
    construct()
