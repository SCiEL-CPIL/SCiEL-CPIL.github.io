import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0 (flat-top)
    # Vertices at 0, 60, ..., 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals for the outer hexagon sides
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value (approximate penetration depth).
        If they are separated, returns the positive distance.
        """
        min_gap = float('inf')
        
        # Axes from poly1 edges
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9: continue
            axis = normal1 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap on this axis
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                # Overlapping on this axis
                min_gap = min(min_gap, gap)
            else:
                # Separated on this axis -> disjoint
                return gap
                
        # Axes from poly2 edges
        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9: continue
            axis = normal2 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        return min_gap

    def is_valid(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    def objective_penalty(params, penalty_weight):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # Calculate outer side length
        side_len = calc_min_outer_side(positions, angles, 0.0)
        
        # Calculate overlap penalty
        penalty = 0.0
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    penalty += penalty_weight * gap**2
        
        return side_len + penalty

    best_score = float('inf')
    best_result = None

    # Multi-start Nelder-Mead with penalty method
    # We use several random initializations and let Nelder-Mead optimize.
    # The penalty weight is annealed or kept high to force validity.
    
    np.random.seed(42)
    
    # Number of restarts
    num_restarts = 20
    
    for k in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        # Initialize positions on a loose lattice or randomly
        # A loose lattice is a good starting point
        sqrt3 = math.sqrt(3)
        v1 = np.array([1.5, sqrt3/2])
        v2 = np.array([0, sqrt3])
        
        # Random subset of lattice points
        points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                p = i * v1 + j * v2
                points.append(p)
        
        # Shuffle and pick 12
        np.random.shuffle(points)
        selected = np.array(points[:12])
        
        # Add random noise to break symmetry
        selected += np.random.normal(0, 0.1, size=(N, 2))
        
        # Center the cluster
        centroid = np.mean(selected, axis=0)
        selected -= centroid
        
        # Random angles
        angles = np.random.uniform(0, 360, size=N)
        
        x0 = np.concatenate([selected.flatten(), angles])
        
        # Run Nelder-Mead
        # Start with high penalty, decrease it? 
        # Or just use a high penalty and rely on NM to find valid config.
        # Since NM is derivative free, it handles non-smoothness okay.
        
        try:
            # Use a relatively high penalty to ensure we find a valid config first
            # Then we can refine with lower penalty? 
            # Actually, if we minimize side_len + penalty, and penalty is large,
            # it will prioritize validity. Once valid, penalty is 0, and it minimizes side_len.
            
            res = minimize(
                lambda x: objective_penalty(x, 100.0), 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True, 'disp': False}
            )
            
            pos = res.x[:2*N].reshape(N, 2)
            ang = res.x[2*N:]
            
            # Check validity
            if is_valid(pos, ang, tol=1e-5):
                score = calc_min_outer_side(pos, ang, 0.0)
                if score < best_score:
                    best_score = score
                    best_result = res.x.copy()
        except Exception:
            pass

    # If we found a valid result, do a final polish with lower penalty and tighter tolerance
    if best_result is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                lambda x: objective_penalty(x, 1000.0), 
                best_result, 
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True, 'disp': False}
            )
            
            pos = res_final.x[:2*N].reshape(N, 2)
            ang = res_final.x[2*N:]
            
            if is_valid(pos, ang, tol=1e-6):
                score_final = calc_min_outer_side(pos, ang, 0.0)
                if score_final < best_score:
                    best_score = score_final
                    best_result = res_final.x.copy()
        except Exception:
            pass

    # Fallback if no valid result found (should not happen with loose lattice start)
    if best_result is None:
        # Use the loose lattice directly
        sqrt3 = math.sqrt(3)
        v1 = np.array([1.5, sqrt3/2])
        v2 = np.array([0, sqrt3])
        points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                p = i * v1 + j * v2
                points.append(p)
        points.sort(key=lambda p: np.dot(p, p))
        selected = np.array(points[:12])
        centroid = np.mean(selected, axis=0)
        selected -= centroid
        angles = np.zeros(N)
        best_result = np.concatenate([selected.flatten(), angles])
        best_score = calc_min_outer_side(selected, angles, 0.0)

    final_positions = best_result[:2*N].reshape(N, 2)
    final_angles = best_result[2*N:]
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0

if __name__ == "__main__":
    construct()
