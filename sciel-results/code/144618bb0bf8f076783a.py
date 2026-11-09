"""Strategy: Constrained Non-Linear Optimization (SLSQP) for Hexagon Packing."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute base hexagon geometry
    # Vertices of a unit regular hexagon (side 1) centered at origin, pointy-top (vertex at 90 deg? or 0 deg?)
    # Let's use standard math angles: 0, 60, 120, 180, 240, 300.
    # This is "pointy right".
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    base_vertices = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    # Edge normals for SAT.
    # Edge i is between vertex i and vertex i+1.
    # Normal is perpendicular to edge, pointing outward.
    # For a regular polygon, outward normal of edge i is at angle (angle_i + angle_{i+1})/2 + 90?
    # Let's compute normals explicitly.
    base_normals = np.zeros((6, 2))
    for i in range(6):
        v1 = base_vertices[i]
        v2 = base_vertices[(i+1)%6]
        edge_vec = v2 - v1
        # Rotate edge_vec by -90 degrees to get outward normal?
        # For CCW vertices, outward normal is (dy, -dx) normalized?
        # Let's check: Edge from (1,0) to (0.5, 0.866). Vec (-0.5, 0.866).
        # Outward should point away from origin.
        # Midpoint (0.75, 0.433).
        # Normal (0.866, 0.5) is at 30 deg. Midpoint angle is 30 deg. Correct.
        # So normal = (dy, -dx) / |edge| ?
        # dy = 0.866, -dx = 0.5. Yes.
        nx = edge_vec[1]
        ny = -edge_vec[0]
        norm = np.sqrt(nx*nx + ny*ny)
        base_normals[i] = [nx/norm, ny/norm]
        
    # Note: base_vertices are defined for a hexagon centered at 0.
    # When rotated by theta, vertices become R(theta) @ base_vertices.
    # Normals also rotate: R(theta) @ base_normals.

    def get_rotated_matrix(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    def hex_vertices(center, theta):
        R = get_rotated_matrix(theta)
        return center + (R @ base_vertices.T).T

    def hex_normals(theta):
        R = get_rotated_matrix(theta)
        return R @ base_normals.T

    def sat_overlap(v1, n1, v2, n2):
        """
        Returns the overlap amount between two convex polygons using SAT.
        v1, n1: vertices and outward normals of poly 1
        v2, n2: vertices and outward normals of poly 2
        Returns > 0 if overlapping, 0 if touching, < 0 if separated.
        We want to enforce separation, so constraint will be -overlap >= 0 (i.e. overlap <= 0).
        Actually, standard SAT: if there is a separating axis, they are disjoint.
        We check all axes (normals of both polygons).
        For each axis, project both polygons.
        If ranges do not overlap, return distance (separation).
        If all axes overlap, they are intersecting. The minimum overlap depth is the penetration.
        
        For optimization, we need a smooth-ish penalty or hard constraint.
        SLSQP handles inequalities.
        Constraint: min_separation >= 0.
        If they overlap, min_separation is negative (penetration depth).
        So constraint: -min_separation <= 0 ? No, we want them NOT to overlap.
        So we want min_separation >= 0.
        In scipy minimize, constraints are:
        {'type': 'ineq', 'fun': lambda x: g(x)} where g(x) >= 0.
        So fun should return the minimum separation distance.
        """
        min_sep = float('inf')
        
        # Check axes from poly 1
        for i in range(6):
            axis = n1[i]
            proj1 = np.dot(v1, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            
            proj2 = np.dot(v2, axis)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Separation distance on this axis
            if max1 < min2:
                sep = min2 - max1
            elif max2 < min1:
                sep = min1 - max2
            else:
                # Overlap on this axis
                sep = -min(max1 - min2, max2 - min1) # Penetration depth
            
            if sep < min_sep:
                min_sep = sep
                
            # Early exit if we found a separating axis (sep > 0)
            # Wait, if sep > 0, they are separated on this axis, so they are disjoint.
            # We want the MINIMUM separation over all axes? No.
            # If ANY axis has sep > 0, the polygons are disjoint.
            # If ALL axes have sep <= 0, they are overlapping. The penetration is the max overlap (min negative sep).
            
            if sep > 1e-9:
                return sep # Disjoint
        
        # If we get here, all axes overlapped.
        # min_sep is the most negative value (deepest penetration).
        # Return min_sep (which is negative).
        return min_sep

    def is_inside_outer_hexagon(point, S, outer_theta):
        """
        Check if point is inside outer hexagon of side S, rotated by outer_theta.
        Outer hexagon vertices are at distance S from center.
        Normals are at distance S * sqrt(3)/2 from center.
        """
        # Outer hexagon is centered at (0,0) for simplicity in constraint calculation, 
        # but the problem allows outer center to move? 
        # "outer hexagon center (x, y)". 
        # However, by symmetry, we can fix outer center at (0,0) and let inner hexagons move.
        # Or fix inner relative to outer.
        # Let's fix outer center at (0,0) and outer rotation at 0 to reduce variables?
        # The problem says "outer hexagon center (x, y), side length and rotation".
        # But score is just side length.
        # We can fix outer center to (0,0) and outer rotation to 0 without loss of generality 
        # for finding the minimal side length, because we can always translate/rotate the whole system.
        # So variables are only for inner hexagons: (x_i, y_i, theta_i) for i=1..N.
        
        R_out_inradius = S * np.sqrt(3) / 2.0
        
        # Outer normals (for theta=0) are at 30, 90, 150...
        # base_normals are for pointy-right hex (vertices 0, 60...).
        # Wait, my base_vertices were 0, 60... which is pointy-right.
        # Normals are 30, 90...
        # This matches.
        
        # Point is inside if dot(point, normal) <= R_out_inradius for all 6 normals.
        dists = np.dot(point, base_normals.T)
        return np.all(dists <= R_out_inradius + 1e-9)

    def constraints_func(x, S):
        """
        x: flattened array of [x1, y1, theta1, x2, y2, theta2, ...]
        Returns array of constraint values. All must be >= 0.
        """
        cons = []
        
        # 1. Disjointness constraints
        for i in range(N):
            for j in range(i+1, N):
                ci = x[3*i:3*i+2]
                ti = x[3*i+2]
                
                cj = x[3*j:3*j+2]
                tj = x[3*j+2]
                
                vi = hex_vertices(ci, ti)
                ni = hex_normals(ti)
                
                vj = hex_vertices(cj, tj)
                nj = hex_normals(tj)
                
                sep = sat_overlap(vi, ni, vj, nj)
                # We want sep >= 0.
                # Add small margin to avoid numerical issues?
                cons.append(sep + 1e-6)
                
        # 2. Containment constraints
        # For each inner hexagon, all vertices must be inside outer hexagon.
        # Actually, it's sufficient to check the vertices of the inner hexagon.
        # But S is also a variable? 
        # In SLSQP, we minimize S.
        # So x should include S.
        # Let's define x = [S, x1, y1, theta1, ...]
        
        # Re-evaluating variable structure.
        # x[0] = S
        # x[1:] = positions and angles
        
        S_val = x[0]
        
        for i in range(N):
            ci = x[1+3*i:1+3*i+2]
            ti = x[1+3*i+2]
            
            vi = hex_vertices(ci, ti)
            
            # Check all vertices
            for v in vi:
                # Constraint: v inside outer hexagon of side S_val.
                # dot(v, normal) <= R_out(S_val)
                # R_out(S_val) = S_val * sqrt(3)/2
                # Constraint: S_val * sqrt(3)/2 - dot(v, normal) >= 0
                dists = np.dot(v, base_normals.T)
                margin = S_val * np.sqrt(3)/2.0 - dists
                # We need all margins >= 0.
                # SLSQP takes a vector of constraints.
                cons.extend(margin)
                
        return np.array(cons)

    def objective(x):
        return x[0] # Minimize S

    # Initial guess
    # S_init = 5.0
    # Generate a hexagonal lattice of 11 points.
    # Center at (0,0).
    # Ring 0: 1 point
    # Ring 1: 6 points
    # Ring 2: 4 points (to make 11)
    
    # Let's just use a grid-like initialization that is valid.
    # Spacing 2*sqrt(3) is safe.
    # But we want compact.
    # Let's try to initialize with a valid configuration for S=5.
    
    # Simple initialization:
    # Place them in a circle? No, they will overlap.
    # Place them in a hex lattice with spacing 2.0 (safe for side 1 hexagons? 
    # Min dist between centers of touching hexagons is sqrt(3) ~ 1.732.
    # So spacing 1.8 is safe.
    
    init_positions = []
    # 1 center
    init_positions.append((0, 0))
    # 6 around
    for k in range(6):
        angle = k * np.pi / 3
        init_positions.append((1.8 * np.cos(angle), 1.8 * np.sin(angle)))
    # 4 more
    # Place at radius 3.6
    for k in range(4):
        angle = k * np.pi / 2 + np.pi/4
        init_positions.append((3.6 * np.cos(angle), 3.6 * np.sin(angle)))
        
    # Convert to x vector
    x0 = np.zeros(1 + 3*N)
    x0[0] = 5.0
    for i in range(N):
        x0[1+3*i] = init_positions[i][0]
        x0[1+3*i+1] = init_positions[i][1]
        x0[1+3*i+2] = 0.0 # No rotation initially
        
    # Run SLSQP
    # Bounds:
    # S > 0
    # Positions: -10 to 10
    # Angles: 0 to 2*pi (or -pi to pi)
    
    bounds = [(0.1, 10.0)]
    for i in range(N):
        bounds.append((-10.0, 10.0))
        bounds.append((-10.0, 10.0))
        bounds.append((0.0, 2*np.pi))
        
    # Constraints
    # We need to define the constraint function properly for SLSQP.
    # SLSQP expects constraints as a list of dicts.
    # {'type': 'ineq', 'fun': fun} where fun returns array >= 0.
    
    # However, the number of constraints is large:
    # Disjointness: N*(N-1)/2 = 55
    # Containment: N * 6 = 66
    # Total ~ 121 constraints.
    # Variables: 1 + 33 = 34.
    # This is a well-posed NLP.
    
    # One issue: sat_overlap is not differentiable at contact points.
    # SLSQP uses gradients. We can provide analytical gradients or let it approximate.
    # Approximation might be slow or inaccurate.
    # Let's try without gradients first.
    
    # To improve robustness, we can run multiple times with different random perturbations.
    
    best_S = 100.0
    best_x = None
    
    for restart in range(5):
        if time.time() > deadline - 5.0:
            break
            
        # Perturb initial guess
        x_try = x0.copy()
        if restart > 0:
            # Small random noise
            noise = np.random.normal(0, 0.1, size=x_try.shape)
            noise[0] = 0 # Don't perturb S initial
            # Don't perturb angles too much?
            x_try += noise
            # Ensure angles are in bounds
            for i in range(N):
                idx = 1 + 3*i + 2
                x_try[idx] = x_try[idx] % (2*np.pi)
                
        try:
            res = minimize(objective, x_try, method='SLSQP', bounds=bounds,
                           constraints=[{'type': 'ineq', 'fun': constraints_func}],
                           options={'maxiter': 500, 'ftol': 1e-9})
            
            if res.success:
                S_val = res.x[0]
                # Verify validity manually to be safe
                # Check constraints
                c_vals = constraints_func(res.x)
                if np.all(c_vals >= -1e-6):
                    if S_val < best_S:
                        best_S = S_val
                        best_x = res.x
            else:
                # Even if not "success", if it improved and is valid, take it
                S_val = res.x[0]
                c_vals = constraints_func(res.x)
                if np.all(c_vals >= -1e-6) and S_val < best_S:
                    best_S = S_val
                    best_x = res.x
                    
        except Exception as e:
            pass
            
    if best_x is None:
        # Fallback to a known valid configuration
        # Use the initial lattice with S=5.0
        best_S = 5.0
        best_x = x0
        
    # Extract results
    S_final = best_x[0]
    inner = []
    for i in range(N):
        x_i = best_x[1+3*i]
        y_i = best_x[1+3*i+1]
        angle_deg = np.rad2deg(best_x[1+3*i+2])
        inner.append([float(x_i), float(y_i), float(angle_deg)])
        
    return inner, (0.0, 0.0), float(S_final), 0.0

if __name__ == "__main__":
    construct()
