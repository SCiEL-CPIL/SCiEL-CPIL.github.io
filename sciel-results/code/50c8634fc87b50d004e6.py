"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def sat_separation(verts1, verts2, margin=0.0):
        """
        Return minimum separation between two convex polygons.
        Positive means separated by that amount, negative means overlapping by that amount.
        """
        min_sep = np.inf
        all_edges = []
        for i in range(len(verts1)):
            all_edges.append(verts1[i] - verts1[(i+1) % len(verts1)])
        for i in range(len(verts2)):
            all_edges.append(verts2[i] - verts2[(i+1) % len(verts2)])
        
        for edge in all_edges:
            norm = np.linalg.norm(edge)
            if norm < 1e-12:
                continue
            normal = np.array([-edge[1], edge[0]])
            normal = normal / norm
            
            proj1 = np.dot(verts1, normal)
            proj2 = np.dot(verts2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Separation on this axis
            sep = max(min2 - max1, min1 - max2)
            if sep < min_sep:
                min_sep = sep
            # If already negative, polygons overlap on this axis
            if min_sep < -margin:
                break
        
        return min_sep

    def compute_min_separation(inner, outer_center, outer_side, outer_angle):
        """
        Compute the minimum separation between all inner hexagons and the outer hexagon,
        and between all pairs of inner hexagons.
        Returns the minimum separation (positive = all disjoint and inside with margin).
        """
        outer_verts = hexagon_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
        inner_verts = []
        for i in range(N):
            v = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            inner_verts.append(v)
        
        min_sep = np.inf
        
        # Check containment: inner inside outer
        for i in range(N):
            # For containment, we need the inner hexagon to be inside the outer.
            # SAT for containment: project both, check inner range is within outer range
            # on all axes. The "separation" for containment is different.
            # Actually, let's use a penalty: if inner is not inside outer, penalty.
            # A hexagon is inside another if for every edge of the outer hexagon,
            # all vertices of the inner hexagon are on the same side (inside).
            outer_edges = []
            for i in range(6):
                outer_edges.append(outer_verts[i] - outer_verts[(i+1) % 6])
            
            for edge in outer_edges:
                norm = np.linalg.norm(edge)
                if norm < 1e-12:
                    continue
                normal = np.array([-edge[1], edge[0]])
                normal = normal / norm
                # The normal should point inward for the outer hexagon
                # Check: center of outer hexagon projected vs edge midpoint
                mid = (outer_verts[0] + outer_verts[1]) / 2
                center_proj = np.dot(outer_center, normal)
                mid_proj = np.dot(mid, normal)
                if center_proj < mid_proj:
                    normal = -normal
                
                # All inner vertices should be on the inside side
                inner_proj = np.dot(inner_verts[i], normal)
                outer_min_proj = np.min(np.dot(outer_verts, normal))
                
                # The inner hexagon is inside if all its vertices project
                # to values >= outer_min_proj (for this particular normal direction)
                # Actually, the constraint is: min(inner_proj) >= outer_min_proj
                # Wait, that's not right either. Let me think again.
                
                # For a convex polygon, point p is inside if for every edge,
                # p is on the same side as the interior.
                # The interior side is determined by the normal pointing inward.
                # So we need: dot(p - edge_point, inward_normal) >= 0 for all p in inner hexagon.
                # The minimum of dot(p, inward_normal) over p in inner hexagon
                # should be >= dot(edge_point, inward_normal).
                
                min_inner = np.min(inner_proj)
                # The edge line: dot(x, normal) = dot(edge_point, normal)
                # We need min_inner >= dot(edge_point, normal)
                # Actually, we need the entire inner hexagon to satisfy this.
                # The most restrictive point is the one with minimum projection.
                
                edge_point = outer_verts[0]  # Any point on the edge
                edge_proj = np.dot(edge_point, normal)
                
                # Separation: how far inside is the inner hexagon?
                sep = min_inner - edge_proj
                if sep < min_sep:
                    min_sep = sep
        
        # Check disjointness between inner hexagons
        for i in range(N):
            for j in range(i+1, N):
                sep = sat_separation(inner_verts[i], inner_verts[j])
                if sep < min_sep:
                    min_sep = sep
        
        return min_sep

    def objective(params):
        """
        Objective: minimize side length, penalize violations.
        params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12, cx, cy, side, angle]
        """
        inner = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            angle = params[3*i+2]
            inner.append([x, y, angle])
        
        center = [params[3*N], params[3*N+1]]
        side = params[3*N+2]
        angle = params[3*N+3]
        
        min_sep = compute_min_separation(inner, center, side, angle)
        
        # Penalty for violations (negative separation)
        if min_sep < 0:
            penalty = -min_sep * 100
        else:
            penalty = 0
        
        return side + penalty

    def initial_guess(pattern='hex'):
        params = np.zeros(3*N + 4)
        
        if pattern == 'hex':
            # Hexagonal packing: center + ring of 6 + partial ring of 5
            positions = []
            positions.append((0, 0, 0))
            for i in range(6):
                theta = i * np.pi / 3
                positions.append((2*np.cos(theta), 2*np.sin(theta), 0))
            for i in range(5):
                theta = i * 2*np.pi / 5 + np.pi/5
                positions.append((4*np.cos(theta), 4*np.sin(theta), 0))
        elif pattern == 'circle':
            positions = [(0, 0, 0)]
            for i in range(6):
                theta = i * np.pi / 3
                positions.append((2.5*np.cos(theta), 2.5*np.sin(theta), 0))
            for i in range(5):
                theta = i * 2*np.pi / 5
                positions.append((5*np.cos(theta), 5*np.sin(theta), 0))
        elif pattern == 'random':
            positions = []
            for i in range(N):
                x = np.random.uniform(-5, 5)
                y = np.random.uniform(-5, 5)
                a = np.random.uniform(0, 60)
                positions.append((x, y, a))
        else:
            positions = []
            for i in range(N):
                x = np.random.uniform(-5, 5)
                y = np.random.uniform(-5, 5)
                a = 0
                positions.append((x, y, a))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 7.0
        params[3*N+3] = 0.0
        return params

    best_params = initial_guess('hex')
    best_score = objective(best_params)

    patterns = ['hex', 'circle', 'random', 'random', 'random']
    
    for p_idx, pattern in enumerate(patterns):
        if time.time() > deadline - 30:
            break
        
        params = initial_guess(pattern)
        
        # First try Nelder-Mead
        try:
            result = minimize(objective, params, method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
        
        # Then try L-BFGS-B from the best point
        if time.time() > deadline - 30:
            break
        
        try:
            result2 = minimize(objective, best_params, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-4})
            
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    # Extract best result
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    side = float(best_params[3*N+2])
    angle = float(best_params[3*N+3])

    # Final validation with a more careful check
    def is_valid(inner, center, side, angle):
        outer_center = np.array(center)
        outer_angle = angle
        outer_verts = hexagon_vertices(outer_center[0], outer_center[1], side, outer_angle)
        
        inner_verts = []
        for i in range(N):
            v = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            inner_verts.append(v)
        
        # Check containment
        for i in range(N):
            outer_edges = []
            for k in range(6):
                outer_edges.append(outer_verts[k] - outer_verts[(k+1) % 6])
            
            for edge in outer_edges:
                norm = np.linalg.norm(edge)
                if norm < 1e-12:
                    continue
                normal = np.array([-edge[1], edge[0]])
                normal = normal / norm
                mid = (outer_verts[0] + outer_verts[1]) / 2
                center_proj = np.dot(outer_center, normal)
                mid_proj = np.dot(mid, normal)
                if center_proj < mid_proj:
                    normal = -normal
                
                inner_proj = np.dot(inner_verts[i], normal)
                min_inner = np.min(inner_proj)
                edge_point = outer_verts[0]
                edge_proj = np.dot(edge_point, normal)
                
                if min_inner < edge_proj - 1e-9:
                    return False
        
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                sep = sat_separation(inner_verts[i], inner_verts[j])
                if sep < -1e-9:
                    return False
        
        return True

    if not is_valid(inner, center, side, angle):
        # Try to repair by increasing side slightly
        for _ in range(100):
            side += 0.01
            if is_valid(inner, center, side, angle):
                break
        else:
            # Fallback
            inner = [[0.0, 0.0, 0.0]] * N
            center = (0.0, 0.0)
            side = 100.0
            angle = 0.0

    return inner, center, side, angle
# EVOLVE-BLOCK-END
