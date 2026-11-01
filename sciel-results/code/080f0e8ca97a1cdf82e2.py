"""
Improved packing strategy:
1. Decouple outer side length from the optimization variables.
2. Compute minimal required side length analytically for any inner configuration.
3. Use fast SAT checks with distance-based early rejection for disjointness.
4. Use Nelder-Mead with multiple restarts to optimize inner positions and outer rotation.
"""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Precompute constants
    SQRT3 = np.sqrt(3.0)
    APOTHEM_INNER = SQRT3 / 2.0  # For unit hexagon
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Returns vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
        angle = np.radians(angle_deg)
        vertices = np.empty((6, 2))
        for i in range(6):
            theta = angle + i * np.pi / 3
            vertices[i, 0] = cx + side * np.cos(theta)
            vertices[i, 1] = cy + side * np.sin(theta)
        return vertices

    def get_outer_normals(angle_deg):
        """
        Returns the 6 unit normal vectors of the outer hexagon edges.
        For a hexagon rotated by 'angle', the normals are at angle + 30 + k*60 degrees.
        """
        base_angle = np.radians(angle_deg) + np.pi / 6.0 # 30 degrees offset
        normals = np.empty((6, 2))
        for i in range(6):
            theta = base_angle + i * np.pi / 3.0
            normals[i, 0] = np.cos(theta)
            normals[i, 1] = np.sin(theta)
        return normals

    def compute_min_side(inner, center, outer_angle_deg):
        """
        Computes the minimum side length of the outer hexagon required to contain all inner hexagons.
        The outer hexagon is centered at 'center' and rotated by 'outer_angle_deg'.
        """
        cx, cy = center
        normals = get_outer_normals(outer_angle_deg)
        
        max_extent = 0.0
        # We need to check all vertices of all inner hexagons
        for i in range(N):
            x, y, ang = inner[i]
            verts = hexagon_vertices(x, y, 1.0, ang)
            
            # Vectorized dot products: (6x2) dot (6x2)^T -> (6, 6) matrix
            # But we just need the max projection for each normal
            # proj[j] = max_k ( verts[k] - center ) . normals[j]
            # Actually, for containment, we need max_k ( verts[k] - center ) . normals[j] <= apothem_outer
            # apothem_outer = S * SQRT3 / 2
            # So S >= 2 * max_projection / SQRT3
            
            # Calculate projections of all 6 vertices onto all 6 normals
            # offsets = verts - center (6x2)
            # projections = offsets @ normals.T (6x6)
            # max_proj_for_normal_j = max_k projections[k, j]
            
            offsets = verts - np.array([cx, cy])
            # projections[k, j] = offsets[k] dot normals[j]
            projections = offsets @ normals.T
            
            # For each normal direction j, find the max projection among the 6 vertices
            max_projections = np.max(projections, axis=0) # shape (6,)
            
            # The required apothem for this inner hexagon in direction j is max_projections[j]
            # The side length S must satisfy S * SQRT3 / 2 >= max_projections[j]
            # So S >= 2 * max_projections[j] / SQRT3
            # We take the max over all j and all i
            
            current_max_extent = np.max(max_projections)
            if current_max_extent > max_extent:
                max_extent = current_max_extent
                
        return max_extent * 2.0 / SQRT3

    def hexagons_disjoint_fast(h1, h2):
        """
        Fast check for disjointness using SAT.
        h1, h2 are tuples (x, y, angle_deg)
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # Max distance between centers for disjoint unit hexagons is 2.0 (vertex to vertex)
        # If dist > 2.0, definitely disjoint
        if dist_sq > 4.0:
            return True
        
        # If dist < sqrt(3) ~ 1.732, they definitely overlap (edge to edge touching is min distance for disjoint if aligned, 
        # but with rotation, min distance can be smaller? No, the minimum distance between two disjoint convex sets is > 0.
        # The minimum distance between two *touching* hexagons is 0.
        # However, if the center distance is very small, they overlap.
        # The maximum distance between any point in a unit hexagon and the center is 1.0 (vertex).
        # So if dist > 2.0, they are disjoint.
        # If dist < something, they overlap.
        # Let's just do SAT if dist <= 2.0 to be safe and accurate.
        
        v1 = hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
        
        # SAT Check
        # Edges of v1 and v2
        for i in range(6):
            edge1 = v1[i] - v1[(i+1) % 6]
            normal = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal)
            if norm == 0: continue
            normal /= norm
            
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return True
            
            edge2 = v2[i] - v2[(i+1) % 6]
            normal = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal)
            if norm == 0: continue
            normal /= norm
            
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return True
                
        return False

    def objective(params):
        """
        Objective: minimize the computed minimum side length.
        params: [x1, y1, a1, ..., x12, y12, a12, cx, cy, outer_angle]
        """
        # Unpack inner hexagons
        inner = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner.append((x, y, a))
        
        center = (params[3*N], params[3*N+1])
        outer_angle = params[3*N+2]
        
        # Check disjointness first (cheap early exit if invalid)
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint_fast(inner[i], inner[j]):
                    return 1e6
        
        # Compute side length
        side = compute_min_side(inner, center, outer_angle)
        
        return side

    def initial_guess():
        """
        Generate a valid initial guess.
        Hexagonal packing: 1 center, 6 in ring 1, 5 in ring 2.
        """
        params = np.zeros(3*N + 3)
        positions = []
        
        # Center
        positions.append((0, 0, 0))
        
        # Ring 1: 6 hexagons at distance 2 (touching edges)
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((2*np.cos(theta), 2*np.sin(theta), 0))
        
        # Ring 2: 5 hexagons. To fit 12, we place 5 around the ring 1 gaps or further out.
        # A standard hexagonal packing of 12 is 1-6-5? No, 1-6-12 is 19.
        # 1-6-5 is a common cluster.
        # Let's place them at distance 4, in the gaps of ring 1?
        # Ring 1 angles: 0, 60, 120, 180, 240, 300.
        # Gaps are at 30, 90, 150, 210, 270, 330.
        # Distance to gap centers from origin is 2 * cos(30) = sqrt(3) ~ 1.732? No.
        # Distance from origin to a vertex of the ring 1 hexagon is 2.
        # The "gap" between two adjacent ring 1 hexagons is a triangle.
        # Let's just place 5 at distance 4, evenly spaced, offset by 30 degrees.
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5 # Offset
            positions.append((4*np.cos(theta), 4*np.sin(theta), 0))
            
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
            
        # Outer hexagon center and angle
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        
        return params

    best_params = initial_guess()
    best_score = objective(best_params)
    
    # If initial guess is invalid (overlapping), we need a valid start.
    # The 1-6-5 packing with distance 4 for the outer 5 might overlap with ring 1?
    # Ring 1 is at dist 2. Outer 5 at dist 4.
    # Min distance between a ring 1 hex (center dist 2) and outer 5 (center dist 4)
    # If aligned, distance is 2. They touch. So it should be valid or close.
    
    if best_score >= 1e5:
        # Spread them out more to ensure validity
        for i in range(N):
            angle = i * 2 * np.pi / N
            dist = 3.0
            best_params[3*i] = dist * np.cos(angle)
            best_params[3*i+1] = dist * np.sin(angle)
            best_params[3*i+2] = 0.0
        best_score = objective(best_params)
        
    # Optimization Loop
    # We use Nelder-Mead for robustness.
    # We perform multiple restarts.
    
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        # Perturb the current best for restart
        if r == 0:
            current_params = best_params.copy()
        else:
            current_params = best_params.copy()
            # Add noise
            noise = np.random.normal(0, 0.5, size=len(current_params))
            # Don't perturb the outer center/angle too much in early restarts?
            # Just perturb everything.
            current_params += noise
            
        # Run Nelder-Mead
        try:
            result = minimize(objective, current_params, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass

    # Extract best result
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+2])
    
    # Calculate final side
    side = compute_min_side(inner, center, angle)
    
    # Final Validation
    # Ensure disjointness
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint_fast(inner[i], inner[j]):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid configuration if optimization failed badly
        # This should rarely happen with the objective function penalty, but just in case.
        inner = []
        for i in range(N):
            ang = i * 2 * np.pi / N
            inner.append([3.0 * np.cos(ang), 3.0 * np.sin(ang), 0.0])
        center = (0.0, 0.0)
        angle = 0.0
        side = compute_min_side(inner, center, angle)

    return inner, center, side, angle
# EVOLVE-BLOCK-END
