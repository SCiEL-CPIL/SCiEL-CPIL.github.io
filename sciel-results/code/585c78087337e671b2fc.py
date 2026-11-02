"""Rewritten program: Honeycomb Lattice Deformation with Smooth MTV Penalty and Nelder-Mead."""
import numpy as np
import time
import random
import scipy.optimize

N = 12
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Returns vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def min_side_length(inner, center, angle_deg):
        """
        Computes the minimum side length of a regular hexagon centered at (center) 
        with rotation angle_deg that contains all inner hexagons.
        """
        max_extent = 0.0
        # Precompute all vertices for all inner hexagons
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        offsets = all_verts - np.array(center)
        
        # Edge normals of outer hexagon
        for k in range(6):
            normal_angle = np.radians(angle_deg) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            # Dot product of all offsets with normal
            extents = np.dot(offsets, normal)
            max_e = np.max(extents)
            if max_e > max_extent:
                max_extent = max_e
        
        return max_extent * 2.0 / SQRT3

    def get_mtv_depth(v1, v2):
        """
        Returns the minimum translation depth to separate two convex polygons v1 and v2.
        If they are disjoint, returns 0.
        If they overlap, returns the positive depth.
        Uses Separating Axis Theorem.
        """
        min_overlap = float('inf')
        
        # Axes from v1
        for i in range(6):
            edge = v1[(i+1)%6] - v1[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10: continue
            normal /= norm
            
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            if max1 < min2 or max2 < min1:
                return 0.0 # Disjoint
            
            # Calculate overlap on this axis
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap:
                min_overlap = overlap
        
        # Axes from v2
        for i in range(6):
            edge = v2[(i+1)%6] - v2[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10: continue
            normal /= norm
            
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            if max1 < min2 or max2 < min1:
                return 0.0 # Disjoint
            
            # Calculate overlap on this axis
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap:
                min_overlap = overlap
                
        return min_overlap if min_overlap > 0 else 0.0

    def objective(params):
        """
        Smooth objective function: Minimize outer side length + Penalty for overlaps.
        Penalty is sum of squared MTV depths.
        """
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        
        side = min_side_length(inner, center, angle)
        
        penalty = 0.0
        # Precompute vertices for penalty calculation
        verts = [hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)]
        
        for i in range(N):
            for j in range(i+1, N):
                # Fast rejection: if center distance > 2.0, no overlap (max distance between centers of touching hexagons is 2.0? No, circumradius is 1, so distance 2 means touching at vertices. Inscribed radius is sqrt(3)/2 ~ 0.866. Distance 2*0.866=1.732 is touching at flat sides. So if dist > 2.0, definitely disjoint. If dist < 1.732, might overlap.
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                dist_sq = dx*dx + dy*dy
                if dist_sq > 4.0: # 2.0^2
                    continue
                
                depth = get_mtv_depth(verts[i], verts[j])
                if depth > 0:
                    penalty += depth * depth
        
        return side + 100.0 * penalty

    def is_valid(inner):
        """Strict validity check."""
        verts = [hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                if dx*dx + dy*dy > 4.0:
                    continue
                depth = get_mtv_depth(verts[i], verts[j])
                if depth > 1e-6: # Small tolerance
                    return False
        return True

    def initial_honeycomb():
        """
        Generates parameters for a honeycomb-like cluster.
        1 center, 6 ring 1, 5 ring 2.
        """
        params = np.zeros(3*N + 3)
        positions = []
        # Center
        positions.append((0, 0, 0))
        # Ring 1: Distance 2 * (sqrt(3)/2 * 2 / cos(30)? No.
        # For unit hexagons, center-to-center distance for touching is 2 * (sqrt(3)/2) * 2 / 2?
        # Height of hexagon is sqrt(3). Width is 2.
        # Touching on flat sides: distance is sqrt(3).
        # Touching on vertices: distance is 2.
        # Honeycomb lattice vectors:
        # a1 = (1.5, sqrt(3)/2) * 2 ? No.
        # Let's use the standard honeycomb coordinates.
        # Axial coords (q, r).
        # Cartesian: x = 1.5 * q, y = sqrt(3) * (r + q/2).
        # Distance between (0,0) and (1,0) is 1.5.
        # Distance between (0,0) and (0,1) is sqrt(3).
        # These are touching distances.
        
        # We want 12 hexagons.
        # Center (0,0).
        # Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
        # These 6 are at distance 1.5 or sqrt(3).
        # Next ring: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)?
        # Let's pick 5 from the second ring to make 12 total.
        
        def axial_to_cartesian(q, r):
            x = 1.5 * q
            y = SQRT3 * (r + q/2.0)
            return x, y

        positions.append((0,0,0))
        
        ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        for q,r in ring1:
            x,y = axial_to_cartesian(q,r)
            positions.append((x,y,0))
            
        ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,1)]
        for q,r in ring2:
            x,y = axial_to_cartesian(q,r)
            positions.append((x,y,0))
            
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
            
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    def initial_random():
        params = np.zeros(3*N + 3)
        for i in range(N):
            params[3*i] = random.uniform(-3, 3)
            params[3*i+1] = random.uniform(-3, 3)
            params[3*i+2] = random.uniform(0, 60)
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    best_score = float('inf')
    best_params = None

    # Multi-start
    starts = [initial_honeycomb(), initial_random(), initial_random()]
    
    for start_idx, init_p in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
            
        try:
            # Use Nelder-Mead for robustness on non-smooth (but mostly smooth) objective
            result = scipy.optimize.minimize(
                objective, 
                init_p, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False}
            )
            
            if result.fun < best_score:
                # Check validity
                p = result.x
                inner = [(p[3*i], p[3*i+1], p[3*i+2]) for i in range(N)]
                if is_valid(inner):
                    side = min_side_length(inner, (p[3*N], p[3*N+1]), p[3*N+2])
                    if side < best_score:
                        best_score = side
                        best_params = p.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback
        best_params = initial_honeycomb()
        inner = [(best_params[3*i], best_params[3*i+1], best_params[3*i+2]) for i in range(N)]
        best_score = min_side_length(inner, (best_params[3*N], best_params[3*N+1]), best_params[3*N+2])

    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+2])
    side = min_side_length(inner, center, angle)
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
