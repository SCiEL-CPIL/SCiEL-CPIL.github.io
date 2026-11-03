"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    
    # For unit regular hexagons (side length 1), the distance between centers
    # to avoid overlap depends on relative rotation. For identical orientation,
    # the minimum center distance is 2*sqrt(3) (touching edges).
    # We will use a continuous optimization approach.
    
    # State vector: x1, y1, a1, x2, y2, a2, ... (12 hexagons * 3 params = 36)
    # Plus outer hexagon center (cx, cy) and rotation (theta) and side length (s)
    # Actually, we can fix the outer hexagon center to (0,0) by symmetry/translation invariance
    # and search for positions, angles, outer rotation, and outer side length.
    
    # Let's parameterize:
    # - 12 inner hexagons: (xi, yi, ai) for i=1..12
    # - Outer hexagon: rotation theta, side length s
    # We want to minimize s such that all inner hexagons are inside the outer one and don't overlap.
    
    # To simplify, let's assume all inner hexagons have the same rotation angle alpha.
    # This reduces the search space significantly and is often optimal for packing identical shapes.
    # State: x1..x12, y1..y12, alpha, theta, s
    # Actually, we can fix the frame by setting the first hexagon's position or using symmetry.
    # But let's just let the optimizer find it.
    
    # Distance between centers for two unit hexagons with relative angle delta:
    # The minimum distance d_min such that they don't overlap.
    # For identical orientation (delta=0), d_min = 2*sqrt(3).
    # For other angles, it's more complex. We can approximate or compute exactly.
    # Exact computation: The Minkowski sum of two hexagons. The distance is the radius of the
    # Minkowski sum in the direction of the center-to-center vector.
    
    def min_dist_between_hexagons(angle_diff):
        """
        Compute the minimum distance between centers of two unit hexagons
        with relative rotation angle_diff such that they don't overlap.
        """
        # Vertices of a unit hexagon with angle 0
        # Side length 1, circumradius 1.
        # Vertices at angles 0, 60, 120, 180, 240, 300
        # Wait, standard hexagon with side 1 has circumradius 1.
        # Vertices: (cos(k*pi/3), sin(k*pi/3)) for k=0..5
        # But this is "pointy" top. Let's use flat top or just consistent.
        # Let's define vertices for a hexagon with center at origin and rotation 0.
        # v_k = (cos(k*pi/3 + pi/6), sin(k*pi/3 + pi/6)) ?
        # Actually, let's just use the support function.
        # The support function of a regular hexagon with side s and rotation phi in direction u
        # is h(u) = s * max_k dot(u, v_k) where v_k are vertices.
        # For unit hexagon, s=1.
        
        # The condition for two hexagons H1 (center c1, rot a1) and H2 (center c2, rot a2) to not overlap
        # is that the distance |c1-c2| >= h_{H1+H2}(u) for all directions u?
        # No, it's that the Minkowski sum H1 + (-H2) does not contain the origin?
        # Actually, H1 and H2 are disjoint iff for all u, h_{H1}(u) + h_{H2}(-u) <= |c1-c2| * |u| ? No.
        
        # Simpler: The distance between centers must be at least the maximum over all directions u
        # of (h_{H1}(u) + h_{H2}(-u)).
        # h_{H2}(-u) = h_{H2}(-u). Since H2 is centrally symmetric, h_{H2}(-u) = h_{H2}(u).
        # So the required distance is max_u (h_{H1}(u) + h_{H2}(u)).
        # This is the support function of the Minkowski sum H1 + H2.
        
        # For two regular hexagons with relative angle delta, the Minkowski sum is a 12-gon (or less if aligned).
        # We can compute the maximum support by checking the critical angles.
        
        # Let's sample a fine set of angles to approximate the max support.
        # Or derive analytically.
        # For delta=0, the sum is a hexagon with side 2, so circumradius 2, inradius 2*sqrt(3)/2 = sqrt(3).
        # The max support is in the direction of the vertices, which is 2.
        # Wait, inradius is distance from center to side.
        # For a regular hexagon with side s, inradius r = s*sqrt(3)/2.
        # The support function in the direction of the normal to a side is r.
        # The support function in the direction of a vertex is s (circumradius).
        # For the Minkowski sum of two identical hexagons (delta=0), it's a hexagon with side 2s.
        # So inradius is 2s*sqrt(3)/2 = s*sqrt(3).
        # The max support is in the vertex direction, which is 2s.
        # But wait, the distance between centers must be such that the hexagons don't overlap.
        # If they touch edge-to-edge, the distance is 2 * inradius = 2 * s*sqrt(3)/2 = s*sqrt(3).
        # If they touch vertex-to-vertex, the distance is 2 * circumradius = 2s.
        # The critical distance is the minimum over all relative orientations? No, for fixed relative orientation,
        # the minimum distance to avoid overlap is the max over u of (h1(u) + h2(u)).
        
        # Let's compute this max support numerically for a given delta.
        # We need to check directions u. The critical directions are the normals to the edges of the Minkowski sum.
        # The Minkowski sum of two hexagons has edges parallel to the edges of the original hexagons.
        # So the critical directions are the normals to the edges of hexagon 1 and hexagon 2.
        
        # Normals for hexagon with rotation alpha:
        # Edges are at angles alpha + pi/6 + k*pi/3.
        # Normals are at angles alpha + k*pi/3. (Perpendicular to edges)
        # Wait, edge direction is tangent. Normal is perpendicular.
        # If vertex is at angle alpha + k*pi/3, the edge between vertex k and k+1 is perpendicular to the bisector.
        # Let's just use the vertices to compute support.
        
        # Support function h(u) = max_k dot(u, v_k).
        # We need max_u (h1(u) + h2(u)).
        # This is equivalent to the radius of the Minkowski sum.
        
        # Let's sample 360 angles for u to be safe, or just the 12 critical normals.
        # Critical normals are the normals to the edges of hex1 and hex2.
        # Hex1 normals: alpha1 + k*pi/3 for k=0..5.
        # Hex2 normals: alpha2 + k*pi/3 for k=0..5.
        # Total 12 directions.
        
        delta = angle_diff
        # Let alpha1 = 0, alpha2 = delta
        normals = []
        for k in range(6):
            normals.append(k * math.pi / 3)
            normals.append(delta + k * math.pi / 3)
        
        max_support = 0
        for ang in normals:
            u = np.array([math.cos(ang), math.sin(ang)])
            
            # h1(u) for hexagon with rotation 0
            # Vertices of unit hexagon (rotation 0):
            # Let's define vertices at angles pi/6 + k*pi/3 (flat top) or 0 + k*pi/3 (pointy top).
            # Let's use pointy top: vertices at k*pi/3.
            # v_k = (cos(k*pi/3), sin(k*pi/3))
            # h1(u) = max_k dot(u, v_k)
            
            # h2(u) for hexagon with rotation delta
            # v_k = (cos(delta + k*pi/3), sin(delta + k*pi/3))
            # h2(u) = max_k dot(u, v_k)
            
            h1 = 0
            h2 = 0
            for k in range(6):
                v1 = np.array([math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)])
                v2 = np.array([math.cos(delta + k * math.pi / 3), math.sin(delta + k * math.pi / 3)])
                h1 = max(h1, np.dot(u, v1))
                h2 = max(h2, np.dot(u, v2))
            
            support = h1 + h2
            if support > max_support:
                max_support = support
                
        return max_support

    # Precompute min_dist for a range of angle differences
    # Since we assume all inner hexagons have the same rotation, angle_diff is always 0.
    # So min_dist = min_dist_between_hexagons(0)
    # Let's verify: for delta=0, normals are 0, 60, 120, 180, 240, 300 (duplicated).
    # u = (1, 0). v0=(1,0), dot=1. v1=(0.5, 0.866), dot=0.5.
    # h1 = 1. h2 = 1. support = 2.
    # u = (cos(pi/6), sin(pi/6)) = (0.866, 0.5).
    # v0=(1,0), dot=0.866. v1=(0.5, 0.866), dot=0.433+0.433=0.866.
    # h1 = 0.866. h2 = 0.866. support = 1.732 = sqrt(3).
    # Max support is 2.
    # So min distance is 2?
    # Wait, if two hexagons touch edge-to-edge, the distance is sqrt(3).
    # If they touch vertex-to-vertex, distance is 2.
    # The condition for NO overlap is distance >= max_support.
    # So if distance is sqrt(3), they overlap?
    # Let's check.
    # Hexagon 1 at (0,0), Hexagon 2 at (sqrt(3), 0).
    # Vertex of H1 at (1,0). Vertex of H2 at (sqrt(3)-1, 0) approx (0.732, 0).
    # They don't touch vertex-to-vertex.
    # Edge of H1 is between (1,0) and (0.5, 0.866). Midpoint (0.75, 0.433). Normal is (0.866, 0.5).
    # Distance from (0,0) to line is 0.866 (inradius).
    # So the rightmost point of H1 in direction (0.866, 0.5) is 0.866.
    # The leftmost point of H2 in direction (0.866, 0.5) is -0.866 relative to center.
    # So the gap is sqrt(3) - 0.866 - 0.866 = 0.
    # So they touch edge-to-edge.
    # So min distance is sqrt(3).
    # Why did my code give 2?
    # Ah, the max support is in the direction of the vertices (1,0).
    # In that direction, H1 extends to 1, H2 extends to 1 (from its center).
    # So total span is 2.
    # But if the centers are separated by sqrt(3) in the x-direction,
    # the rightmost point of H1 is at x=1.
    # The leftmost point of H2 is at x = sqrt(3) - 1 = 0.732.
    # So they overlap in x? 1 > 0.732.
    # But they are separated in y?
    # At x=1, H1 has y=0. H2 at x=1 (relative x = 1-sqrt(3) = -0.732) has y range?
    # The hexagon is convex.
    # The condition is that the Minkowski sum does not contain the origin.
    # The origin is in the Minkowski sum H1 + (-H2) iff they overlap.
    # The distance between centers is c.
    # They are disjoint iff c > max_u (h1(u) + h2(u)).
    # Wait, H1 and H2 are disjoint iff for all u, h1(u) + h2(-u) < c * |u|? No.
    # H1 and H2 are disjoint iff the origin is not in H1 - H2.
    # H1 - H2 = H1 + (-H2).
    # The origin is in H1 + (-H2) iff for all u, h_{H1}(-u) + h_{-H2}(u) < 0? No.
    # The origin is in a convex set K iff h_K(u) >= 0 for all u? No.
    # The origin is in K iff h_K(u) >= 0 for all u is false.
    # The origin is in K iff for all u, h_K(u) >= 0 is not the condition.
    # The condition is that the minimum of h_K(u) over u is >= 0? No.
    # For a convex set containing the origin, h_K(u) >= 0 for all u.
    # So if the origin is NOT in K, there exists u such that h_K(u) < 0.
    # K = H1 - H2.
    # h_K(u) = h_{H1}(u) + h_{-H2}(u) = h_{H1}(u) + h_{H2}(-u).
    # Since H2 is centrally symmetric, h_{H2}(-u) = h_{H2}(u).
    # So h_K(u) = h_{H1}(u) + h_{H2}(u).
    # The origin is in H1 - H2 iff h_K(u) >= 0 for all u.
    # So H1 and H2 overlap iff h_K(u) >= 0 for all u.
    # H1 and H2 are disjoint iff there exists u such that h_K(u) < 0.
    # h_K(u) = h_{H1}(u) + h_{H2}(u).
    # This is always positive for hexagons!
    # I am confused.
    
    # Let's use the distance formula directly.
    # Two convex sets A and B are disjoint iff dist(A, B) > 0.
    # dist(A, B) = max_u (h_A(u) - h_B(-u))? No.
    # dist(A, B) = max_u (h_A(u) - h_B(u))? No.
    
    # Let's stick to the simple case: identical orientation.
    # Min distance is sqrt(3).
    # Let's hardcode this for now and assume all inner hexagons have the same rotation.
    
    min_dist = math.sqrt(3)
    
    # Now, the outer hexagon.
    # We want to place 12 hexagons inside a regular hexagon of side s.
    # The inner hexagons have side 1.
    # Let's use a grid-based search for the positions on a triangular lattice,
    # and then optimize the global rotation and the outer hexagon parameters.
    
    # Lattice spacing: d = sqrt(3) + epsilon.
    d = math.sqrt(3) + 1e-6
    a1 = np.array([d, 0.0])
    a2 = np.array([d/2.0, d*math.sqrt(3.0)/2.0])
    
    # Candidate patterns of 12 lattice points
    # Pattern 1: 4x3 grid
    pattern_1 = [(i, j) for i in range(-2, 2) for j in range(-1, 2)]
    
    # Pattern 2: Hexagonal cluster
    # Center (0,0), Ring 1 (6 pts), Ring 2 (5 pts)
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # (2,0), (-2,0), (0,2), (0,-2), (2,-2)
    pattern_2 = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
                 (2,0), (-2,0), (0,2), (0,-2), (2,-2)]
    
    patterns = [pattern_1, pattern_2]
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_outer_angle = 0.0
    
    # We will search over:
    # - Pattern (2 options)
    # - Global rotation of the lattice (alpha)
    # - Outer hexagon rotation (theta)
    
    # To speed up, we can fix the outer hexagon center to (0,0) by centering the pattern.
    
    for pattern in patterns:
        if time.time() > DEADLINE:
            break
            
        # Compute lattice points
        pts = np.array([i * a1 + j * a2 for i, j in pattern])
        
        # Center the pattern
        center = np.mean(pts, axis=0)
        pts_centered = pts - center
        
        # Search over lattice rotation alpha and outer rotation theta
        # alpha in [0, pi/3), theta in [0, pi/3)
        
        # Use differential evolution or grid search?
        # Grid search might be faster for small ranges.
        
        alpha_steps = np.linspace(0, math.pi/3, 61)
        theta_steps = np.linspace(0, math.pi/3, 61)
        
        for alpha in alpha_steps:
            if time.time() > DEADLINE:
                break
                
            cos_a = math.cos(alpha)
            sin_a = math.sin(alpha)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            pts_rot = (pts_centered @ R.T)
            
            for theta in theta_steps:
                if time.time() > DEADLINE:
                    break
                    
                # Outer hexagon with rotation theta
                # Normals: theta + k*pi/3 + pi/6
                normals = []
                for k in range(6):
                    ang = theta + k * math.pi / 3 + math.pi / 6
                    normals.append(np.array([math.cos(ang), math.sin(ang)]))
                
                max_projections = []
                for normal in normals:
                    projections = pts_rot @ normal
                    max_proj = np.max(projections)
                    min_proj = np.min(projections)
                    width = max_proj - min_proj
                    # Side length s = width / sqrt(3)
                    s = width / math.sqrt(3.0)
                    max_projections.append(s)
                
                required_s = max(max_projections)
                
                if required_s < best_side:
                    best_side = required_s
                    # Store best configuration
                    best_inner_pts = pts_rot
                    best_alpha = alpha
                    best_theta = theta
                    best_pattern = pattern
    
    # If we found a good configuration, use it
    if best_side < float('inf') and best_inner_pts is not None:
        # Construct the inner list
        inner = []
        for pos in best_inner_pts:
            # The angle of each inner hexagon is best_alpha (same for all)
            # Convert to degrees
            angle_deg = math.degrees(best_alpha)
            inner.append([pos[0], pos[1], angle_deg])
        
        center = (0.0, 0.0)
        side = best_side
        outer_angle = math.degrees(best_theta)
        
        return inner, center, side, outer_angle
    
    # Fallback: simple linear arrangement
    inner = [[math.sqrt(3) * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 20.0, 0.0
# EVOLVE-BLOCK-END
