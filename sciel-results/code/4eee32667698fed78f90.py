"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Unit regular hexagon geometry
    # Flat-topped hexagon: vertices at angles 0, 60, 120, 180, 240, 300
    # Width (flat-to-flat) = 2, Height (point-to-point) = sqrt(3)
    # In hex grid, neighbors are at distance sqrt(3) in 3 directions
    
    # For hex grid with axial coordinates (q, r):
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # Adjacent hexagons are sqrt(3) apart
    
    # Strategy: place N hexagons in a hexagonal cluster pattern
    # Then compute minimal enclosing hexagon
    
    def hex_grid_positions(n):
        """Generate n positions in a hex grid, centered at origin."""
        positions = [(0, 0)]
        # Add rings around center
        ring = 1
        while len(positions) < n:
            # Ring r has 6*r hexagons
            for i in range(6 * ring):
                if len(positions) >= n:
                    break
                # Position on ring r, step i
                # Directions: E, SE, S, SW, W, NE (going around)
                # Axial coords for ring r:
                # Start at (r, 0), go to (r, -r), (0, -r), (-r, 0), (-r, r), (0, r)
                if i < ring:
                    q = ring - i
                    r_coord = -i
                elif i < 2 * ring:
                    q = ring
                    r_coord = -(i - ring) + ring - ring  # This is getting complex
                    # Let me use a cleaner approach
                    q = ring - (i - ring)  # Wait, let me redo this
                    # Actually for ring r, the 6 sides:
                    # Side 0 (i=0..r-1): (r-i, -i) -> no
                    # Let me use the standard ring generation
                    pass
                # Let me use a simpler explicit approach
            ring += 1
        return positions[:n]
    
    # Cleaner hex grid generation
    def generate_hex_positions(n):
        """Generate n hex grid positions centered at origin using axial coords."""
        if n == 1:
            return [(0, 0)]
        
        positions = [(0, 0)]
        ring = 1
        while len(positions) < n:
            # Ring r consists of 6r hexagons
            # Start at (r, 0) and go clockwise
            # The 6 directions in axial coords:
            # E: (1, 0), SE: (0, -1), S: (-1, 1), SW: (-1, 0), W: (0, 1), NE: (1, -1)
            # Wait, standard axial: neighbors of (q,r) are:
            # (q+1, r), (q+1, r-1), (q, r-1), (q-1, r), (q-1, r+1), (q, r+1)
            
            # For ring r, start at (r, 0)
            q, r_coord = ring, 0
            for step in range(6 * ring):
                if len(positions) >= n:
                    break
                if (q, r_coord) not in positions:
                    positions.append((q, r_coord))
                # Move to next position on ring
                # Directions to traverse ring: SE, S, SW, W, NW, NE (in some order)
                # Actually, let's just enumerate all positions at distance r
            # Simpler: enumerate all (q, r) with max(|q|, |r|, |q+r|) == ring
            for q in range(-ring, ring + 1):
                for r_coord in range(-ring, ring + 1):
                    if max(abs(q), abs(r_coord), abs(q + r_coord)) == ring:
                        if (q, r_coord) not in positions and len(positions) < n:
                            positions.append((q, r_coord))
            ring += 1
        
        return positions[:n]
    
    def axial_to_xy(q, r):
        """Convert axial hex coords to pixel coords for flat-topped hexagons."""
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    # Generate positions
    hex_positions = generate_hex_positions(N)
    
    # Convert to xy
    xy_positions = [axial_to_xy(q, r) for q, r in hex_positions]
    
    # Now compute the minimal enclosing regular hexagon
    # A regular hexagon centered at (cx, cy) with side length s and rotation theta
    # contains point (x, y) if the distance in all 3 hex directions is <= s
    
    # For a flat-topped hexagon with side s, centered at origin:
    # The hexagon is the intersection of 3 strips:
    # |x| <= s (in the x direction, but actually it's more complex)
    
    # Actually, for a regular hexagon with "radius" (center to vertex) = R = s:
    # Wait, for a regular hexagon, side length = circumradius = R
    # The inradius (apothem) = R * sqrt(3)/2
    
    # For a hexagon oriented with flat top (vertex at 0 degrees):
    # Vertices at (R*cos(k*60), R*sin(k*60)) for k=0..5
    # The hexagon is: |x| <= R, |y| <= R*sqrt(3)/2, |x*sqrt(3)/2 + y/2| <= R
    # Wait, let me think more carefully.
    
    # A regular hexagon with side length s has circumradius R = s.
    # For orientation with a vertex pointing right (0 degrees):
    # The 6 vertices are at angles 0, 60, 120, 180, 240, 300 degrees
    # Vertex k: (R*cos(k*60°), R*sin(k*60°))
    
    # The hexagon can be described as the set of points where:
    # |dot(p, n_i)| <= apothem for i = 0,1,2 (three pairs of parallel sides)
    # where n_i are unit normals to the sides.
    
    # For vertex-at-0° orientation:
    # Side normals are at 30°, 90°, 150° (and their opposites)
    # Apothem = R * cos(30°) = R * sqrt(3)/2
    
    # So constraints:
    # |x*cos(30°) + y*sin(30°)| <= R*sqrt(3)/2
    # |x*cos(90°) + y*sin(90°)| <= R*sqrt(3)/2  => |y| <= R*sqrt(3)/2
    # |x*cos(150°) + y*sin(150°)| <= R*sqrt(3)/2
    
    # For a rotated hexagon by angle theta, the normals rotate too.
    
    def min_enclosing_hexagon(points, max_iter=100):
        """Find minimal enclosing regular hexagon for given points."""
        if not points:
            return (0.0, 0.0, 0.0, 0.0)
        
        best_s = float('inf')
        best_cx, best_cy, best_theta = 0.0, 0.0, 0.0
        
        # Try different rotations
        for theta_deg in range(0, 30, 2):  # 0 to 28 degrees in steps of 2
            if time.time() > deadline:
                break
            theta = math.radians(theta_deg)
            
            # For this rotation, find minimal s and center
            # The hexagon with center (cx, cy), side s, rotation theta contains
            # point (x, y) if for all 3 normal directions, the projection is bounded.
            
            # Normal directions (unit vectors) for the 3 pairs of sides:
            # Base normals at 30°, 90°, 150° for theta=0
            # After rotation by theta: 30°+theta, 90°+theta, 150°+theta
            
            normals = []
            for base_angle in [30, 90, 150]:
                a = math.radians(base_angle) + theta
                normals.append((math.cos(a), math.sin(a)))
            
            # For center (cx, cy), the constraint for point (x, y) is:
            # |dot((x-cx, y-cy), n_i)| <= s * sqrt(3)/2 for all i
            
            # This is a linear programming problem: minimize s such that
            # |dot((x_i - cx, y_i - cy), n_j)| <= s * sqrt(3)/2 for all i, j
            
            # Let's use a simple iterative approach:
            # Start with center at centroid, then adjust.
            
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                # Compute required s for current center
                s_needed = 0.0
                for (x, y) in points:
                    for (nx, ny) in normals:
                        proj = abs((x - cx) * nx + (y - cy) * ny)
                        s_candidate = proj / (math.sqrt(3) / 2)
                        if s_candidate > s_needed:
                            s_needed = s_candidate
                
                # Try to improve center by moving toward the most constraining point
                # Simple heuristic: try small perturbations
                best_local_s = s_needed
                best_local_cx, best_local_cy = cx, cy
                
                for dcx in [-0.5, -0.25, 0, 0.25, 0.5]:
                    for dcy in [-0.5, -0.25, 0, 0.25, 0.5]:
                        tcx, tcy = cx + dcx, cy + dcy
                        ts = 0.0
                        for (x, y) in points:
                            for (nx, ny) in normals:
                                proj = abs((x - tcx) * nx + (y - tcy) * ny)
                                s_cand = proj / (math.sqrt(3) / 2)
                                if s_cand > ts:
                                    ts = s_cand
                        if ts < best_local_s:
                            best_local_s = ts
                            best_local_cx, best_local_cy = tcx, tcy
                
                if abs(best_local_s - s_needed) < 1e-6:
                    break
                cx, cy = best_local_cx, best_local_cy
                s_needed = best_local_s
            
            if s_needed < best_s:
                best_s = s_needed
                best_cx, best_cy, best_theta = cx, cy, theta
        
        return (best_cx, best_cy, best_s, best_theta)
    
    # Compute minimal enclosing hexagon
    cx, cy, side, theta = min_enclosing_hexagon(xy_positions)
    
    # Build inner list: (x, y, angle_degrees) for each unit hexagon
    # All inner hexagons have the same orientation (0 degrees for simplicity)
    inner = []
    for (x, y) in xy_positions:
        inner.append([x, y, 0.0])
    
    # Outer hexagon: center (cx, cy), side length, rotation
    # The angle parameter is the rotation of the outer hexagon
    # We used theta in radians, convert to degrees
    outer_angle = math.degrees(theta)
    
    return inner, (cx, cy), side, outer_angle
# EVOLVE-BLOCK-END

