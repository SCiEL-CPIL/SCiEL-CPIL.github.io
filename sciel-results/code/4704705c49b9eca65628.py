"""Packing unit hexagons inside a regular hexagon – improved seed."""
import math
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill

    def hex_support(r, theta_deg, side=1.0):
        """Support function of a regular hexagon with side 'side', rotated theta_deg."""
        # For a regular hexagon centered at origin with circumradius R = side,
        # the support function in direction theta is R * max over 6 vertices.
        # Actually for a regular hexagon of side s, the circumradius is s.
        # The support in direction theta is s * max_{k} cos(theta - pi*k/3)
        # But more precisely: h(theta) = s * max_{k=0..5} cos(theta - k*pi/3)
        theta = math.radians(theta_deg)
        vals = [math.cos(theta - k * math.pi / 3.0) for k in range(6)]
        return side * max(vals)

    def hex_overlap_check(p1, a1, p2, a2, tol=1e-9):
        """
        Check if two unit hexagons (centered at p1, rotated a1; centered at p2, rotated a2) overlap.
        Use separating axis theorem: hexagon has 6 unique normal directions (each side normal).
        For each normal direction of both hexagons, project and check separation.
        """
        # Get the 6 normal directions for each hexagon
        # A regular hexagon with side 1 has normals at angles: a + k*pi/3 for k=0..5, but only 3 unique
        # Actually 6 normals but opposite pairs are equivalent, so 3 unique axes.
        # Let's use all 6 to be safe.
        
        def normals(angle_deg):
            a = math.radians(angle_deg)
            # Normals to sides of a regular hexagon with flat-top orientation at 0 deg:
            # The hexagon vertices are at angles 0, 60, 120, 180, 240, 300 (if circumradius=1)
            # Side normals are at 30, 90, 150, 210, 270, 330 degrees from center
            # So normals at a + (30 + 60*k) degrees for k=0..5
            return [a + math.radians(30 + 60 * k) for k in range(6)]
        
        n1 = normals(a1)
        n2 = normals(a2)
        all_normals = n1 + n2
        
        d = (p2[0] - p1[0], p2[1] - p1[1])
        dist2 = d[0]**2 + d[1]**2
        
        for n_ang in all_normals:
            nx = math.cos(n_ang)
            ny = math.sin(n_ang)
            # Project difference onto normal
            proj = d[0]*nx + d[1]*ny
            # Max extent along this normal for each hexagon
            h1 = hex_support(nx, ny, 1.0)
            # Wait, support function: h(n) = max of dot(v, n) for v in hexagon
            # For a regular hexagon of side s centered at origin, rotated by angle a:
            # h(n) = s * max_k cos(n_angle - a - k*pi/3) ... let me re-derive.
            # Actually, for a regular hexagon with circumradius R (=s for unit side),
            # the support in direction u (unit vector) is R * max_{k=0..5} u . v_k
            # where v_k are the 6 unit vectors at angles a + k*pi/3.
            # So h = R * max_k cos(n_angle - (a + k*pi/3))
            
            # Let me compute h1 properly
            a1r = math.radians(a1)
            a2r = math.radians(a2)
            h1 = max(math.cos(n_ang - a1r - k*math.pi/3.0) for k in range(6))
            h2 = max(math.cos(n_ang - a2r - k*math.pi/3.0) for k in range(6))
            
            # If the projection distance is greater than h1+h2, they're separated
            if abs(proj) > h1 + h2 + tol:
                continue  # Separated along this axis
            # If we reach here for all axes, they overlap
            # But we need to check ALL axes; if any axis separates, no overlap
            # So we continue only if separated. If we finish the loop without separation, they overlap.
        # If we got here, no axis separated them
        # But wait, the logic above is wrong: we should return False (no overlap) if ANY axis separates
        # Let me restructure
        
        return True  # placeholder, will fix

    def check_overlap(p1, a1, p2, a2, tol=1e-9):
        """Return True if hexagons overlap."""
        a1r = math.radians(a1)
        a2r = math.radians(a2)
        
        def normals(angle_rad):
            # Side normals of regular hexagon (circumradius 1)
            return [angle_rad + math.radians(30 + 60*k) for k in range(6)]
        
        n1 = normals(a1r)
        n2 = normals(a2r)
        all_normals = n1 + n2
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        for n_ang in all_normals:
            nx = math.cos(n_ang)
            ny = math.sin(n_ang)
            proj = dx*nx + dy*ny
            h1 = max(math.cos(n_ang - a1r - k*math.pi/3.0) for k in range(6))
            h2 = max(math.cos(n_ang - a2r - k*math.pi/3.0) for k in range(6))
            if abs(proj) >= h1 + h2 - tol:
                return False  # Separated
        return True  # Overlapping

    def fits_in_outer(hex_center, hex_angle, outer_center, outer_side, outer_angle):
        """Check if unit hexagon fits inside the outer hexagon."""
        oc = math.radians(outer_angle)
        hc = math.radians(hex_angle)
        
        # For each of the 6 sides of the outer hexagon, check that the inner hexagon
        # is on the correct side (inside).
        # The outer hexagon has side length outer_side.
        # The distance from center to each side (apothem) = outer_side * sqrt(3)/2
        # The normals to the outer hexagon sides are at angles outer_angle + (30+60*k) degrees
        # For a point to be inside, its projection along each outward normal must be <= apothem
        # For the inner hexagon, the max extent along each outward normal of the outer hexagon
        # plus the center's projection must be <= apothem.
        
        apothem = outer_side * math.sqrt(3) / 2.0
        
        dx = hex_center[0] - outer_center[0]
        dy = hex_center[1] - outer_center[1]
        
        for k in range(6):
            n_ang = oc + math.radians(30 + 60*k)
            nx = math.cos(n_ang)
            ny = math.sin(n_ang)
            center_proj = dx*nx + dy*ny
            # Max extent of inner hexagon along this direction
            inner_ext = max(math.cos(n_ang - hc - j*math.pi/3.0) for j in range(6))
            if center_proj + inner_ext > apothem + 1e-9:
                return False
        return True

    def is_valid_placement(centers, angles, outer_center, outer_side, outer_angle):
        """Check all inner hexagons fit and don't overlap."""
        n = len(centers)
        for i in range(n):
            if not fits_in_outer(centers[i], angles[i], outer_center, outer_side, outer_angle):
                return False
        for i in range(n):
            for j in range(i+1, n):
                if check_overlap(centers[i], angles[i], centers[j], angles[j]):
                    return False
        return True

    # Initial layout: try a grid-like arrangement
    # For N hexagons, use a hexagonal grid pattern
    # Start with a reasonable outer side
    
    best_result = None
    best_side = 40.0  # initial guess from seed
    
    # Try different arrangements
    # For small N, we can try various patterns
    
    # Pattern 1: linear arrangement along x-axis
    # Pattern 2: hexagonal cluster
    # Pattern 3: offset rows
    
    # Let's try a few configurations and find the minimum outer side that works
    
    # For N hexagons of side 1, the minimal outer hexagon side is at least related to N
    # A hexagonal number arrangement: 1, 7, 19, 37, ...
    # For N=11, a compact arrangement is possible
    
    # Try hexagonal cluster: center + 6 around + 4 more
    # Or just use a grid and binary search on outer side
    
    # Strategy: for each candidate arrangement, binary search the minimum outer_side
    
    def find_min_side(centers, angles, outer_center, outer_angle, lo=1.0, hi=50.0, iters=60):
        """Binary search for minimum outer side length."""
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            if is_valid_placement(centers, angles, outer_center, mid, outer_angle):
                hi = mid
            else:
                lo = mid
            if time.time() > deadline:
                break
        return hi

    # Generate candidate arrangements
    arrangements = []
    
    # Arrangement 1: All in a line
    for gap in [2.0, 2.1, 2.2]:
        centers = [(k*gap, 0.0) for k in range(N)]
        angles = [0.0]*N
        arrangements.append((centers, angles))
    
    # Arrangement 2: Two rows
    half1 = N // 2
    half2 = N - half1
    for gap in [2.0, 2.2]:
        y_offset = math.sqrt(3)
        centers = [(k*gap, 0.0) for k in range(half1)]
        centers += [((k+0.5)*gap, y_offset) for k in range(half2)]
        angles = [0.0]*N
        arrangements.append((centers, angles))
    
    # Arrangement 3: Hexagonal pattern
    # Center at (0,0), then ring of 6
    if N >= 7:
        centers = [(0.0, 0.0)]
        angles = [0.0]
        for k in range(6):
            ang = k * math.pi / 3
            r = 2.0
            centers.append((r*math.cos(ang), r*math.sin(ang)))
            angles.append(0.0)
        # Add remaining
        remaining = N - 7
        for k in range(remaining):
            ang = (k*61) * math.pi / 180  # spread out
            r = 4.0
            centers.append((r*math.cos(ang), r*math.sin(ang)))
            angles.append(0.0)
        arrangements.append((centers, angles))
    
    # Arrangement 4: Triangular grid
    # Pack in triangular lattice with spacing 2*sqrt(3)/sqrt(2) for rotated hexagons?
    # For axis-aligned hexagons, spacing needs to be at least 2 in certain directions
    
    # Let's also try with some rotations to allow denser packing
    for rot in [0.0, 30.0]:
        for gap in [2.0, 2.05, 2.1]:
            centers = [(k*gap, 0.0) for k in range(N)]
            angles = [rot]*N
            arrangements.append((centers, angles))
    
    # Evaluate each arrangement
    for centers, angles in arrangements:
        if time.time() > deadline:
            break
        side = find_min_side(centers, angles, (0.0, 0.0), 0.0)
        if side < best_side:
            best_side = side
            best_result = ([(c[0], c[1], a) for c, a in zip(centers, angles)], (0.0, 0.0), side, 0.0)
    
    # If no valid result found, return the seed-like result
    if best_result is None:
        inner = [(4.0*k, 0.0, 0.0) for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
