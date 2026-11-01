"""Packing 11 unit hexagons in a regular hexagon via force-directed layout."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
# Hexagon vertices for side length 1 (flat-top orientation), centered at origin
# Flat-top: vertices at angles 0, 60, 120, 180, 240, 300 degrees
_HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, np.sqrt(3)/2],
    [-0.5, np.sqrt(3)/2],
    [-1.0, 0.0],
    [-0.5, -np.sqrt(3)/2],
    [0.5, -np.sqrt(3)/2],
])

def _rotate(pts, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit hexagon at center (cx,cy) rotated by angle_deg."""
    theta = np.radians(angle_deg)
    return _rotate(_HEX_VERTS, theta) + np.array([cx, cy])

def _min_enclosing_hexagon_sides(positions, angles):
    """
    For a set of unit hexagons at given positions/angles, find the minimum
    side length of a regular hexagon (any rotation) that contains all of them.
    
    We use the fact that a regular hexagon with side s and rotation phi contains
    a point p iff for all 6 directions u_k (k=0..5), u_k . (p - center) <= s * sqrt(3)/2 * 2... 
    Actually for a regular hexagon of side s, the apothem (inradius) is s*sqrt(3)/2.
    A regular hexagon with inradius r = s*sqrt(3)/2 and rotation phi contains point p
    iff for each of the 6 face-normal directions n_k (unit vectors at angles phi + k*60deg),
    n_k . (p - center) <= r.
    
    So for a cluster of points, the minimal enclosing regular hexagon with rotation phi
    has: center = some point, and r = max over all points p and all 6 normals n_k of n_k.(p-center).
    To minimize r over center and phi: for fixed phi, the optimal center is the midpoint
    of the max and min projections... actually it's a 2D problem.
    
    Simpler: for fixed phi, the minimal r is half the max over k of (max_p n_k.p - min_p n_k.p)
    and the center is at (max_p n_k.p + min_p n_k.p)/2 projected... 
    Actually: r_min(phi) = 0.5 * max_k [max_p (n_k . p) - min_p (n_k . p)]
    and side s = r / (sqrt(3)/2) = 2r/sqrt(3).
    Then minimize over phi.
    
    But we also need to verify the center placement. For a regular hexagon, if we place
    center at c, the constraint is n_k.(p - c) <= r for all k, p. The minimal r for fixed
    c, phi is max_{k,p} n_k.(p-c). To minimize over c, for each k we need c to be in the
    middle of the projection range. The optimal c satisfies: for each k, n_k.c = (max_p n_k.p + min_p n_k.p)/2.
    These 6 equations may not be simultaneously satisfiable, so we need to solve:
    minimize r s.t. n_k.(p - c) <= r for all k, p.
    This is a linear program in (c_x, c_y, r). But for simplicity and speed, I'll use:
    r(phi) = 0.5 * max_k [max_p(n_k.p) - min_p(n_k.p)]
    This is a valid upper bound on the minimal r for rotation phi (achieved by choosing c appropriately
    if the hexagon is centrally symmetric, which it is). Actually for a centrally symmetric convex body
    like a regular hexagon, the minimal enclosing copy with fixed orientation has r = 0.5 * max_k (range of projection).
    Yes! Because the hexagon is centrally symmetric, the minimal enclosing hexagon with fixed orientation
    centered at c has r = max_k (max_p n_k.(p-c)) and we can choose c to equalize. The minimal such r is
    exactly 0.5 * max_k (max_p n_k.p - min_p n_k.p).
    """
    # Compute all vertices of all hexagons
    all_pts = []
    for i in range(len(positions)):
        cx, cy = positions[i]
        ang = angles[i]
        verts = _hex_vertices(cx, cy, ang)
        all_pts.append(verts)
    all_pts = np.vstack(all_pts)
    
    best_s = np.inf
    # Try many rotations
    for phi_deg in np.linspace(0, 60, 121):
        phi = np.radians(phi_deg)
        # 6 face normal directions
        normals = np.array([[np.cos(phi + k*np.pi/3), np.sin(phi + k*np.pi/3)] for k in range(6)])
        # Project all points onto each normal
        max_range = 0.0
        for n in normals:
            proj = all_pts @ n
            max_range = max(max_range, proj.max() - proj.min())
        r = 0.5 * max_range
        s = 2.0 * r / np.sqrt(3.0)
        if s < best_s:
            best_s = s
    return best_s

def _check_validity(positions, angles, side, center):
    """Verify no overlaps and all inside outer hexagon."""
    # Check overlaps: for each pair, check if hexagons intersect
    # Two convex polygons intersect iff one vertex is inside the other or edges cross.
    # For hexagons of same size, they overlap iff distance between centers < some threshold
    # depending on relative rotation. Simplest: check if any vertex of one is inside the other.
    for i in range(len(positions)):
        vi = _hex_vertices(positions[i][0], positions[i][1], angles[i])
        for j in range(i+1, len(positions)):
            vj = _hex_vertices(positions[j][0], positions[j][1], angles[j])
            # Check if any vertex of i is inside j, or vice versa
            for v in vi:
                if _point_in_hexagon(v, positions[j][0], positions[j][1], angles[j]):
                    return False
            for v in vj:
                if _point_in_hexagon(v, positions[i][0], positions[i][1], angles[i]):
                    return False
    # Check all inside outer hexagon
    for i in range(len(positions)):
        vi = _hex_vertices(positions[i][0], positions[i][1], angles[i])
        for v in vi:
            if not _point_in_hexagon(v, center[0], center[1], 0.0):
                # Need to check with the outer hexagon's side length
                pass
    return True

def _point_in_hexagon(p, cx, cy, angle_deg):
    """Check if point p is inside or on boundary of unit hexagon at (cx,cy) with rotation angle_deg."""
    # Transform to hexagon local frame
    theta = np.radians(angle_deg)
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    local = R @ (p - np.array([cx, cy]))
    # Unit hexagon: |x| <= 1, |y| <= sqrt(3)/2, and |x| + |y|/sqrt(3) <= 1... 
    # Actually for flat-top unit hexagon, the constraint is:
    # |x| <= 1, |y| <= sqrt(3)/2, |x| + |y| <= 1.5? No.
    # The hexagon is the intersection of 6 half-planes.
    # For flat-top: normals at 0, 60, 120, 180, 240, 300 degrees, inradius = sqrt(3)/2.
    # n_k . local <= sqrt(3)/2 for all k.
    normals = np.array([[np.cos(k*np.pi/3), np.sin(k*np.pi/3)] for k in range(6)])
    return np.all(normals @ local <= np.sqrt(3)/2 + 1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Initial configuration: hexagonal cluster
    # 11 hexagons in a hexagonal arrangement: center + 6 around + 4 more
    # Actually, a compact arrangement: 1 center, 6 in ring, 4 in second ring partial
    # Let me use a known compact packing: 1+6+4 = 11
    # Hexagonal lattice spacing for touching hexagons: distance = 2 * apothem = sqrt(3)
    # For flat-top hexagons, adjacent centers are at distance sqrt(3) in the 6 lattice directions
    
    sqrt3 = np.sqrt(3.0)
    
    # Base positions for a hexagonal cluster
    # Center at (0,0), 6 neighbors at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300
    # Then 4 more in second ring
    base_positions = []
    base_angles = []
    # Center
    base_positions.append([0.0, 0.0])
    base_angles.append(0.0)
    # First ring (6 hexagons)
    for k in range(6):
        ang = k * np.pi / 3
        base_positions.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang)])
        base_angles.append(0.0)
    # Second ring: add 4 more at distance 2*sqrt(3) in directions 0, 120, 240, 300
    # Actually let me place them at the gaps
    for k in [0, 2, 4, 5]:  # 4 positions
        ang = k * np.pi / 3
        base_positions.append([2*sqrt3 * np.cos(ang), 2*sqrt3 * np.sin(ang)])
        base_angles.append(0.0)
    
    best_score = np.inf
    best_positions = None
    best_angles = None
    
    # Force-directed optimization
    def energy(params):
        """Energy function: repulsion for overlap + compactness."""
        pos = params[:2*N].reshape(N, 2)
        angs = params[2*N:]
        
        # Repulsion: for each pair, penalize if hexagons overlap
        # Use a smooth approximation: penalize based on center distance vs minimum
        # For two unit hexagons, minimum center distance for no overlap depends on relative rotation.
        # Worst case (same orientation): distance = sqrt(3)
        # Best case (rotated 30 deg): distance can be smaller
        # Use a conservative minimum distance of sqrt(3) * 0.95 to allow slight overlap penalty
        min_dist = sqrt3 * 0.98
        repulsion = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    # Smooth repulsion: quadratic penalty
                    overlap = min_dist - d
                    repulsion += 10.0 * overlap * overlap
                # Also add a small always-on repulsion to prevent exact coincidence
                repulsion += 0.01 / (d + 0.1)
        
        # Compactness: minimize the diameter of the cluster
        # Use the max pairwise distance as a proxy
        max_d = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                max_d = max(max_d, d)
        compactness = max_d ** 2
        
        # Regularization to keep angles near 0
        angle_reg = 0.01 * np.sum(angs ** 2)
        
        return repulsion + 0.1 * compactness + angle_reg
    
    # Multi-start optimization
    n_starts = 8
    for start in range(n_starts):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb initial configuration
        np.random.seed(42 + start)
        if start == 0:
            pos0 = np.array(base_positions)
            ang0 = np.array(base_angles)
        else:
            pos0 = np.array(base_positions) + np.random.randn(N, 2) * 0.5
            ang0 = np.array(base_angles) + np.random.randn(N) * 10.0 * np.pi / 180
        
        params0 = np.concatenate([pos0.flatten(), ang0])
        
        # Use Nelder-Mead
        result = minimize(energy, params0, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if time.time() > deadline - 5.0:
            break
        
        pos_opt = result.x[:2*N].reshape(N, 2)
        ang_opt = result.x[2*N:]
        
        # Compute the actual enclosing hexagon side length
        side = _min_enclosing_hexagon_sides(pos_opt, ang_opt)
        
        if side < best_score:
            best_score = side
            best_positions = pos_opt.copy()
            best_angles = ang_opt.copy()
    
    if best_positions is None:
        # Fallback to initial
        best_positions = np.array(base_positions)
        best_angles = np.array(base_angles)
        best_score = _min_enclosing_hexagon_sides(best_positions, best_angles)
    
    # Compute center of the enclosing hexagon
    all_pts = []
    for i in range(N):
        verts = _hex_vertices(best_positions[i][0], best_positions[i][1], best_angles[i])
        all_pts.append(verts)
    all_pts = np.vstack(all_pts)
    
    # Find optimal center and rotation
    best_center = (0.0, 0.0)
    best_side_final = best_score
    for phi_deg in np.linspace(0, 60, 61):
        phi = np.radians(phi_deg)
        normals = np.array([[np.cos(phi + k*np.pi/3), np.sin(phi + k*np.pi/3)] for k in range(6)])
        # For each normal, find max and min projection
        proj = all_pts @ normals.T
        # Center should be at midpoint of projections for each axis
        # Use average of midpoints
        midpoints = (proj.max(axis=0) + proj.min(axis=0)) / 2
        # Convert back to 2D: center in the projection space
        # c such that n_k . c = mid_k for all k. Least squares:
        # normals @ c = midpoints
        c, _, _, _ = np.linalg.lstsq(normals, midpoints, rcond=None)
        center = (float(c[0]), float(c[1]))
        # Compute side
        r = 0.0
        for k in range(6):
            n = normals[k]
            proj_k = all_pts @ n
            r = max(r, np.max(proj_k - n @ c))
        side = 2.0 * r / np.sqrt(3.0)
        if side < best_side_final:
            best_side_final = side
            best_center = center
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i][0]), float(best_positions[i][1]), float(np.degrees(best_angles[i]))])
    
    return inner, best_center, float(best_side_final), 0.0
