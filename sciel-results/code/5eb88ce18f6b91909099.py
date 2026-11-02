import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Arrange 12 unit hexagons in concentric rings (1 center, 5 inner, 6 outer)
    and optimize ring radii + global rotation to minimize the enclosing hexagon side length.
    """
    # Unit hexagon vertices (pointy-top, side=1)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees from center
    def hex_vertices(cx, cy, angle_deg, side=1.0):
        a = np.radians(angle_deg)
        angles = a + np.radians(60 * np.arange(6))
        return np.column_stack([cx + side * np.cos(angles),
                                cy + side * np.sin(angles)])

    def all_hex_points(cx, cy, angle_deg, side=1.0):
        """Return all 6 vertices of a hexagon centered at (cx,cy)."""
        return hex_vertices(cx, cy, angle_deg, side)

    def min_enclosing_hex_side(points, rot_deg):
        """
        Find the minimum side length s of a regular hexagon (rotated by rot_deg)
        that contains all given points. The hexagon is centered optimally.
        For a regular hexagon with side s and rotation theta, the 6 outward normals
        are at angles theta + 30 + 60*k. The support function gives the max projection.
        For a centered hexagon with 'inradius' r = s * sqrt(3)/2, the constraint is
        that max projection on each normal <= r. The minimal s is determined by the
        max over all 6 normals of the required half-width.
        
        Actually for a regular hexagon, the distance from center to each side (apothem)
        = s * sqrt(3)/2. The hexagon contains point p iff for all 6 normals n_i,
        n_i . (p - c) <= apothem.
        To minimize s, we need to find center c and apothem r such that all points
        satisfy the constraints, minimizing r (thus s = 2r/sqrt(3)).
        
        This is a LP: minimize r subject to n_i.(p - c) <= r for all i, all p.
        Equivalent: for each normal i, max_p n_i.(p-c) <= r.
        max_p n_i.p - n_i.c <= r
        Let h_i = max_p n_i.p (support function of point set in direction n_i).
        Then h_i - n_i.c <= r  =>  n_i.c >= h_i - r
        Also -h_i' + n_i.c <= r where h_i' = max_p (-n_i).p = -min_p n_i.p
        So: min_p n_i.p + n_i.c <= r  =>  n_i.c <= r - min_p n_i.p
        
        So for each direction i: (h_i - r) <= n_i.c <= (r - l_i) where l_i = min_p n_i.p
        This requires h_i - r <= r - l_i  =>  h_i + l_i <= 2r  =>  r >= (h_i + l_i)/2
        
        So minimal r = max_i (h_i + l_i)/2, and c can be chosen as (h_i + l_i)/2 in each direction... 
        but c must satisfy all simultaneously. Actually the minimal r is max_i (h_i - l_i)/2? 
        Let me reconsider.
        
        Width in direction i: w_i = h_i - l_i (where h_i = max n_i.p, l_i = min n_i.p)
        For a centered hexagon with apothem r in direction i, we need w_i/2 <= r.
        So r >= max_i w_i / 2.
        And the center c must be at the midpoint in each direction: c_i such that
        n_i.c = (h_i + l_i)/2 for all i. This is consistent only if the point set
        is symmetric... but in general we just need r >= max_i w_i/2 and c exists.
        
        Actually, the minimal enclosing regular hexagon (fixed orientation) has:
        r_min = max_i (h_i - l_i) / 2
        and the center is at c where n_i . c = (h_i + l_i)/2 for all i (this may not be
        exactly solvable, but the minimal r is indeed max_i width_i / 2).
        
        So s_min = 2 * r_min / sqrt(3) = max_i (h_i - l_i) / sqrt(3)
        """
        rot = np.radians(rot_deg)
        # Normals of hexagon sides: at angles rot + 30 + 60*k
        normal_angles = rot + np.radians(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Project points onto each normal
        proj = points @ normals.T  # shape (n_points, 6)
        h = proj.max(axis=0)  # max projection in each direction
        l = proj.min(axis=0)  # min projection in each direction
        widths = h - l
        r_min = np.max(widths) / 2.0
        s_min = 2.0 * r_min / np.sqrt(3.0)
        return s_min

    def cluster_points(params):
        """
        params = [r_inner, r_outer, global_rot]
        Returns list of (cx, cy, angle) for 12 hexagons and their vertex points.
        """
        r_in, r_out, grot = params
        pts = []
        centers = []
        
        # Center hexagon
        centers.append((0.0, 0.0, grot))
        
        # 5 inner hexagons at radius r_in, angles 0, 72, 144, 216, 288
        for k in range(5):
            a = np.radians(72.0 * k + grot)
            cx = r_in * np.cos(a)
            cy = r_in * np.sin(a)
            centers.append((cx, cy, grot))
        
        # 6 outer hexagons at radius r_out, angles 30, 90, 150, 210, 270, 330 (offset by 30 from inner)
        for k in range(6):
            a = np.radians(60.0 * k + 30.0 + grot)
            cx = r_out * np.cos(a)
            cy = r_out * np.sin(a)
            centers.append((cx, cy, grot))
        
        # Collect all vertices
        for (cx, cy, ang) in centers:
            verts = hex_vertices(cx, cy, ang, 1.0)
            pts.append(verts)
        
        all_pts = np.vstack(pts)
        return centers, all_pts

    def objective(params):
        r_in, r_out, grot = params
        if r_in < 0.1 or r_out < 0.1:
            return 1e6
        # Check for overlaps: distance between centers must be >= 2 (for unit hexagons,
        # min distance to avoid overlap is 2*apothem*2? Actually for two unit hexagons,
        # they overlap if center distance < 2 * (sqrt(3)/2) * 2 = 2*sqrt(3) ~ 3.46? 
        # No: the minimum distance between two non-overlapping unit hexagons depends on
        # relative orientation. The "diameter" of a unit hexagon is 2 (distance between
        # opposite vertices). Two hexagons can be as close as... the inradius is sqrt(3)/2.
        # For same orientation, min center distance to avoid overlap = 2 * sqrt(3)/2 * 2? 
        # Actually the width of a unit hexagon in any direction is between sqrt(3) and 2.
        # To be safe, require center distance >= 2.0 (this is conservative but ensures no overlap
        # for any relative orientation since max width is 2).
        centers, all_pts = cluster_points(params)
        n = len(centers)
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 2.0:
                    return 1e6 + (2.0 - d) * 100
        return min_enclosing_hex_side(all_pts, 0.0)  # hexagon rotation is 0, we optimize grot

    # Initial guess
    r_in0 = 1.8
    r_out0 = 3.2
    grot0 = 0.0
    x0 = np.array([r_in0, r_out0, grot0])
    
    # Optimize
    best_result = None
    for grot_init in [0.0, 15.0, 30.0, 45.0]:
        x0_try = np.array([r_in0, r_out0, grot_init])
        try:
            res = minimize(objective, x0_try, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if best_result is None or res.fun < best_result.fun:
                best_result = res
        except:
            pass
    
    if best_result is None or not np.isfinite(best_result.fun):
        # Fallback
        r_in, r_out, grot = r_in0, r_out0, grot0
    else:
        r_in, r_out, grot = best_result.x
        r_in = max(0.5, r_in)
        r_out = max(0.5, r_out)
    
    centers, all_pts = cluster_points([r_in, r_out, grot])
    
    # Compute the enclosing hexagon side length
    side = min_enclosing_hex_side(all_pts, 0.0)
    
    # Build inner list: [x, y, angle_degrees]
    inner = [[c[0], c[1], c[2]] for c in centers]
    
    return inner, (0.0, 0.0), float(side), 0.0
# EVOLVE-BLOCK-END
