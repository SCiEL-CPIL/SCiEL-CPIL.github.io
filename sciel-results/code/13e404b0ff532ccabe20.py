import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY = 8.0
EPS = 1e-9

# Pre-computed hexagon vertices (flat-top)
_HEX_VERTS = np.array([(1.0, 0.0), (0.5, APOTHEM), (-0.5, APOTHEM),
                       (-1.0, 0.0), (-0.5, -APOTHEM), (0.5, -APOTHEM)])

def construct():
    T = lambda: time.time() - START_TIME
    if T() > TIME_LIMIT - SAFETY:
        # fallback: one hexagon
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def _verts(theta, cx=0.0, cy=0.0):
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return R @ _HEX_VERTS.T + np.array([cx, cy])

    def _pair_dist(v1, v2):
        # v1, v2: (2,6) arrays
        d = v1[:, None, :] - v2[None, :, :]
        return np.min(np.linalg.norm(d, axis=0))

    def _overlap(theta, cx, cy, ref):
        v = _verts(theta, cx, cy)
        for (t2, c2x, c2y) in ref:
            if _pair_dist(v, _verts(t2, c2x, c2y)) < -1e-7:
                return True
        return False

    def _fits(hexes, R):
        # R: (6,2) rotated axis normals of outer hexagon
        # Compute max projection distance in each of 6 directions
        pts = []
        for (theta, cx, cy) in hexes:
            pts.append(_verts(theta, cx, cy))
        pts = np.concatenate(pts, axis=1)  # (2, 6n)
        proj = np.abs(R @ pts)  # (6, 6n)
        D = proj.max(axis=1)    # (6,)
        return D.max() <= (R[0, 0] * 1.0 + APOTHEM)  # side = D+APOTHEM; here R[0,0]=1
        # Actually side = D_max + APOTHEM, so we check D_max <= side - APOTHEM

    def _side_from_projection(hexes, beta):
        # beta: rotation of outer hexagon
        # Outer hexagon axes are at angles beta + pi/6 + k*pi/3
        # Normals:
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        # Collect all vertices
        all_pts = []
        for (theta, cx, cy) in hexes:
            all_pts.append(_verts(theta, cx, cy))
        all_pts = np.concatenate(all_pts, axis=1)
        # Project onto each normal
        proj = np.abs(normals @ all_pts)
        D = proj.max()
        return D + APOTHEM

    # ---- Axial-to-Cartesian for a given hexagon set ----
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # ---- Candidate axial configurations ----
    axial_candidates = [
        # Compact 11-hex shapes
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
         (2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),
         (0,2),(1,2),(2,2),(-1,1),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),
         (0,2),(1,2),(2,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),
         (0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),
         (3,1),(4,1),(5,1)],
        # Star-like
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
         (2,1),(-2,2),(-1,3),(1,-2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
         (1,1),(-1,2),(-2,1),(2,-1)],
        # Row+branches
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),
         (0,2),(1,2),(2,2),(0,3),(1,3)],
        # Other compact
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),
         (0,2),(1,2),(2,2),(3,2)],
    ]

    # ---- Quick estimate: bounding hex for a configuration at beta=0 ----
    def quick_est(axial):
        pts = axial_to_cartesian(axial)
        cx, cy = pts[:,0].mean(), pts[:,1].mean()
        D = 0
        for k in range(6):
            ang = math.pi/6 + k*math.pi/3
            nx, ny = math.cos(ang), math.sin(ang)
            pr = (pts[:,0]-cx)*nx + (pts[:,1]-cy)*ny
            D = max(D, pr.max(), -pr.min())
        return D + APOTHEM

    sorted_axials = sorted(axial_candidates, key=quick_est)

    best_S = float('inf')
    best_result = None

    for axial in sorted_axials:
        if T() > TIME_LIMIT - SAFETY:
            break

        pts = axial_to_cartesian(axial)
        cx0, cy0 = float(pts[:,0].mean()), float(pts[:,1].mean())

        # Initial hexagons: all at center, theta=0
        init = [0.0, 0.0]  # cx, cy of the whole set (we move the set as a rigid body)
        # Actually for the rigid-body approach, we just optimize the set's position and rotation
        # But inner hexagons all share the same rotation theta_inner relative to the set

        # We'll optimize: cx, cy, theta (inner hex rotation), beta (outer hex rotation)
        # The set of points is fixed (from axial), just rotated by theta and translated

        def rigid_obj(params):
            cx, cy, theta, beta = params
            # Rotate axial points by theta, translate
            c, s = math.cos(theta), math.sin(theta)
            R = np.array([[c, -s], [s, c]])
            pts_r = R @ pts.T + np.array([cx, cy])

            # Build vertex list for all 11 hexagons
            verts_list = []
            for px, py in pts_r:
                c2, s2 = math.cos(theta), math.sin(theta)
                R2 = np.array([[c2, -s2], [s2, c2]])
                verts_list.append(R2 @ _HEX_VERTS.T + np.array([px, py]))
            all_v = np.concatenate(verts_list, axis=1)

            normals = np.array([
                (math.cos(beta + math.pi/6 + k*math.pi/3),
                 math.sin(beta + math.pi/6 + k*math.pi/3))
                for k in range(6)
            ])
            proj = np.abs(normals @ all_v)
            D = proj.max()
            side = D + APOTHEM

            # Penaltize overlaps
            # For each pair, check distance between hexagons
            overlap_pen = 0.0
            for i in range(len(pts_r)):
                for j in range(i+1, len(pts_r)):
                    d = math.hypot(pts_r[i,0]-pts_r[j,0], pts_r[i,1]-pts_r[j,1])
                    if d < 1.9:  # threshold for possible overlap
                        v1 = R2 @ _HEX_VERTS.T + np.array([pts_r[i,0], pts_r[i,1]])
                        v2 = R2 @ _HEX_VERTS.T + np.array([pts_r[j,0], pts_r[j,1]])
                        # Compute min distance between two convex hexagons
                        dd = np.min(np.linalg.norm(v1[:,None,:] - v2[None,:,:], axis=0))
                        if dd < 0:
                            overlap_pen += (-dd) ** 2
            return side + 1000.0 * overlap_pen

        # Grid search over theta and beta first
        best_rigid = None
        best_theta = 0.0
        best_beta = 0.0
        for theta_d in range(0, 60, 5):
            theta = math.radians(theta_d)
            for beta_d in range(0, 60, 5):
                beta = math.radians(beta_d)
                val = rigid_obj([cx0, cy0, theta, beta])
                if best_rigid is None or val < best_rigid:
                    best_rigid = val
                    best_theta = theta
                    best_beta = beta

        # Refine with minimize
        try:
            res = minimize(rigid_obj, [cx0, cy0, best_theta, best_beta],
                          method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                best_result = res.x
        except:
            pass

        # Second refinement with tighter grid
        if best_result is not None:
            cx, cy, theta, beta = best_result
            for dt in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                for db in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                    try:
                        res2 = minimize(rigid_obj, [cx, cy, theta+dt, beta+db],
                                       method='Nelder-Mead',
                                       options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-12})
                        if res2.fun < best_S:
                            best_S = res2.fun
                            best_result = res2.x
                    except:
                        pass

        if T() > TIME_LIMIT - SAFETY:
            break

    # If we found a good result
    if best_result is not None:
        cx, cy, theta, beta = best_result
        angle_deg = math.degrees(theta)
        outer_angle = math.degrees(beta)

        # Compute final side
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        pts_r = R @ pts.T + np.array([cx, cy])

        verts_list = []
        for px, py in pts_r:
            R2 = np.array([[c, -s], [s, c]])
            verts_list.append(R2 @ _HEX_VERTS.T + np.array([px, py]))
        all_v = np.concatenate(verts_list, axis=1)

        normals = np.array([
            (math.cos(beta + math.pi/6 + k*math.pi/3),
             math.sin(beta + math.pi/6 + k*math.pi/3))
            for k in range(6)
        ])
        proj = np.abs(normals @ all_v)
        side = float(proj.max() + APOTHEM)

        inner = []
        for px, py in pts_r:
            inner.append([float(px), float(py), float(angle_deg)])

        center = [float(cx), float(cy)]
        return inner, center, side, float(outer_angle)

    # Fallback
    return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

