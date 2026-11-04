import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
D = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _get_outer_normals(angle_deg):
    """Unit normals for the 6 sides of a regular hexagon with given rotation."""
    angle_rad = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        theta = angle_rad + np.pi / 6 + i * np.pi / 3
        normals.append((np.cos(theta), np.sin(theta)))
    return np.array(normals)

def _min_enclosing_hex_side(composite_verts, outer_angle_deg, outer_center):
    """Exact minimum side length of a regular hexagon enclosing the point set.
    
    The hexagon is centered at outer_center with rotation outer_angle_deg.
    For a regular hexagon with side S, the distance from center to each side
    (apothem) is S * sqrt(3) / 2. So S = 2 * max_projection / sqrt(3).
    """
    normals = _get_outer_normals(outer_angle_deg)
    rel_verts = composite_verts - outer_center
    dots = rel_verts @ normals.T
    max_proj = np.max(np.abs(dots))
    return 2 * max_proj / np.sqrt(3)

def _all_inner_vertices(centers, angles_deg):
    """All vertices of all inner hexagons, as (N*6, 2)."""
    all_verts = []
    for i, (cx, cy) in enumerate(centers):
        all_verts.append(_hex_vertices(cx, cy, angles_deg[i]))
    return np.vstack(all_verts)

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill

    # ------------------------------------------------------------------
    # Phase 0: Direct construction of the known optimal "flower" packing.
    # 1 center + 6 at distance sqrt(3) + 4 at distance 3, all rotated 30 deg.
    # Theoretical minimum outer side length for this 11-hexagon cluster is 4.0.
    # We build it with a tiny epsilon gap for strict disjointness.
    # ------------------------------------------------------------------
    eps = 1e-9
    centers_list = []
    angles_list = []

    # Ring 0: center
    centers_list.append([0.0, 0.0])
    angles_list.append(30.0)

    # Ring 1: 6 hexagons at distance sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = D + eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        angles_list.append(30.0)

    # Ring 2: 4 hexagons at distance 3 + 2*eps, at 30, 90, 150, 210 deg
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        angles_list.append(30.0)

    centers = np.array(centers_list)
    angles = np.array(angles_list)
    n_inner = len(centers)

    # Compute the enclosing side of this exact configuration.
    # Outer hexagon centered at origin, rotated 30 deg.
    all_verts = _all_inner_vertices(centers, angles)
    side_flower = _min_enclosing_hex_side(all_verts, 30.0, np.array([0.0, 0.0]))

    best_S = side_flower
    best_outer = np.array([0.0, 0.0, 30.0])
    best_centers = centers.copy()

    # ------------------------------------------------------------------
    # Phase 1: Fast polish of the OUTER hexagon only (3 variables: cx, cy, angle)
    # using Nelder-Mead with the EXACT containment objective.
    # Each objective eval is one matrix multiply -- very cheap.
    # ------------------------------------------------------------------
    if time.time() < DEADLINE - 10.0:
        try:
            def _outer_objective(vars):
                outer_cx, outer_cy, outer_angle = vars
                outer_center = np.array([outer_cx, outer_cy])
                return _min_enclosing_hex_side(all_verts, outer_angle, outer_center)

            x0 = best_outer.copy()
            res = minimize(_outer_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_S:
                best_S = res.fun
                best_outer = res.x.copy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase 2: Joint polish of inner positions + outer hexagon.
    # Keep inner angles fixed at 30 deg (known optimum) and only
    # translate inner centers + adjust outer hexagon.
    # Use Nelder-Mead on 3 + 2*N = 25 variables.
    # ------------------------------------------------------------------
    if time.time() < DEADLINE - 15.0:
        try:
            def _joint_objective(vars):
                outer_cx = vars[0]
                outer_cy = vars[1]
                outer_angle = vars[2]
                c = vars[3:3 + 2 * n_inner].reshape(n_inner, 2)
                verts = []
                for i in range(n_inner):
                    verts.append(_hex_vertices(c[i][0], c[i][1], angles[i]))
                all_v = np.vstack(verts)
                return _min_enclosing_hex_side(all_v, outer_angle, np.array([outer_cx, outer_cy]))

            x0_joint = np.zeros(3 + 2 * n_inner)
            x0_joint[0] = best_outer[0]
            x0_joint[1] = best_outer[1]
            x0_joint[2] = best_outer[2]
            for i in range(n_inner):
                x0_joint[3 + 2 * i] = best_centers[i][0]
                x0_joint[4 + 2 * i] = best_centers[i][1]

            res = minimize(_joint_objective, x0_joint, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if time.time() < DEADLINE - 5.0 and res.fun < best_S:
                best_S = res.fun
                best_outer = res.x[:3].copy()
                best_centers = res.x[3:3 + 2 * n_inner].reshape(n_inner, 2).copy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase 3: Try alternative outer rotations with L-BFGS-B on inner positions.
    # The outer hexagon rotation is a discrete choice that can affect the
    # optimal inner placement. Try several rotations.
    # ------------------------------------------------------------------
    if time.time() < DEADLINE - 10.0:
        try:
            best_final_S = best_S
            best_final_params = best_centers.copy()
            best_final_outer = best_outer.copy()

            for alpha_test in np.linspace(0, 60, 7):  # 0, 10, 20, ..., 60 degrees
                if time.time() > DEADLINE - 5.0:
                    break

                def obj_alpha(c_flat, alpha=alpha_test):
                    c = c_flat.reshape(n_inner, 2)
                    verts = []
                    for i in range(n_inner):
                        verts.append(_hex_vertices(c[i][0], c[i][1], angles[i]))
                    all_v = np.vstack(verts)
                    outer_center = np.array([best_final_outer[0], best_final_outer[1]])
                    return _min_enclosing_hex_side(all_v, alpha, outer_center)

                x0 = best_final_params.flatten().copy()
                bounds = [(-6, 6), (-6, 6)] * n_inner

                res = minimize(obj_alpha, x0, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_final_S:
                    best_final_S = res.fun
                    best_final_params = res.x.reshape(n_inner, 2).copy()
                    best_final_outer = np.array([best_final_outer[0], best_final_outer[1], alpha_test])

            if best_final_S < best_S:
                best_S = best_final_S
                best_centers = best_final_params
                best_outer = best_final_outer
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------
    outer_cx, outer_cy, outer_angle = best_outer
    outer_center = (outer_cx, outer_cy)

    inner = []
    for i in range(n_inner):
        inner.append([best_centers[i][0], best_centers[i][1], angles[i]])

    side = best_S

    return inner, outer_center, side, outer_angle
