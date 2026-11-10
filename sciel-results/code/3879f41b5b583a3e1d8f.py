import numpy as np
import time
from scipy.optimize import minimize


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger
    regular hexagon, minimizing the side length of the outer hexagon.
    """
    start_time = time.time()
    timeout = 575  # Safety margin for 630s limit

    n = 12
    sqrt3 = np.sqrt(3)

    # Edge normals for aligned (angle=0) unit hexagons
    # These are the 3 unique normals (opposite edges share a normal direction)
    n1 = np.array([sqrt3 / 2, 0.5])
    n2 = np.array([0, 1])
    n3 = np.array([-sqrt3 / 2, 0.5])
    axes = np.array([n1, n2, n3])

    # For two aligned unit hexagons with centers c1, c2:
    # They are separated iff max_i | (c1-c2) . n_i | > sqrt(3)
    # (The inradius of a unit hexagon is sqrt(3)/2, so 2*inradius = sqrt(3))
    min_allowed_proj = sqrt3 + 1e-7

    # Containment: for outer hexagon of side s, centered at origin, angle 0
    # A point p is inside iff max_i |p . n_i| <= s * sqrt(3) / 2
    # So s >= 2/sqrt(3) * max_i |p . n_i|
    # For inner hexagon with center c and angle 0, the max extent along n_i
    # is |c . n_i| + sqrt(3)/2
    # So s >= 2/sqrt(3) * (max_i |c . n_i| + sqrt(3)/2)
    #            = 2/sqrt(3) * max_i |c . n_i| + 1
    scale = 2.0 / sqrt3

    # Base honeycomb 3x4 grid positions (rigid cluster)
    e1 = np.array([1.5, sqrt3 / 2])
    e2 = np.array([0, sqrt3])
    base = np.array([i * e1 + j * e2 for i in range(3) for j in range(4)])
    base -= base.mean(axis=0)

    def compute_s(centers):
        """Compute minimal enclosing outer hexagon side length for aligned
        inner hexagons centered at `centers`."""
        c_n1 = centers @ n1
        c_n2 = centers @ n2
        c_n3 = centers @ n3
        s1 = scale * max(np.max(c_n1), -np.min(c_n1)) + 1
        s2 = scale * max(np.max(c_n2), -np.min(c_n2)) + 1
        s3 = scale * max(np.max(c_n3), -np.min(c_n3)) + 1
        return max(s1, s2, s3)

    def check_separation(centers):
        """Return True if all pairs of aligned unit hexagons are strictly
        separated (no intersection)."""
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                projs = np.abs(diff @ axes.T)
                if np.max(projs) <= min_allowed_proj:
                    return False
        return True

    def objective_24(x):
        """Full 24-DOF objective: centers with separation penalty."""
        centers = x.reshape((n, 2))
        # Separation penalty
        max_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                projs = np.abs(diff @ axes.T)
                mp = np.max(projs)
                if mp < min_allowed_proj:
                    v = min_allowed_proj - mp
                    if v > max_viol:
                        max_viol = v
        if max_viol > 0:
            return 1e10 + max_viol
        return compute_s(centers)

    def objective_3p(p):
        """3-parameter affine transform of the rigid honeycomb cluster.
        p = [angle_deg, shear, scale_factor]."""
        angle = np.radians(p[0])
        shear = p[1]
        sf = p[2]
        ca, sa = np.cos(angle), np.sin(angle)
        R = np.array([[ca, -sa], [sa, ca]])
        S = np.array([[1, shear], [0, 1]])
        M = R @ S * sf
        centers = (base @ M.T)
        return compute_s(centers)

    def objective_2p(p):
        """2-parameter rotation + scale of the rigid honeycomb cluster.
        p = [angle_deg, scale_factor]."""
        angle = np.radians(p[0])
        sf = p[1]
        ca, sa = np.cos(angle), np.sin(angle)
        R = np.array([[ca, -sa], [sa, ca]])
        centers = (base @ R.T) * sf
        return compute_s(centers)

    best_x = None
    best_f = np.inf

    # Phase 0: Coarse grid search over angle
    print("Phase 0: Coarse grid search")
    best_grid_f = np.inf
    best_grid_p = None
    for angle_deg in np.arange(-30, 31, 1.0):
        if time.time() - start_time > timeout - 200:
            break
        angle = np.radians(angle_deg)
        ca, sa = np.cos(angle), np.sin(angle)
        R = np.array([[ca, -sa], [sa, ca]])
        centers = base @ R.T
        if check_separation(centers):
            f = compute_s(centers)
            if f < best_grid_f:
                best_grid_f = f
                best_grid_p = np.array([angle_deg, 1.0])

    if best_grid_p is not None:
        print(f"  Grid best: angle={best_grid_p[0]:.1f}, s={best_grid_f:.6f}")

    # Phase 1: 3-parameter affine transform search (rotation + shear + scale)
    print("Phase 1: 3-parameter affine transform search")
    rng = np.random.default_rng(42)
    n_restarts = 120
    for restart in range(n_restarts):
        if time.time() - start_time > timeout - 150:
            break
        if restart == 0:
            p0 = np.array([0.0, 0.0, 1.0])
        elif restart == 1:
            p0 = np.array([30.0, 0.0, 1.0])
        elif restart == 2:
            p0 = np.array([-30.0, 0.0, 1.0])
        elif restart == 3:
            p0 = np.array([15.0, 0.2, 1.0])
        elif restart == 4:
            p0 = np.array([-15.0, -0.2, 1.0])
        elif restart == 5 and best_grid_p is not None:
            p0 = np.array([best_grid_p[0], 0.0, 1.0])
        elif restart == 6 and best_grid_p is not None:
            p0 = np.array([best_grid_p[0], 0.1, 1.0])
        elif restart == 7 and best_grid_p is not None:
            p0 = np.array([best_grid_p[0], -0.1, 1.0])
        elif restart == 8:
            p0 = np.array([5.0, 0.05, 1.0])
        elif restart == 9:
            p0 = np.array([-5.0, -0.05, 1.0])
        elif restart == 10:
            p0 = np.array([10.0, 0.1, 1.0])
        elif restart == 11:
            p0 = np.array([-10.0, -0.1, 1.0])
        else:
            p0 = np.array([
                rng.uniform(-35, 35),
                rng.uniform(-0.5, 0.5),
                rng.uniform(0.95, 1.05),
            ])
        res = minimize(objective_3p, p0, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
        p_best = res.x
        sf = p_best[2]
        angle = np.radians(p_best[0])
        shear = p_best[1]
        ca, sa = np.cos(angle), np.sin(angle)
        R = np.array([[ca, -sa], [sa, ca]])
        S = np.array([[1, shear], [0, 1]])
        M = R @ S * sf
        centers = base @ M.T
        if check_separation(centers):
            f = compute_s(centers)
            if f < best_f:
                best_f = f
                best_x = centers.flatten()
                if restart % 10 == 0:
                    print(f"  restart {restart}: s={f:.6f}")

    # Phase 1b: 2-parameter rotation + scale search
    print("Phase 1b: 2-parameter rotation+scale search")
    for restart in range(60):
        if time.time() - start_time > timeout - 150:
            break
        if restart == 0:
            p0 = np.array([0.0, 1.0])
        elif restart == 1:
            p0 = np.array([30.0, 1.0])
        elif restart == 2:
            p0 = np.array([-30.0, 1.0])
        elif restart == 3 and best_grid_p is not None:
            p0 = np.array([best_grid_p[0], 1.0])
        else:
            p0 = np.array([
                rng.uniform(-35, 35),
                rng.uniform(0.95, 1.05),
            ])
        res = minimize(objective_2p, p0, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
        p_best = res.x
        angle = np.radians(p_best[0])
        sf = p_best[1]
        ca, sa = np.cos(angle), np.sin(angle)
        R = np.array([[ca, -sa], [sa, ca]])
        centers = (base @ R.T) * sf
        if check_separation(centers):
            f = compute_s(centers)
            if f < best_f:
                best_f = f
                best_x = centers.flatten()
                if restart % 10 == 0:
                    print(f"  2p restart {restart}: s={f:.6f}")

    if best_x is None:
        # Fallback: use the rigid 3x4 grid with a tiny expansion
        centers = base * 1.001
        best_x = centers.flatten()
        best_f = compute_s(centers)
        print(f"  fallback rigid grid: s={best_f:.6f}")

    # Phase 2: Full 24-DOF Nelder-Mead polish from the best result
    print("Phase 2: 24-DOF Nelder-Mead polish")
    x0 = best_x.copy()
    for polish_round in range(8):
        if time.time() - start_time > timeout - 60:
            break
        res = minimize(objective_24, x0, method='Nelder-Mead',
                       options={'maxiter': 15000, 'xatol': 1e-9, 'fatol': 1e-12})
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
            print(f"  polish {polish_round}: s={best_f:.6f}")
        # Perturb for next restart
        if time.time() - start_time > timeout - 30:
            break
        x0 = best_x + np.random.normal(0, 0.03, size=24)

    # Phase 3: Final high-precision polish pass
    print("Phase 3: high-precision polish")
    if time.time() - start_time < timeout - 30:
        res = minimize(objective_24, best_x, method='Nelder-Mead',
                       options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-13})
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
            print(f"  final polish: s={best_f:.6f}")

    # Extract best result
    centers = best_x.reshape((n, 2))

    # Final safety check: if separation is violated, scale up slightly
    if not check_separation(centers):
        centroid = centers.mean(axis=0)
        centers = (centers - centroid) * 1.002 + centroid
        best_f = compute_s(centers)

    # Compute final s
    s = compute_s(centers)

    # Add a small safety margin to ensure strict containment (floating point)
    s = s + 0.003

    # Construct return arrays
    inner_hex_data = np.zeros((n, 3))
    for i in range(n):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0  # Angle 0

    outer_hex_data = np.array([0.0, 0.0, 0.0])

    return inner_hex_data, outer_hex_data, s
