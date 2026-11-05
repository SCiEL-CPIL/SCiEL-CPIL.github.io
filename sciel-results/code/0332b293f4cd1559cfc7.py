import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Centers of 11 hexagons in a proper "flower" pattern
    # Minimum center-to-center distance for non-overlapping unit hexagons is 2*sqrt(3)
    eps = 1e-10
    centers = np.array([[0.0, 0.0]])

    # 6 hexagons at distance 2*sqrt(3) (touching the center hexagon)
    r1 = 2.0 * np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])

    # 4 hexagons at distance 4*sqrt(3) in the gaps between the ring
    r2 = 4.0 * np.sqrt(3) + eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])

    n_hex = len(centers)

    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])

    def compute_outer_side(theta_out, phis):
        """
        Computation of minimum outer hexagon side length.
        """
        # Outer hexagon face normals: at angles 30 + k*60 + theta_out
        angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)

        max_proj = 0.0
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]  # (6, 2)
            dots = verts @ normals.T  # (6, 6)
            max_proj = max(max_proj, np.max(dots))

        return 2.0 * max_proj / np.sqrt(3)

    def objective(params):
        """
        params: [theta_out, phi_0, ..., phi_10]
        Returns the outer hexagon side length.
        """
        theta_out = params[0]
        phis = params[1:]
        return compute_outer_side(theta_out, phis)

    # Initial guess: all hexagons aligned at 30 degrees
    x0 = np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex)

    best_score = objective(x0)
    best_params = x0.copy()

    # Multi-start Powell optimization
    n_starts = 8
    start_variants = [
        np.array([np.deg2rad(30)] + [np.deg2rad(30)] * n_hex),
        np.array([0.0] + [0.0] * n_hex),
        np.array([np.pi/6] + [0.0] * n_hex),
        np.array([0.0] + [np.pi/6] * n_hex),
        np.array([np.deg2rad(15)] + [np.deg2rad(15)] * n_hex),
        np.array([np.deg2rad(45)] + [np.deg2rad(45)] * n_hex),
        np.array([np.deg2rad(20)] + [np.deg2rad(20)] * n_hex),
        np.array([np.deg2rad(40)] + [np.deg2rad(40)] * n_hex),
    ]

    np.random.seed(42)
    for i in range(n_starts):
        if time.time() > deadline - 40:
            break

        if i < len(start_variants):
            x_start = start_variants[i].copy()
        else:
            x_start = np.random.uniform(0, np.pi/3, n_hex + 1)

        try:
            res = minimize(objective, x_start, method='Powell',
                          options={'maxiter': 300, 'disp': False, 'ftol': 1e-14, 'xtol': 1e-14})
            if res.fun < best_score - 1e-15:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Coordinate descent polish
    if time.time() < deadline - 15:
        for iteration in range(50):
            if time.time() > deadline - 10:
                break

            improved = False
            for j in range(n_hex + 1):
                if time.time() > deadline - 10:
                    break

                current_val = best_params[j]
                best_local = best_score
                best_local_val = current_val

                # Try multiple step sizes
                for delta in [1e-5, -1e-5, 1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2]:
                    test_val = current_val + delta
                    test_param = best_params.copy()
                    test_param[j] = test_val
                    try:
                        test_score = objective(test_param)
                        if test_score < best_local - 1e-16:
                            best_local = test_score
                            best_local_val = test_val
                            current_val = test_val
                    except Exception:
                        continue

                if best_local < best_score - 1e-15:
                    best_params[j] = best_local_val
                    best_score = best_local
                    improved = True

            if not improved:
                break

    # Extract solution
    theta_out = best_params[0]
    phis = best_params[1:]

    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])

    side = best_score + 1e-9  # Tiny margin for floating point safety

    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)

def _fallback_solution():
    """
    Analytical solution: all inner hexagons at 30 degrees, outer at 30 degrees.
    """
    eps = 1e-10
    centers = []
    centers.append((0.0, 0.0))

    r1 = 2.0 * np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers.append((r1 * np.cos(angle), r1 * np.sin(angle)))

    r2 = 4.0 * np.sqrt(3) + eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers.append((r2 * np.cos(angle), r2 * np.sin(angle)))

    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])

    # Compute side length analytically
    theta_out = np.deg2rad(30)
    phis = [np.deg2rad(30)] * 11

    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])

    normals = np.array([
        [np.cos(np.deg2rad(30 + k * 60) + theta_out),
         np.sin(np.deg2rad(30 + k * 60) + theta_out)]
        for k in range(6)
    ])

    max_proj = 0.0
    for i in range(11):
        phi = phis[i]
        c, s = np.cos(phi), np.sin(phi)
        rot = np.array([[c, -s], [s, c]])
        verts = base_vertices @ rot.T + np.array(centers[i])
        dots = verts @ normals.T
        max_proj = max(max_proj, np.max(dots))

    side = 2.0 * max_proj / np.sqrt(3) + 1e-9

    return inner, (0.0, 0.0), side, 30.0

if __name__ == "__main__":
    construct()
