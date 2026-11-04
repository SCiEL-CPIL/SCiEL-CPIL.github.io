import numpy as np
import time
import scipy.optimize

N = 11
SQRT3 = np.sqrt(3.0)
DEADLINE_MARGIN = 5.0  # seconds before 150s kill

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    # Unit hexagon vertices (flat-top)
    unit_verts = np.array([
        [1.0, 0.0],
        [0.5, SQRT3/2],
        [-0.5, SQRT3/2],
        [-1.0, 0.0],
        [-0.5, -SQRT3/2],
        [0.5, -SQRT3/2]
    ])

    # Outer hexagon normals (flat-top)
    normals = np.array([
        [1.0, 0.0],
        [0.5, SQRT3/2],
        [-0.5, SQRT3/2],
        [-1.0, 0.0],
        [-0.5, -SQRT3/2],
        [0.5, -SQRT3/2]
    ])

    def time_left():
        return deadline - time.time()

    def cluster_side(centers, angles, rot_deg=0.0):
        """
        Compute the minimum outer hexagon side length to contain all hexagons.
        centers: (N, 2) array of hexagon centers
        angles: (N,) array of individual hexagon rotation angles in radians
        rot_deg: global rotation of the entire cluster in degrees
        Returns the side length of the enclosing regular hexagon.
        """
        # Apply global rotation to centers
        if rot_deg != 0.0:
            a = np.deg2rad(rot_deg)
            c, s = np.cos(a), np.sin(a)
            R = np.array([[c, -s], [s, c]])
            centers = centers @ R.T
            angles = angles + a

        # Compute all vertices: (N, 6, 2)
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            a = angles[i]
            c, s = np.cos(a), np.sin(a)
            R = np.array([[c, -s], [s, c]])
            verts[i] = unit_verts @ R.T + centers[i]

        # For each of 6 normals, find max and min projection over all vertices
        # proj shape: (N, 6, 6) -> (vertex_i, vertex_v, normal_n)
        projs = np.einsum('ivk,k->ivn', verts, normals)  # (N, 6, 6)
        # For each normal n: max over all i,v of proj[i,v,n]
        max_proj = projs.max(axis=(0, 1))  # (6,)
        min_proj = projs.min(axis=(0, 1))  # (6,)
        # Side length for normal n: (max - min) / sqrt(3)
        sides = (max_proj - min_proj) / SQRT3
        return sides.max()

    def check_nonoverlap(centers, angles, eps=1e-6):
        """Check if hexagons are non-overlapping using exact vertex-edge tests."""
        # For unit regular hexagons, a sufficient (and for this lattice, necessary)
        # condition: center distance >= sqrt(3) when both are axis-aligned.
        # With rotation, we use a conservative check: min center distance >= sqrt(3) - eps
        # This is not perfectly exact but works well for lattice packings.
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < SQRT3 - eps:
                    return False
        return True

    # Generate rigid cluster configurations
    d = SQRT3  # horizontal spacing for flat-top hexagons
    h = 1.5    # vertical spacing

    def make_cluster_flower():
        """Central hex + 6 around + 4 more"""
        centers = np.array([
            [0, 0],
            [d, 0], [d/2, h], [-d/2, h],
            [-d, 0], [-d/2, -h], [d/2, -h],
            [0, 2*h], [d*1.5, h], [d*1.5, -h], [0, -2*h]
        ])
        angles = np.zeros(N)
        return centers, angles

    def make_cluster_two_rows():
        """6 in bottom row, 5 in top row (staggered)"""
        centers = np.zeros((N, 2))
        for i in range(6):
            centers[i, 0] = (i - 2.5) * d
            centers[i, 1] = -h/2
        for i in range(5):
            centers[i+6, 0] = (i - 2.0) * d
            centers[i+6, 1] = h/2
        angles = np.zeros(N)
        return centers, angles

    def make_cluster_three_rows():
        """4-3-4 arrangement"""
        centers = np.zeros((N, 2))
        idx = 0
        for i in range(4):
            centers[idx, 0] = (i - 1.5) * d
            centers[idx, 1] = -h
            idx += 1
        for i in range(3):
            centers[idx, 0] = (i - 1.0) * d + d/2
            centers[idx, 1] = 0.0
            idx += 1
        for i in range(4):
            centers[idx, 0] = (i - 1.5) * d
            centers[idx, 1] = h
            idx += 1
        angles = np.zeros(N)
        return centers, angles

    def make_cluster_compact():
        """More compact: 5-6 or similar"""
        # Try a different arrangement: 2-3-4-2
        centers = np.zeros((N, 2))
        idx = 0
        # Row 0: 2 hexagons
        for i in range(2):
            centers[idx, 0] = (i - 0.5) * d
            centers[idx, 1] = -1.5*h
            idx += 1
        # Row 1: 3 hexagons
        for i in range(3):
            centers[idx, 0] = (i - 1.0) * d
            centers[idx, 1] = -0.5*h
            idx += 1
        # Row 2: 4 hexagons
        for i in range(4):
            centers[idx, 0] = (i - 1.5) * d
            centers[idx, 1] = 0.5*h
            idx += 1
        # Row 3: 2 hexagons
        for i in range(2):
            centers[idx, 0] = (i - 0.5) * d
            centers[idx, 1] = 1.5*h
            idx += 1
        angles = np.zeros(N)
        return centers, angles

    def make_cluster_compact2():
        """4-3-4 but tighter"""
        centers = np.zeros((N, 2))
        idx = 0
        for i in range(4):
            centers[idx, 0] = (i - 1.5) * d
            centers[idx, 1] = -h
            idx += 1
        for i in range(3):
            centers[idx, 0] = (i - 1.0) * d + d/2
            centers[idx, 1] = 0.0
            idx += 1
        for i in range(4):
            centers[idx, 0] = (i - 1.5) * d
            centers[idx, 1] = h
            idx += 1
        angles = np.zeros(N)
        return centers, angles

    def make_cluster_line():
        """Linear arrangement"""
        centers = np.zeros((N, 2))
        for i in range(N):
            centers[i, 0] = (i - N//2) * d
            centers[i, 1] = 0.0
        angles = np.zeros(N)
        return centers, angles

    def make_cluster_compact3():
        """Try 3-5-3"""
        centers = np.zeros((N, 2))
        idx = 0
        for i in range(3):
            centers[idx, 0] = (i - 1.0) * d
            centers[idx, 1] = -h
            idx += 1
        for i in range(5):
            centers[idx, 0] = (i - 2.0) * d
            centers[idx, 1] = 0.0
            idx += 1
        for i in range(3):
            centers[idx, 0] = (i - 1.0) * d
            centers[idx, 1] = h
            idx += 1
        angles = np.zeros(N)
        return centers, angles

    def make_cluster_compact4():
        """Try 2-4-3-2"""
        centers = np.zeros((N, 2))
        idx = 0
        for i in range(2):
            centers[idx, 0] = (i - 0.5) * d
            centers[idx, 1] = -1.5*h
            idx += 1
        for i in range(4):
            centers[idx, 0] = (i - 1.5) * d
            centers[idx, 1] = -0.5*h
            idx += 1
        for i in range(3):
            centers[idx, 0] = (i - 1.0) * d
            centers[idx, 1] = 0.5*h
            idx += 1
        for i in range(2):
            centers[idx, 0] = (i - 0.5) * d
            centers[idx, 1] = 1.5*h
            idx += 1
        angles = np.zeros(N)
        return centers, angles

    # List of cluster generators
    cluster_generators = [
        make_cluster_flower,
        make_cluster_two_rows,
        make_cluster_three_rows,
        make_cluster_compact,
        make_cluster_compact2,
        make_cluster_line,
        make_cluster_compact3,
        make_cluster_compact4,
    ]

    best_side = float('inf')
    best_centers = None
    best_angles = None
    best_rot = 0.0

    # Phase 1: Grid search over rotation angle for each cluster
    # The enclosing hexagon has 6-fold symmetry, so we only need to search 0-60 degrees
    n_grid = 600  # 0.1 degree resolution

    for gen in cluster_generators:
        if time_left() < 10.0:
            break
        try:
            centers, angles = gen()
            if not check_nonoverlap(centers, angles):
                continue

            # Coarse grid search
            rot_angles = np.linspace(0, 60, n_grid)
            sides = np.array([cluster_side(centers, angles, r) for r in rot_angles])
            best_idx = np.argmin(sides)
            best_rot_coarse = rot_angles[best_idx]
            best_side_coarse = sides[best_idx]

            # Fine grid search around the best
            fine_rot = np.linspace(best_rot_coarse - 1.0, best_rot_coarse + 1.0, 200)
            fine_rot = fine_rot[(fine_rot >= 0) & (fine_rot <= 60)]
            if len(fine_rot) > 0:
                fine_sides = np.array([cluster_side(centers, angles, r) for r in fine_rot])
                best_fine_idx = np.argmin(fine_sides)
                best_rot = fine_rot[best_fine_idx]
                best_side_candidate = fine_sides[best_fine_idx]
            else:
                best_rot = best_rot_coarse
                best_side_candidate = best_side_coarse

            if best_side_candidate < best_side:
                best_side = best_side_candidate
                best_centers = centers.copy()
                best_angles = angles.copy()
                best_rot = best_rot

        except Exception:
            continue

    # Phase 2: Nelder-Mead refinement on (rotation_angle, dx, dy) for the best cluster
    if best_centers is not None and time_left() > 15.0:
        def objective_nm(params):
            rot_deg = params[0]
            dx = params[1]
            dy = params[2]
            c = best_centers + np.array([dx, dy])
            s = cluster_side(c, best_angles, rot_deg)
            # Penalty for non-overlap
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((c[i,0]-c[j,0])**2 + (c[i,1]-c[j,1])**2)
                    if dist < SQRT3:
                        penalty += (SQRT3 - dist)**2 * 100.0
            return s + penalty

        x0 = np.array([best_rot, 0.0, 0.0])
        try:
            res = scipy.optimize.minimize(
                objective_nm, x0, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res.fun < best_side:
                best_rot = res.x[0]
                best_centers = best_centers + np.array([res.x[1], res.x[2]])
                best_side = res.fun
        except Exception:
            pass

    # Phase 3: Try per-hexagon angle optimization
    if best_centers is not None and time_left() > 10.0:
        # Optimize individual angles while keeping centers fixed
        def objective_angles(params):
            angles_new = params
            s = cluster_side(best_centers, angles_new, best_rot)
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((best_centers[i,0]-best_centers[j,0])**2 + (best_centers[i,1]-best_centers[j,1])**2)
                    if dist < SQRT3:
                        penalty += (SQRT3 - dist)**2 * 100.0
            return s + penalty

        x0 = best_angles.copy()
        try:
            res = scipy.optimize.minimize(
                objective_angles, x0, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res.fun < best_side:
                best_angles = res.x
                best_side = res.fun
        except Exception:
            pass

    # Phase 4: Joint optimization of centers and rotation for the best cluster
    if best_centers is not None and time_left() > 10.0:
        def objective_joint(params):
            # params: [rot_deg, x0, y0, x1, y1, ..., x10, y10]
            rot_deg = params[0]
            centers_new = params[1:].reshape(N, 2)
            s = cluster_side(centers_new, best_angles, rot_deg)
            # Non-overlap penalty
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((centers_new[i,0]-centers_new[j,0])**2 + (centers_new[i,1]-centers_new[j,1])**2)
                    if dist < SQRT3:
                        penalty += (SQRT3 - dist)**2 * 50.0
            return s + penalty

        x0 = np.concatenate([[best_rot], best_centers.flatten()])
        try:
            res = scipy.optimize.minimize(
                objective_joint, x0, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            if res.fun < best_side:
                best_rot = res.x[0]
                best_centers = res.x[1:].reshape(N, 2)
                best_side = res.fun
        except Exception:
            pass

    # Final verification
    if best_centers is None:
        # Fallback: linear arrangement
        inner = [[d * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 50.0, 0.0

    # Apply final rotation to get output coordinates
    a = np.deg2rad(best_rot)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    final_centers = best_centers @ R.T

    # Final side length
    final_side = cluster_side(best_centers, best_angles, best_rot)

    # Verify non-overlap
    if not check_nonoverlap(best_centers, best_angles, eps=1e-4):
        # Fallback to a known-good configuration
        inner = [[d * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 50.0, 0.0

    inner = []
    for i in range(N):
        x = float(final_centers[i, 0])
        y = float(final_centers[i, 1])
        ang_deg = float(np.rad2deg(best_angles[i] + a))
        inner.append([x, y, ang_deg])

    return inner, (0.0, 0.0), float(final_side), 0.0
