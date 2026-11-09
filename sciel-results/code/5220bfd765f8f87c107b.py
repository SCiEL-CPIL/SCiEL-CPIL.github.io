import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path (n, 3) for the moving sofa problem.
    Uses a 2D arc center search with fast early-exit checking.
    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 15.0

    # Sofa grid
    x_g = np.arange(-3.0, 1.5, 0.01)
    y_g = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_g, y_g)
    Wx = X.flatten()
    Wy = Y.flatten()
    N_pts = len(Wx)

    def check_placement(x_c, y_c, a_deg, alive_mask):
        """Check which points in alive_mask are inside placement. Returns new alive_mask."""
        a_rad = np.deg2rad(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)

        dx = Wx[alive_mask] - x_c
        dy = Wy[alive_mask] - y_c

        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy

        in_p1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        in_p2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        inside = in_p1 | in_p2

        if not np.any(inside):
            return None  # Empty

        new_mask = alive_mask.copy()
        idx = np.where(alive_mask)[0]
        new_mask[idx[~inside]] = False
        return new_mask

    def compute_area_fast(path):
        """Compute area for a given path with early exit."""
        # Build all placements
        placements = [(0.0, 0.0, 0.0)]

        prev = (0.0, 0.0, 0.0)
        for i in range(len(path)):
            curr = tuple(path[i])
            da = abs(curr[2] - prev[2])
            m = max(1, int(np.ceil(da)))
            if m > 1:
                for k in range(1, m):
                    t = k / m
                    ix = prev[0] + t * (curr[0] - prev[0])
                    iy = prev[1] + t * (curr[1] - prev[1])
                    ia = prev[2] + t * (curr[2] - prev[2])
                    placements.append((ix, iy, ia))
            placements.append(curr)
            prev = curr

        placements.append((-2.0, 0.0, 90.0))

        # Check all placements with early exit
        alive = np.ones(N_pts, dtype=bool)
        for px, py, pa in placements:
            alive = check_placement(px, py, pa, alive)
            if alive is None:
                return 0.0

        return np.sum(alive) * 0.0001

    def make_path_from_params(params, n_segments=12):
        """
        Generate a path from parameters.
        params: [cx, cy, r, ...] where (cx,cy) is arc center, r is radius.
        The path follows an arc from angle_start to angle_end around (cx, cy).
        """
        cx, cy, r = params[0], params[1], params[2]

        # Start placement: (0, 0, 0) - inner corner at origin
        # End placement: (-2, 0, 90) - inner corner at (-2, 0), rotated 90 deg

        # We want the inner corner to trace an arc from (0,0) to (-2,0)
        # with rotation from 0 to 90 degrees.

        # The inner corner position at angle theta on the arc:
        # We need to find the arc that connects (0,0) to (-2,0)

        # Let's parameterize: the corner moves along a circular arc.
        # Start: corner at (0,0), rotation 0
        # End: corner at (-2,0), rotation 90

        # For a circular arc centered at (cx, cy) with radius r:
        # Start point: (0,0) must be on the arc: (0-cx)^2 + (0-cy)^2 = r^2
        # End point: (-2,0) must be on the arc: (-2-cx)^2 + (0-cy)^2 = r^2

        # From these: cx^2 + cy^2 = r^2 and (2+cx)^2 + cy^2 = r^2
        # So cx^2 = (2+cx)^2 => cx^2 = 4 + 4cx + cx^2 => 0 = 4 + 4cx => cx = -1
        # Then r^2 = 1 + cy^2

        # So cx is fixed at -1! The only free parameter is cy (or equivalently r).
        # Let me reconsider - maybe we should use a more general path.

        # Alternative: use a piecewise path with the arc as one segment
        # and straight segments before and after.

        # Actually, let's just use the constraint that cx=-1 and optimize cy.
        # But the cell finding says 2D (cx,cy) search is better. Let me think...

        # Maybe the path doesn't need to be a perfect arc. Let's use a more
        # flexible parameterization: N waypoints with angles fixed.

        # Let's use N waypoints with fixed angles, optimize (x,y) for each.
        n_pts_path = n_segments
        angles = np.linspace(0, 90, n_pts_path + 2)[1:-1]  # Exclude 0 and 90

        # Extract x, y for each waypoint
        xy = params[:2 * n_pts_path].reshape(n_pts_path, 2)

        path = np.zeros((n_pts_path, 3))
        path[:, 0] = xy[:, 0]
        path[:, 1] = xy[:, 1]
        path[:, 2] = angles

        return path, angles

    def make_arc_path(cx, cy, n_segments=12):
        """Generate a path following a circular arc from (0,0) to (-2,0)."""
        # cx should be -1 for the arc to pass through both points
        # But let's allow flexibility in how we parameterize

        # Compute angles from center to start and end points
        # Start: (0, 0), End: (-2, 0)
        # Vector from center to start: (0 - cx, 0 - cy)
        # Vector from center to end: (-2 - cx, 0 - cy)

        r_start = np.sqrt((0 - cx)**2 + (0 - cy)**2)
        r_end = np.sqrt((-2 - cx)**2 + (0 - cy)**2)

        # For a true arc, r_start should equal r_end. If not, we interpolate.
        # Let's use a path that goes from (0,0) to (-2,0) along the arc.

        angle_start = np.arctan2(0 - cy, 0 - cx)
        angle_end = np.arctan2(0 - cy, -2 - cx)

        # Ensure we take the shorter arc
        da = angle_end - angle_start
        if da > np.pi:
            da -= 2 * np.pi
        elif da < -np.pi:
            da += 2 * np.pi

        # Generate n_segments intermediate points
        path = []
        for i in range(1, n_segments):
            t = i / n_segments
            angle = angle_start + t * da
            r = r_start + t * (r_end - r_start)  # Interpolate radius
            px = cx + r * np.cos(angle)
            py = cy + r * np.sin(angle)

            # Rotation angle: interpolate from 0 to 90
            rot = 90.0 * t
            path.append([px, py, rot])

        return np.array(path)

    # Initial guess based on known good paths
    # Use a circular arc with center at (-1, cy)
    # Try different cy values

    best_path = None
    best_area = 0.0

    # First, try a grid of cy values with the arc path
    for cy in np.linspace(-2, 2, 20):
        if time.time() > deadline - safety_margin:
            break
        path = make_arc_path(-1.0, cy, n_segments=12)
        area = compute_area_fast(path)
        if area > best_area:
            best_area = area
            best_path = path.copy()

    # Now do local optimization around the best
    if best_path is not None:
        # Optimize the x, y of each waypoint with fixed angles
        n_segments = len(best_path)
        angles = best_path[:, 2].copy()

        def objective(params):
            if time.time() > deadline - safety_margin:
                return 0.0
            xy = params.reshape(n_segments, 2)
            path = np.zeros((n_segments, 3))
            path[:, 0] = xy[:, 0]
            path[:, 1] = xy[:, 1]
            path[:, 2] = angles
            return -compute_area_fast(path)

        init_params = best_path[:, :2].flatten()

        try:
            result = minimize(
                objective,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6,
                         'adaptive': True}
            )
            if -result.fun > best_area:
                xy = result.x.reshape(n_segments, 2)
                new_path = np.zeros((n_segments, 3))
                new_path[:, 0] = xy[:, 0]
                new_path[:, 1] = xy[:, 1]
                new_path[:, 2] = angles
                best_path = new_path
                best_area = -result.fun
        except Exception:
            pass

        # Try with more segments
        if time.time() < deadline - 60:
            for n_seg in [16, 20]:
                if time.time() > deadline - safety_margin:
                    break
                # Resample the best path to n_seg points
                angles_new = np.linspace(0, 90, n_seg + 2)[1:-1]
                # Interpolate x, y from best_path to new angles
                from scipy.interpolate import interp1d
                f_x = interp1d(best_path[:, 2], best_path[:, 0], kind='linear')
                f_y = interp1d(best_path[:, 2], best_path[:, 1], kind='linear')
                x_new = f_x(angles_new)
                y_new = f_y(angles_new)

                init_path = np.column_stack([x_new, y_new, angles_new])
                area = compute_area_fast(init_path)
                if area > best_area:
                    best_area = area
                    best_path = init_path.copy()

                # Optimize
                def objective2(params):
                    if time.time() > deadline - safety_margin:
                        return 0.0
                    xy = params.reshape(n_seg, 2)
                    path = np.zeros((n_seg, 3))
                    path[:, 0] = xy[:, 0]
                    path[:, 1] = xy[:, 1]
                    path[:, 2] = angles_new
                    return -compute_area_fast(path)

                init_params2 = init_path[:, :2].flatten()
                try:
                    result2 = minimize(
                        objective2,
                        init_params2,
                        method='Nelder-Mead',
                        options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-6,
                                 'adaptive': True}
                    )
                    if -result2.fun > best_area:
                        xy = result2.x.reshape(n_seg, 2)
                        new_path = np.zeros((n_seg, 3))
                        new_path[:, 0] = xy[:, 0]
                        new_path[:, 1] = xy[:, 1]
                        new_path[:, 2] = angles_new
                        best_path = new_path
                        best_area = -result2.fun
                except Exception:
                    pass

    # Final safety check
    if best_path is None:
        # Fallback: simple linear path
        best_path = np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ], dtype=np.float64)

    # Ensure all values are finite
    if not np.all(np.isfinite(best_path)):
        best_path = np.nan_to_num(best_path, nan=0.0, posinf=0.0, neginf=0.0)

    return best_path
