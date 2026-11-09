# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 620.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _base_inside(u, v):
    """Boolean: is (u,v) in the base L-shaped corridor?"""
    horiz = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
    vert = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    return horiz | vert


def _build_sofa_grid(subsample=1):
    """Return (X, Y) for the sofa grid. subsample>1 for faster evaluation."""
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    if subsample > 1:
        xs = xs[::subsample]
        ys = ys[::subsample]
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    return X, Y


def _densify_path(path):
    """Densify path per evaluator rules."""
    p = np.vstack([np.array([[0.0, 0.0, 0.0]]), path, np.array([[-2.0, 0.0, 90.0]])])
    dense = [p[0]]
    for i in range(len(p) - 1):
        a1 = p[i, 2]
        a2 = p[i + 1, 2]
        m = max(1, int(np.ceil(abs(a2 - a1) - 1e-9)))
        if m > 1:
            ts = np.linspace(0.0, 1.0, m + 1)
            for t in ts[1:-1]:
                row = p[i] * (1.0 - t) + p[i + 1] * t
                dense.append(row)
        dense.append(p[i + 1])
    return np.array(dense)


def _eval_path_vectorized(path, X, Y):
    """Vectorized evaluation: returns number of sofa grid points inside EVERY placement."""
    dense = _densify_path(path)
    n_placements = len(dense)

    chunk_size = max(1, min(n_placements, 150))
    inside = np.ones(X.shape, dtype=bool)

    for start in range(0, n_placements, chunk_size):
        end = min(start + chunk_size, n_placements)
        chunk = dense[start:end]
        n_chunk = end - start

        cx = chunk[:, 0]
        cy = chunk[:, 1]
        a_rad = np.deg2rad(chunk[:, 2])
        c = np.cos(a_rad)
        s = np.sin(a_rad)

        dx = X[None, :, :] - cx[:, None, None]
        dy = Y[None, :, :] - cy[:, None, None]

        u = c[:, None, None] * dx + s[:, None, None] * dy
        v = -s[:, None, None] * dx + c[:, None, None] * dy

        horiz = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        vert = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        mask = horiz | vert

        chunk_inside = mask.all(axis=0)
        inside &= chunk_inside

        if not inside.any():
            break

    return int(inside.sum())


def _score(path, X, Y):
    return _eval_path_vectorized(path, X, Y) * 0.0001


def _initial_paths():
    """Return multiple initial path candidates."""
    paths = []

    # Path 1: Arc-shaped, 5 points
    paths.append(np.array([
        [-0.4, 0.4, 20.0],
        [-0.9, 0.5, 45.0],
        [-1.4, 0.3, 65.0],
        [-1.8, 0.1, 80.0],
    ], dtype=np.float64))

    # Path 2: More gradual, 5 points
    paths.append(np.array([
        [-0.3, 0.3, 15.0],
        [-0.7, 0.4, 30.0],
        [-1.1, 0.4, 50.0],
        [-1.5, 0.2, 70.0],
        [-1.8, 0.0, 85.0],
    ], dtype=np.float64))

    # Path 3: Keep y higher longer, 4 points
    paths.append(np.array([
        [-0.5, 0.6, 25.0],
        [-1.0, 0.6, 50.0],
        [-1.5, 0.4, 70.0],
        [-1.9, 0.1, 85.0],
    ], dtype=np.float64))

    # Path 4: S-shaped, 5 points
    paths.append(np.array([
        [-0.4, 0.2, 18.0],
        [-0.8, 0.5, 35.0],
        [-1.2, 0.5, 55.0],
        [-1.6, 0.3, 72.0],
        [-1.9, 0.0, 88.0],
    ], dtype=np.float64))

    # Path 5: 6 points, more detailed
    paths.append(np.array([
        [-0.3, 0.3, 12.0],
        [-0.6, 0.4, 25.0],
        [-0.9, 0.5, 40.0],
        [-1.2, 0.5, 55.0],
        [-1.5, 0.3, 70.0],
        [-1.8, 0.1, 85.0],
    ], dtype=np.float64))

    # Path 6: 7 points, even more detailed
    paths.append(np.array([
        [-0.2, 0.2, 10.0],
        [-0.5, 0.4, 22.0],
        [-0.8, 0.5, 35.0],
        [-1.1, 0.5, 48.0],
        [-1.4, 0.4, 62.0],
        [-1.7, 0.2, 76.0],
        [-1.9, 0.0, 88.0],
    ], dtype=np.float64))

    # Path 7: Original seed-like
    paths.append(np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64))

    # Path 8: High arc, 6 points
    t = np.linspace(0.1, 0.9, 6)
    x = -2.0 * t
    y = 0.55 * 4.0 * t * (1.0 - t) + 0.1
    a = 90.0 * t
    paths.append(np.column_stack([x, y, a]))

    # Path 9: Medium arc, 7 points
    t = np.linspace(0.08, 0.92, 7)
    x = -2.0 * t
    y = 0.45 * 4.0 * t * (1.0 - t) + 0.05
    a = 90.0 * t
    paths.append(np.column_stack([x, y, a]))

    # Path 10: Circular arc parameterization
    center_x, center_y, radius = -1.0, 0.5, 1.2
    theta_start = np.pi * 0.1
    theta_end = np.pi * 0.9
    thetas = np.linspace(theta_start, theta_end, 6)
    x_arc = center_x + radius * np.cos(thetas)
    y_arc = center_y + radius * np.sin(thetas)
    a_arc = np.degrees(np.arctan2(-np.sin(thetas), -np.cos(thetas))) + 90
    a_arc = np.clip(a_arc, 1, 89)
    paths.append(np.column_stack([x_arc, y_arc, a_arc]))

    # Path 11: Another circular arc
    center_x, center_y, radius = -0.8, 0.6, 1.0
    theta_start = np.pi * 0.15
    theta_end = np.pi * 0.85
    thetas = np.linspace(theta_start, theta_end, 5)
    x_arc = center_x + radius * np.cos(thetas)
    y_arc = center_y + radius * np.sin(thetas)
    a_arc = np.degrees(np.arctan2(-np.sin(thetas), -np.cos(thetas))) + 90
    a_arc = np.clip(a_arc, 1, 89)
    paths.append(np.column_stack([x_arc, y_arc, a_arc]))

    return paths


def _nelder_mead(f, x0, max_iter=2000, step=0.05, tol=1e-6):
    """Nelder-Mead minimization with adaptive step size."""
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        pt = x0.copy()
        pt[i] += step
        simplex[i + 1] = pt
    vals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    current_step = step
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 20:
            break
        order = np.argsort(vals)
        simplex = simplex[order]
        vals = vals[order]
        
        # Adaptive tolerance
        spread = vals[-1] - vals[0]
        if spread < tol * max(1.0, abs(vals[0])):
            break

        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)

        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                vals[-1] = fe
            else:
                simplex[-1] = xr
                vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = xr
            vals[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1] = xc
                vals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i])
        
        # Occasionally shrink step if no improvement
        if it > 50 and it % 50 == 0:
            current_step *= 0.7

    best_i = int(np.argmin(vals))
    return simplex[best_i], vals[best_i]


def _coordinate_descent(path, X, Y, steps=3, step_size=0.02):
    """Coordinate descent refinement."""
    current = path.copy()
    n = len(current)

    for _ in range(steps):
        if _elapsed() > _DEADLINE - 10:
            break
        improved = False
        for i in range(n):
            for coord in range(3):
                best_val = _score(current, X, Y)
                best_coord = current[i, coord]
                for delta in [-step_size, -step_size/2, step_size/2, step_size]:
                    trial = current.copy()
                    trial[i, coord] = current[i, coord] + delta
                    if coord == 2:
                        trial[i, coord] = np.clip(trial[i, coord], 1.0, 89.0)
                    val = _score(trial, X, Y)
                    if val > best_val:
                        best_val = val
                        best_coord = trial[i, coord]
                current[i, coord] = best_coord
                if best_coord != path[i, coord]:
                    improved = True
        if not improved:
            break

    return current


def _polish_path(path, X_full, Y_full, X_sub, Y_sub):
    """High-precision polish pass using both grids."""
    current = path.copy()
    n = len(current)
    
    # Phase 1: Fine coordinate descent on subsampled grid
    for step_size in [0.01, 0.005, 0.002]:
        if _elapsed() > _DEADLINE - 15:
            break
        for _ in range(2):
            if _elapsed() > _DEADLINE - 15:
                break
            improved = False
            for i in range(n):
                for coord in range(3):
                    best_val = _score(current, X_sub, Y_sub)
                    best_coord = current[i, coord]
                    for delta in [-step_size, -step_size/2, step_size/2, step_size]:
                        trial = current.copy()
                        trial[i, coord] = current[i, coord] + delta
                        if coord == 2:
                            trial[i, coord] = np.clip(trial[i, coord], 1.0, 89.0)
                        val = _score(trial, X_sub, Y_sub)
                        if val > best_val:
                            best_val = val
                            best_coord = trial[i, coord]
                    current[i, coord] = best_coord
                    if abs(best_coord - path[i, coord]) > 1e-10:
                        improved = True
            if not improved:
                break
    
    # Phase 2: Verify and fine-tune on full grid
    s_full = _score(current, X_full, Y_full)
    for step_size in [0.005, 0.002, 0.001]:
        if _elapsed() > _DEADLINE - 10:
            break
        for _ in range(1):
            if _elapsed() > _DEADLINE - 10:
                break
            improved = False
            for i in range(n):
                for coord in range(3):
                    best_val = _score(current, X_full, Y_full)
                    best_coord = current[i, coord]
                    for delta in [-step_size, step_size]:
                        trial = current.copy()
                        trial[i, coord] = current[i, coord] + delta
                        if coord == 2:
                            trial[i, coord] = np.clip(trial[i, coord], 1.0, 89.0)
                        val = _score(trial, X_full, Y_full)
                        if val > best_val:
                            best_val = val
                            best_coord = trial[i, coord]
                    current[i, coord] = best_coord
                    if abs(best_coord - current[i, coord]) > 1e-10:
                        improved = True
            if not improved:
                break
    
    return current


def moving_sofa_2d() -> np.ndarray:
    """
    Returns a path (n,3) array of corridor inner-corner placements (x,y,a_deg).
    """
    # Full grid for final evaluation
    X_full, Y_full = _build_sofa_grid(1)
    # Subsampled grid for fast optimization
    X_sub, Y_sub = _build_sofa_grid(3)

    # Evaluate all initial paths on subsampled grid for speed
    initial_paths = _initial_paths()
    best_path = initial_paths[0]
    best_score_sub = _score(best_path, X_sub, Y_sub)

    for p in initial_paths[1:]:
        s = _score(p, X_sub, Y_sub)
        if s > best_score_sub:
            best_score_sub = s
            best_path = p

    # Verify best on full grid
    best_score_full = _score(best_path, X_full, Y_full)

    # Optimize with Nelder-Mead on (x,y) with fixed angles - use subsampled grid
    if _elapsed() < _DEADLINE - 150:
        angles_fixed = best_path[:, 2]
        n = len(angles_fixed)
        xy0 = best_path[:, :2].flatten()

        def f_xy(flat):
            p = np.empty((n, 3))
            p[:, 0] = flat[0::2]
            p[:, 1] = flat[1::2]
            p[:, 2] = angles_fixed
            return -_score(p, X_sub, Y_sub)

        xopt, _ = _nelder_mead(f_xy, xy0, max_iter=2000, step=0.04)
        p_opt = np.empty((n, 3))
        p_opt[:, 0] = xopt[0::2]
        p_opt[:, 1] = xopt[1::2]
        p_opt[:, 2] = angles_fixed
        s_opt_sub = _score(p_opt, X_sub, Y_sub)
        s_opt_full = _score(p_opt, X_full, Y_full)
        if s_opt_full > best_score_full:
            best_score_full = s_opt_full
            best_path = p_opt

    # Try optimizing all parameters (x, y, angles)
    if _elapsed() < _DEADLINE - 120:
        n = len(best_path)
        x0 = np.concatenate([best_path[:, 0], best_path[:, 1], best_path[:, 2]])

        def f_full(flat):
            x = flat[0:n]
            y = flat[n:2*n]
            angles = np.clip(flat[2*n:3*n], 1.0, 89.0)
            p = np.column_stack([x, y, angles])
            return -_score(p, X_sub, Y_sub)

        xopt, _ = _nelder_mead(f_full, x0, max_iter=1500, step=0.04)
        x = xopt[0:n]
        y = xopt[n:2*n]
        angles = np.clip(xopt[2*n:3*n], 1.0, 89.0)
        p_full = np.column_stack([x, y, angles])
        s_full_sub = _score(p_full, X_sub, Y_sub)
        s_full_full = _score(p_full, X_full, Y_full)
        if s_full_full > best_score_full:
            best_score_full = s_full_full
            best_path = p_full

    # Try different angle grids with optimization
    if _elapsed() < _DEADLINE - 80:
        angle_grids = [
            np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
            np.array([12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0]),
            np.array([15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0]),
            np.array([8.0, 18.0, 28.0, 38.0, 48.0, 58.0, 68.0, 78.0, 86.0]),
            np.array([10.0, 22.0, 34.0, 46.0, 58.0, 70.0, 82.0]),
            np.array([11.0, 23.0, 35.0, 47.0, 59.0, 71.0, 83.0]),
        ]

        for angles_grid in angle_grids:
            if _elapsed() > _DEADLINE - 40:
                break
            n = len(angles_grid)
            x_interp = np.interp(angles_grid, best_path[:, 2], best_path[:, 0])
            y_interp = np.interp(angles_grid, best_path[:, 2], best_path[:, 1])
            p_init = np.column_stack([x_interp, y_interp, angles_grid])

            s_init_sub = _score(p_init, X_sub, Y_sub)
            s_init_full = _score(p_init, X_full, Y_full)
            if s_init_full > best_score_full:
                best_score_full = s_init_full
                best_path = p_init

            if _elapsed() > _DEADLINE - 30:
                continue

            xy0 = p_init[:, :2].flatten()

            def f_grid(flat, ang=angles_grid, nn=n):
                p = np.empty((nn, 3))
                p[:, 0] = flat[0::2]
                p[:, 1] = flat[1::2]
                p[:, 2] = ang
                return -_score(p, X_sub, Y_sub)

            xopt, _ = _nelder_mead(f_grid, xy0, max_iter=1000, step=0.03)
            p_opt = np.empty((n, 3))
            p_opt[:, 0] = xopt[0::2]
            p_opt[:, 1] = xopt[1::2]
            p_opt[:, 2] = angles_grid
            s_opt_sub = _score(p_opt, X_sub, Y_sub)
            s_opt_full = _score(p_opt, X_full, Y_full)
            if s_opt_full > best_score_full:
                best_score_full = s_opt_full
                best_path = p_opt

    # High-precision polish pass
    if _elapsed() < _DEADLINE - 20:
        p_polished = _polish_path(best_path, X_full, Y_full, X_sub, Y_sub)
        s_polished = _score(p_polished, X_full, Y_full)
        if s_polished > best_score_full:
            best_score_full = s_polished
            best_path = p_polished

    # Final sanity check
    best_path = np.asarray(best_path, dtype=np.float64)
    if best_path.ndim != 2 or best_path.shape[1] != 3 or best_path.shape[0] < 1:
        best_path = _initial_paths()[0]
    if not np.all(np.isfinite(best_path)):
        best_path = _initial_paths()[0]

    return best_path


# EVOLVE-BLOCK-END
