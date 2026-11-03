import numpy as np
import time
from scipy.optimize import least_squares

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    # Try multiple spacings and pick the one that gives best initial radius
    best_pts = None
    best_r = 0.0
    for attempt in range(8):
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * (0.92 + 0.04 * attempt)
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x, y, r))
        if len(rows) < n_points:
            continue
        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        # Sort by radius (prefer points closer to center for better packing)
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]
        # Add small random perturbation
        if attempt > 0:
            selected = selected + rng.normal(0, spacing * 0.02, selected.shape)
            # Ensure inside unit circle
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]
        # Compute initial radius
        d = np.linalg.norm(selected[:, np.newaxis, :] - selected[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)
        if r > best_r:
            best_r = r
            best_pts = selected.copy()
    if best_pts is None:
        # Fallback: simple grid
        spacing = 2.0 / (np.sqrt(n_points) + 1.0)
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]
    return best_pts


def _compute_radius(pts):
    """Compute the packing radius for given points."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    return min(min_pair, min_bound)


def _soft_min(values, T):
    """Soft minimum using log-sum-exp: -logsumexp(-values/T)*T."""
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0
    m = values.min()
    if m < -1e15:
        return m
    shifted = values / T
    shifted = shifted - shifted.max()
    s = np.sum(np.exp(shifted))
    return -T * (np.log(s) + shifted.max())


def _residuals(x, T):
    """Residuals for least_squares: we want to maximize the soft-min of distances."""
    pts = x.reshape(-1, 2)
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms  # shape (N,)
    # Pairwise half-distances
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    pair_half = d / 2.0
    # Collect all finite values
    all_vals = np.concatenate([pair_half.ravel(), boundary.ravel()])
    all_vals = all_vals[all_vals < np.inf]
    if len(all_vals) == 0:
        return np.array([0.0])
    softmin_val = _soft_min(all_vals, T)
    # We want to maximize softmin_val, so minimize -softmin_val
    # Return a scalar residual
    return np.array([-softmin_val])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Initial hexagonal packing
    pts = _hex_init(N, seed=42)
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)

    # Smooth optimization with multiple temperature values
    temperatures = [0.01, 0.005, 0.002, 0.001, 0.0005]
    x0 = pts.ravel()

    for T in temperatures:
        if time.time() > deadline - 20.0:
            break
        try:
            result = least_squares(
                _residuals,
                x0,
                args=(T,),
                method='trf',
                max_nfev=500,
                ftol=1e-10,
                xtol=1e-12,
                gtol=1e-10,
            )
            if time.time() > deadline:
                break
            new_pts = result.x.reshape(-1, 2)
            # Check validity
            norms = np.linalg.norm(new_pts, axis=1)
            if np.any(norms >= 1.0) or not np.all(np.isfinite(new_pts)):
                continue
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
                x0 = new_pts.ravel()
        except Exception:
            continue

    # Final polish: tighter tolerance, smaller temperature
    if time.time() < deadline - 5.0:
        try:
            T_final = 0.0001
            result = least_squares(
                _residuals,
                x0,
                args=(T_final,),
                method='trf',
                max_nfev=1000,
                ftol=1e-12,
                xtol=1e-14,
                gtol=1e-12,
            )
            new_pts = result.x.reshape(-1, 2)
            norms = np.linalg.norm(new_pts, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(new_pts)):
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
        except Exception:
            pass

    # Ensure validity: clip any points that might be outside
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask):
        # Fallback to original hex init
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)

    return best_pts
