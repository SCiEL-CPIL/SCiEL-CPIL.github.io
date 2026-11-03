"""Circle packing in a unit disk: 50 equal circles, maximize radius."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 50
DEADLINE = 140.0  # seconds


def _score(pts):
    """Compute the radius score for a set of points."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = pdist(pts)
    min_dist = dists.min() if len(dists) > 0 else 0.0
    radii = np.hypot(pts[:, 0], pts[:, 1])
    min_margin = (1.0 - radii).min()
    r = min(min_dist / 2.0, min_margin)
    return r if r > 0 else 0.0


def _objective(x, pts_init):
    """Objective: negative of the packing radius (to minimize)."""
    pts = x.reshape(N, 2)
    return -_score(pts)


def construct():
    t_start = time.time()
    best_score = 0.0
    best_pts = None

    # Known ring structure for 50 circles: 1 center + rings
    # Known packing uses rings: 1, 6, 12, 18, 13 (or similar)
    # Let's use a parametric ring-based initialization

    def make_ring_pts(ring_radii, ring_counts):
        """Generate points on concentric rings."""
        pts = []
        for r, n in zip(ring_radii, ring_counts):
            if n == 1:
                pts.append([0.0, 0.0])
            elif n > 1:
                angles = 2 * np.pi * np.arange(n) / n
                pts.extend(list(zip(r * np.cos(angles), r * np.sin(angles))))
        return np.array(pts[:N])

    # Try several ring configurations
    # Known best packing for N=50 has r ≈ 0.1258
    # Ring radii should be roughly: center, then rings at increasing distances
    # For 50 circles with r≈0.126, the outer ring is at about 1-0.126=0.874
    # Try: 1 + 6 + 12 + 18 + 13 = 50
    ring_configs = [
        ([0.0, 0.35, 0.6, 0.82, 0.95], [1, 6, 12, 18, 13]),
        ([0.0, 0.30, 0.55, 0.78, 0.93], [1, 6, 12, 18, 13]),
        ([0.0, 0.40, 0.65, 0.85, 0.96], [1, 6, 12, 18, 13]),
        ([0.0, 0.33, 0.58, 0.80, 0.94], [1, 6, 12, 18, 13]),
        ([0.0, 0.38, 0.62, 0.83, 0.95], [1, 6, 12, 18, 13]),
        # Alternative splits
        ([0.0, 0.35, 0.6, 0.85, 0.95], [1, 6, 12, 18, 13]),
        ([0.0, 0.32, 0.56, 0.79, 0.93], [1, 6, 12, 18, 13]),
        ([0.0, 0.36, 0.61, 0.84, 0.96], [1, 6, 12, 18, 13]),
    ]

    # Also try a hexagonal lattice cut as fallback
    def hex_lattice_init():
        pts = []
        s = 0.2
        for j in range(-6, 7):
            for i in range(-6, 7):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) < 0.9:
                    pts.append([x, y])
        pts = np.array(pts)
        # Take closest N to center
        d = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(d)[:N]
        return pts[idx]

    # Try each initialization
    inits = []
    for rr, cc in ring_configs:
        inits.append(make_ring_pts(rr, cc))
    inits.append(hex_lattice_init())

    for init in inits:
        if time.time() - t_start > DEADLINE:
            break
        if init.shape != (N, 2):
            continue
        if not np.all(np.isfinite(init)):
            continue
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()

    # Local optimization using L-BFGS-B
    # We optimize over the 2D positions directly
    # Use the best initial point
    if best_pts is not None:
        x0 = best_pts.flatten()
        # Add small random perturbations to avoid local minima
        np.random.seed(42)

        for attempt in range(5):
            if time.time() - t_start > DEADLINE - 5:
                break
            if attempt == 0:
                x_init = x0
            else:
                # Perturb
                pert = np.random.randn(N, 2) * 0.05
                x_init = x0 + pert.flatten()

            try:
                res = minimize(
                    _objective,
                    x_init,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                pts = res.x.reshape(N, 2)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
            except Exception:
                pass

    # Final polish: a few more iterations from the best point with tighter tolerance
    if best_pts is not None and time.time() - t_start < DEADLINE - 2:
        x0 = best_pts.flatten()
        try:
            res = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            pts = res.x.reshape(N, 2)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass

    if best_pts is None:
        # Fallback: hexagonal lattice
        best_pts = hex_lattice_init()

    return best_pts
