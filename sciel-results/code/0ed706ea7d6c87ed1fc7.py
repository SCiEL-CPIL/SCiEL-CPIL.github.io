"""Circle packing in perimeter-4 rectangle – hexagonal-lattice + affine deformation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    # We pack N=21 circles into a rectangle of perimeter 4 (w + h = 2).
    # Strategy:
    #   1. Use a hexagonal (staggered) lattice as base positions (3 rows x 7 cols).
    #   2. Apply a local affine deformation (scale_x, scale_y, shear) to the lattice.
    #   3. Optimize the uniform radius r and the rectangle width w (h = 2 - w).
    #   4. Use L-BFGS-B with penalty for violations; validate at the end.

    def hex_lattice_positions(w, h, sx=1.0, sy=1.0, shear=0.0):
        """Generate 3x7 hexagonal lattice centers in [0,w]x[0,h] with affine transform."""
        rows, cols = 3, 7
        # Base cell dimensions
        cw = w / cols
        ch = h / rows
        # Hexagonal offset: alternate rows shifted by half cell width
        centers = np.zeros((N, 2))
        idx = 0
        for i in range(rows):
            for j in range(cols):
                # Base position (center of cell)
                bx = (j + 0.5) * cw
                by = (i + 0.5) * ch
                # Hex shift: odd rows shifted right by half cell
                if i % 2 == 1:
                    bx += cw * 0.5
                # Affine deformation around the center of the rectangle
                # Translate so origin is at rectangle center, apply affine, translate back
                dx = bx - w / 2.0
                dy = by - h / 2.0
                # Affine: x' = sx * dx + shear * dy, y' = sy * dy
                x_new = sx * dx + shear * dy
                y_new = sy * dy
                # Translate back
                cx = x_new + w / 2.0
                cy = y_new + h / 2.0
                centers[idx, 0] = cx
                centers[idx, 1] = cy
                idx += 1
        return centers

    def max_feasible_radius(centers, w, h):
        """Compute the maximum uniform radius such that all circles fit in [0,w]x[0,h] and don't overlap."""
        # Wall constraints
        r_wall = min(
            np.min(centers[:, 0]),
            np.min(w - centers[:, 0]),
            np.min(centers[:, 1]),
            np.min(h - centers[:, 1])
        )
        # Pairwise constraints
        r_pair = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                r_pair = min(r_pair, dist / 2.0)
        return max(0.0, min(r_wall, r_pair))

    def objective(z):
        """Penalty objective: minimize -r + penalties."""
        sx, sy, shear, w = z
        h = 2.0 - w
        centers = hex_lattice_positions(w, h, sx, sy, shear)
        r = max_feasible_radius(centers, w, h)
        
        # Penalty for centers going out of bounds (shouldn't happen if r is feasible, but guard)
        pen = 0.0
        for i in range(N):
            if centers[i, 0] < 0:
                pen += centers[i, 0] ** 2
            if centers[i, 0] > w:
                pen += (centers[i, 0] - w) ** 2
            if centers[i, 1] < 0:
                pen += centers[i, 1] ** 2
            if centers[i, 1] > h:
                pen += (centers[i, 1] - h) ** 2
        
        return -r + 100.0 * pen

    best_val = -np.inf
    best_params = None

    # Try a range of initial aspect ratios and deformation parameters
    # w in [0.5, 1.5] (h = 2 - w)
    w_candidates = np.linspace(0.6, 1.4, 9)  # 9 aspect ratios
    sx_candidates = [0.8, 0.9, 1.0, 1.1, 1.2]
    sy_candidates = [0.8, 0.9, 1.0, 1.1, 1.2]
    shear_candidates = [-0.3, -0.15, 0.0, 0.15, 0.3]

    bounds = [
        (0.5, 2.0),   # sx
        (0.5, 2.0),   # sy
        (-1.0, 1.0),  # shear
        (0.4, 1.6)    # w
    ]

    # First pass: quick evaluation of grid of starting points to find good region
    # Then refine with L-BFGS-B

    # Quick grid search
    grid_results = []
    for w0 in w_candidates:
        if time.time() > deadline - 60:
            break
        for sx0 in sx_candidates:
            for sy0 in sy_candidates:
                for sh0 in shear_candidates:
                    z0 = np.array([sx0, sy0, sh0, w0])
                    val = -objective(z0)
                    grid_results.append((val, z0))
    
    # Sort by value descending, take top 20 for local optimization
    grid_results.sort(key=lambda x: -x[0])
    top_starts = [z for v, z in grid_results[:20]]
    
    # Add a few random starts
    rng = np.random.RandomState(42)
    for _ in range(10):
        z0 = np.array([
            rng.uniform(0.7, 1.3),
            rng.uniform(0.7, 1.3),
            rng.uniform(-0.4, 0.4),
            rng.uniform(0.6, 1.4)
        ])
        top_starts.append(z0)

    # Local optimization from each start
    for z0 in top_starts:
        if time.time() > deadline - 10:
            break
        
        # Clip to bounds
        z0_clipped = np.array([
            np.clip(z0[0], 0.5, 2.0),
            np.clip(z0[1], 0.5, 2.0),
            np.clip(z0[2], -1.0, 1.0),
            np.clip(z0[3], 0.4, 1.6)
        ])
        
        try:
            res = minimize(objective, z0_clipped, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < -0.01:  # Valid improvement
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass

    # If optimization didn't find anything good, use best grid point
    if best_params is None and len(grid_results) > 0:
        best_val = grid_results[0][0]
        best_params = grid_results[0][1].copy()

    # Fallback: simple 3x7 grid
    if best_params is None or best_val < 1.5:
        w, h = 1.4, 0.6
        centers = np.zeros((N, 2))
        idx = 0
        for i in range(3):
            for j in range(7):
                centers[idx, 0] = (j + 0.5) * w / 7
                centers[idx, 1] = (i + 0.5) * h / 3
                idx += 1
        r = max_feasible_radius(centers, w, h)
        circles = np.zeros((N, 3))
        circles[:, :2] = centers
        circles[:, 2] = r
        return circles

    # Reconstruct final circles
    sx, sy, shear, w = best_params
    h = 2.0 - w
    centers = hex_lattice_positions(w, h, sx, sy, shear)
    r = max_feasible_radius(centers, w, h)
    
    # Slight shrink to ensure strict disjoint interiors (tiny epsilon)
    r = r * 0.9999
    
    circles = np.zeros((N, 3))
    circles[:, :2] = centers
    circles[:, 2] = r
    
    # Final validation: clip centers to ensure they're within bounds
    circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], w - circles[:, 2])
    circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], h - circles[:, 2])
    
    # Re-check pairwise after clipping (clipping can cause overlaps)
    # If any overlap, shrink radius slightly
    for _ in range(10):
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = min(min_dist, dist)
        r_needed = min_dist / 2.0
        if r <= r_needed * 1.0001:
            break
        r = r_needed * 0.9999
        circles[:, 2] = r
        # Re-clip
        circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], w - circles[:, 2])
        circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], h - circles[:, 2])
    
    return circles
