"""Fifty equal spheres in a unit cube – maximise the common radius."""
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

N = 50
CUBE_HALF = 0.5
DEADLINE_MARGIN = 5.0  # seconds before the 150 s wall-clock limit


def _score(centres: np.ndarray) -> float:
    """Return the packing radius for a set of centres inside [-0.5,0.5]^3."""
    if centres.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # wall clearance
    wall = CUBE_HALF - np.max(np.abs(centres), axis=1)
    if np.min(wall) < 0.0:
        return 0.0
    # pairwise distances
    d = cdist(centres, centres)
    np.fill_diagonal(d, np.inf)
    pair_min = np.min(d)
    return min(np.min(wall), pair_min / 2.0)


def _bcc_seed(n: int) -> np.ndarray:
    """Generate a BCC-lattice seed of n points in [-0.5,0.5]^3."""
    # BCC lattice: simple cubic + body-centre offset
    # Choose a grid that gives roughly n points
    m = int(np.ceil(n ** (1.0 / 3.0)))
    # Scale so the lattice fits in the cube
    # BCC unit cell has 2 points; we want ~n points total
    # Use a grid of m x m x m cells, each with 2 points (corner + centre)
    # But we only keep points inside the cube
    pts = []
    # Corner points (simple cubic)
    for ix in range(m + 1):
        for iy in range(m + 1):
            for iz in range(m + 1):
                x = -0.5 + ix * (1.0 / m)
                y = -0.5 + iy * (1.0 / m)
                z = -0.5 + iz * (1.0 / m)
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                    pts.append([x, y, z])
    # Body-centre points
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                x = -0.5 + (ix + 0.5) * (1.0 / m)
                y = -0.5 + (iy + 0.5) * (1.0 / m)
                z = -0.5 + (iz + 0.5) * (1.0 / m)
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                    pts.append([x, y, z])
    pts = np.array(pts, dtype=np.float64)
    if len(pts) < n:
        # Pad with random points
        extra = np.random.uniform(-0.45, 0.45, size=(n - len(pts), 3))
        pts = np.vstack([pts, extra])
    else:
        # Randomly select n points
        idx = np.random.choice(len(pts), size=n, replace=False)
        pts = pts[idx]
    return pts[:n]


def _cubic_seed(n: int) -> np.ndarray:
    """Simple cubic grid seed."""
    m = int(np.ceil(n ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    if len(grid) < n:
        extra = np.random.uniform(-0.45, 0.45, size=(n - len(grid), 3))
        grid = np.vstack([grid, extra])
    else:
        idx = np.random.choice(len(grid), size=n, replace=False)
        grid = grid[idx]
    return grid[:n]


def _lloyd_voronoi(centres: np.ndarray, n_iter: int, deadline: float) -> np.ndarray:
    """Lloyd's algorithm using Voronoi cell centroids to equalize clearance."""
    pts = centres.copy()
    for it in range(n_iter):
        if time.time() > deadline:
            break
        try:
            tri = Delaunay(pts)
        except Exception:
            break
        # For each point, compute the centroid of its Voronoi cell
        # Using scipy's Delaunay, we can get the Voronoi vertices
        from scipy.spatial import Voronoi
        vor = Voronoi(pts)
        new_pts = pts.copy()
        for i in range(len(pts)):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                continue
            verts = vor.vertices[region]
            centroid = np.mean(verts, axis=0)
            # Move towards centroid but keep within cube
            new_pts[i] = centroid
        # Clamp to cube
        new_pts = np.clip(new_pts, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
        # Check if all points are still valid (finite, inside)
        if not np.all(np.isfinite(new_pts)):
            break
        # Accept if score improved or is similar
        s_old = _score(pts)
        s_new = _score(new_pts)
        if s_new >= s_old - 1e-12:
            pts = new_pts
        else:
            # Try a smaller step
            step = 0.5
            for _ in range(5):
                candidate = pts + step * (new_pts - pts)
                candidate = np.clip(candidate, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
                s_cand = _score(candidate)
                if s_cand >= s_old - 1e-12:
                    pts = candidate
                    break
                step *= 0.5
    return pts


def _local_search(centres: np.ndarray, deadline: float, n_restarts: int = 3) -> np.ndarray:
    """Local optimization using pairwise repulsion + wall attraction."""
    best = centres.copy()
    best_score = _score(best)

    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        pts = best.copy()
        # Add small perturbation
        if restart > 0:
            pts = pts + np.random.normal(0, 0.01, size=pts.shape)
            pts = np.clip(pts, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)

        # Iterative local optimization
        lr = 0.05
        for it in range(500):
            if time.time() > deadline:
                break
            # Compute forces
            forces = np.zeros_like(pts)
            d = cdist(pts, pts)
            np.fill_diagonal(d, 1e10)
            # Repulsion between close pairs
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    dist = d[i, j]
                    if dist < 0.15:
                        # Strong repulsion when too close
                        force_dir = (pts[i] - pts[j]) / max(dist, 1e-10)
                        forces[i] += force_dir * (0.15 - dist)
            # Wall attraction: push points away from walls
            for i in range(N):
                for k in range(3):
                    if pts[i, k] > 0:
                        forces[i, k] -= pts[i, k] * 0.1
                    else:
                        forces[i, k] -= pts[i, k] * 0.1

            # Update
            pts_new = pts + lr * forces
            pts_new = np.clip(pts_new, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)

            s_old = _score(pts)
            s_new = _score(pts_new)
            if s_new >= s_old:
                pts = pts_new
                lr = min(lr * 1.01, 0.1)
            else:
                lr *= 0.7

            if lr < 1e-5:
                break

        s_final = _score(pts)
        if s_final > best_score:
            best = pts
            best_score = s_final

    return best


def construct() -> np.ndarray:
    """Construct 50 sphere centres in the unit cube to maximise packing radius."""
    t_start = time.time()
    deadline = t_start + 150.0 - DEADLINE_MARGIN

    np.random.seed(42)

    # Generate multiple seeds
    seeds = []
    seeds.append(_cubic_seed(N))
    seeds.append(_bcc_seed(N))
    # Random seeds
    for _ in range(3):
        seeds.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))

    best = None
    best_score = 0.0

    for seed in seeds:
        if time.time() > deadline:
            break

        # Lloyd's algorithm for equalizing clearance
        pts = _lloyd_voronoi(seed, n_iter=200, deadline=deadline)

        # Local search for further refinement
        pts = _local_search(pts, deadline=deadline, n_restarts=2)

        s = _score(pts)
        if s > best_score:
            best = pts
            best_score = s

    # Final polish: very fine local search
    if best is not None and time.time() < deadline:
        best = _local_search(best, deadline=deadline, n_restarts=3)
        best = _lloyd_voronoi(best, n_iter=100, deadline=deadline)

    # Ensure validity
    if best is None:
        best = _cubic_seed(N)
    best = np.clip(best, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    return best
