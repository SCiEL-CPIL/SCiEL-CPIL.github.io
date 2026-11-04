"""Circle packing: 300 equal circles in a unit circle.
Strategy: hexagonal lattice init + force-directed refinement with spatial hashing.
"""

import numpy as np
import time

N = 300


def construct():
    """Return centres of N equal circles inside the unit circle.

    Returns
    -------
    np.ndarray of shape (N, 2)
    """
    t_start = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - t_start

    # --- 1. Hexagonal lattice initialisation ---
    def hex_init():
        # Estimate spacing so that ~N points fit in unit disk
        # Area per hex cell ~ sqrt(3)/2 * s^2, total area ~ pi
        # N * sqrt(3)/2 * s^2 ~ pi  =>  s ~ sqrt(2*pi / (N*sqrt(3)))
        s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
        # Generate points on hex lattice
        k_max = int(1.0 / s) + 2
        pts = []
        for j in range(-k_max, k_max + 1):
            for i in range(-k_max, k_max + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                if x * x + y * y <= 0.95 ** 2:  # slight margin
                    pts.append((x, y))
        pts = np.array(pts)
        if len(pts) < N:
            # If not enough, expand slightly
            for scale in [1.05, 1.1, 1.2]:
                pts2 = []
                for j in range(-k_max - 1, k_max + 2):
                    for i in range(-k_max - 1, k_max + 2):
                        x = s * scale * (i + 0.5 * (j % 2))
                        y = s * scale * j * np.sqrt(3.0) / 2.0
                        if x * x + y * y <= 0.95 ** 2:
                            pts2.append((x, y))
                pts2 = np.array(pts2)
                if len(pts2) >= N:
                    pts = pts2
                    break
        # Select N closest to origin (denser centre)
        dist = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dist, kind='stable')[:N]
        return pts[idx]

    def compute_score(pts):
        """Compute the common radius r for a configuration."""
        n = len(pts)
        # Distance to boundary
        boundary = 1.0 - np.linalg.norm(pts, axis=1)
        min_boundary = np.min(boundary)
        if min_boundary < 0:
            return 0.0
        # Pairwise distances (use spatial hashing for speed)
        cell = max(0.02, 1.0 / (np.sqrt(N) * 1.5))
        inv_cell = 1.0 / cell
        min_dist = 1.0
        for i in range(n):
            ci = (int(pts[i, 0] * inv_cell), int(pts[i, 1] * inv_cell))
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    pass
        # O(N^2) fallback for correctness (N=300 is small enough)
        dx = pts[:, 0:1] - pts[0:1, :]
        dy = pts[:, 1:2] - pts[0:2, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        return min(min_boundary, min_pair / 2.0)

    # --- Spatial hashing helpers for force computation ---
    class SpatialHash:
        def __init__(self, pts, cell_size):
            self.cell = cell_size
            self.inv = 1.0 / cell_size
            self.map = {}
            n = len(pts)
            for i in range(n):
                cx = int(pts[i, 0] * self.inv)
                cy = int(pts[i, 1] * self.inv)
                key = (cx, cy)
                if key not in self.map:
                    self.map[key] = []
                self.map[key].append(i)

        def neighbors(self, i, pts):
            cx = int(pts[i, 0] * self.inv)
            cy = int(pts[i, 1] * self.inv)
            result = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    key = (cx + di, cy + dj)
                    if key in self.map:
                        result.extend(self.map[key])
            return result

    def force_step(pts, step):
        """One force-directed step: repel overlapping pairs, pull boundary points inward."""
        n = len(pts)
        cell = 0.05
        sh = SpatialHash(pts, cell)
        forces = np.zeros_like(pts)

        # Pairwise repulsion for close pairs
        for i in range(n):
            nbrs = sh.neighbors(i, pts)
            for j in nbrs:
                if j <= i:
                    continue
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-10:
                    # Random tiny push
                    forces[i, 0] += 0.001 * (np.random.rand() - 0.5)
                    forces[i, 1] += 0.001 * (np.random.rand() - 0.5)
                    forces[j, 0] -= 0.001 * (np.random.rand() - 0.5)
                    forces[j, 1] -= 0.001 * (np.random.rand() - 0.5)
                elif d < 0.15:
                    # Strong repulsion when close
                    f = 1.0 / (d * d + 1e-6)
                    forces[i, 0] += f * dx / d
                    forces[i, 1] += f * dy / d
                    forces[j, 0] -= f * dx / d
                    forces[j, 1] -= f * dy / d

        # Boundary: push inward if too close to edge
        r = np.linalg.norm(pts, axis=1)
        for i in range(n):
            if r[i] > 0.95:
                # Pull toward origin
                forces[i, 0] -= 2.0 * pts[i, 0] / r[i]
                forces[i, 1] -= 2.0 * pts[i, 1] / r[i]

        # Apply forces with adaptive step
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 0:
            scale = step / max_force
            pts = pts + forces * scale

        # Clamp to unit disk (with small margin)
        r = np.linalg.norm(pts, axis=1)
        mask = r > 0.999
        if np.any(mask):
            pts[mask] *= 0.999 / r[mask]

        return pts

    # --- Main optimization loop ---
    best_pts = hex_init()
    best_score = compute_score(best_pts)

    # Schedule of step sizes
    schedule = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]

    for step in schedule:
        n_steps = 300
        for _ in range(n_steps):
            if elapsed() > DEADLINE - 10:
                break
            best_pts = force_step(best_pts, step)
            s = compute_score(best_pts)
            if s > best_score:
                best_score = s
        if elapsed() > DEADLINE - 15:
            break

    # Final polish: tighter steps
    for step in [0.0001, 0.00005, 0.00002]:
        for _ in range(200):
            if elapsed() > DEADLINE - 5:
                break
            best_pts = force_step(best_pts, step)
            s = compute_score(best_pts)
            if s > best_score:
                best_score = s
        if elapsed() > DEADLINE - 5:
            break

    return best_pts
