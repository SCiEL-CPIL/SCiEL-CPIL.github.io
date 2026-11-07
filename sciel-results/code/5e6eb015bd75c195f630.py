"""Portfolio-search program for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 300.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _make_fcc(n):
    """Generate a compact FCC-like cluster of n atoms."""
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 2
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                pts.append([i, j, k])
    # Keep only points within a sphere of radius ~ n^(1/3)
    radius = 0.6 * n ** (1.0 / 3.0)
    pts = [p for p in pts if np.linalg.norm(p) <= radius]
    # If not enough, add more from larger shell
    if len(pts) < n:
        mc2 = mc + 2
        for i in range(-mc2, mc2 + 1):
            for j in range(-mc2, mc2 + 1):
                for k in range(-mc2, mc2 + 1):
                    p = [i, j, k]
                    if p not in pts:
                        pts.append(p)
        pts = [p for p in pts if np.linalg.norm(p) <= radius + 1.5]
    
    # Sort by distance from center and take first n
    pts.sort(key=lambda p: np.linalg.norm(p))
    pts = pts[:n]
    if len(pts) < n:
        # Fill remaining with random points
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            pts.append(p.tolist())
    return np.array(pts, dtype=float)


def _make_bcc(n):
    """Generate a compact BCC-like cluster of n atoms."""
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 2
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                # FCC lattice points
                pts.append([i, j, k])
                # BCC center points
                pts.append([i + 0.5, j + 0.5, k + 0.5])
    radius = 0.6 * n ** (1.0 / 3.0)
    pts = [p for p in pts if np.linalg.norm(p) <= radius]
    if len(pts) < n:
        mc2 = mc + 2
        for i in range(-mc2, mc2 + 1):
            for j in range(-mc2, mc2 + 1):
                for k in range(-mc2, mc2 + 1):
                    for p in [[i, j, k], [i + 0.5, j + 0.5, k + 0.5]]:
                        if p not in pts:
                            pts.append(p)
        pts = [p for p in pts if np.linalg.norm(p) <= radius + 1.5]
    
    pts.sort(key=lambda p: np.linalg.norm(p))
    pts = pts[:n]
    if len(pts) < n:
        rng = np.random.default_rng(43)
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            pts.append(p.tolist())
    return np.array(pts, dtype=float)


def _make_random_cluster(n, seed=0):
    """Generate a random compact cluster of n atoms."""
    rng = np.random.default_rng(seed)
    radius = 0.7 * n ** (1.0 / 3.0)
    pts = []
    for _ in range(n):
        while True:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                break
        pts.append(p)
    return np.array(pts, dtype=float)


def construct(instance):
    """Portfolio search: try FCC, BCC, and random starts with relaxation."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate candidate starting structures
    candidates = [
        _make_fcc(size),
        _make_bcc(size),
        _make_random_cluster(size, seed=size),
        _make_random_cluster(size, seed=size + 100),
    ]
    
    for idx, start in enumerate(candidates):
        if _time_left() < 5.0:
            break
        try:
            energy, positions = relax(start, potential, parameter)
            if _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            continue
    
    if best_positions is None:
        # Fallback: return a valid random cluster
        best_positions = _make_random_cluster(size, seed=size + 200)
    
    return best_positions
# EVOLVE-BLOCK-END
