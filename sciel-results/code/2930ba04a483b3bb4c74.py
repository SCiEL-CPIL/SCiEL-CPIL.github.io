"""Dzugutov cluster global minima: icosahedral shell seeds + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_DEADLINE = 280.0  # safety margin below 300s
_THREAD_LOCK = threading.Lock()
_A = 5.82
_B = 1.28
_a = 1.87
_b = 1.94
_c = 1.10
_d = 0.27
_m = 16


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _energy(pos):
    """Compute Dzugutov energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    mask = (r > 1e-10) & (r < _b)
    r2 = r[mask]
    e1 = _A * (r2 ** (-_m) - _B) * np.exp(_c / (r2 - _a))
    e2 = _B * np.exp(_d / (r2 - _b))
    # e1 is only for r < a, e2 for a <= r < b
    mask_a = r2 < _a
    e = np.where(mask_a, e1, e2)
    return np.sum(e)


def _icosahedron_vertices():
    """12 vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def _dodecahedron_vertices():
    """20 vertices of a regular dodecahedron (golden ratio based)."""
    phi = (1 + np.sqrt(5)) / 2
    iphi = 1.0 / phi
    v = []
    # 8 vertices: (±1, ±1, ±1)
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                v.append([sx, sy, sz])
    # 4 vertices: (0, ±1/phi, ±phi)
    for sy in [-1, 1]:
        for sz in [-phi, phi]:
            v.append([0, sy * iphi, sz])
    # 4 vertices: (±1/phi, ±phi, 0)
    for sx in [-1, 1]:
        for sy in [-phi, phi]:
            v.append([sx * iphi, sy, 0])
    # 4 vertices: (±phi, 0, ±1/phi)
    for sx in [-phi, phi]:
        for sz in [-1, 1]:
            v.append([sx, 0, sz * iphi])
    v = np.array(v, dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def _cube_vertices():
    """8 vertices of a cube."""
    v = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                v.append([sx, sy, sz])
    v = np.array(v, dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def _generate_icosahedral_cluster(n, shell_radii=None, twist=None, rng=None):
    """Generate a cluster with icosahedral shell structure.
    
    Shells: center (1), inner icosahedron (12), outer dodecahedron (20), 
    then additional shells for larger n.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    ico = _icosahedron_vertices()
    dod = _dodecahedron_vertices()
    cube = _cube_vertices()
    
    points = [np.zeros(3)]  # center atom
    
    # Shell 1: 12 icosahedron vertices
    r1 = 1.15
    twist1 = twist if twist is not None else rng.uniform(-0.3, 0.3)
    for v in ico:
        # Apply twist around z-axis
        angle = twist1
        ct, st = np.cos(angle), np.sin(angle)
        rotated = np.array([v[0]*ct - v[1]*st, v[0]*st + v[1]*ct, v[2]])
        points.append(rotated * r1)
    
    # Shell 2: 20 dodecahedron vertices
    r2 = 1.85
    twist2 = rng.uniform(-0.2, 0.2)
    for v in dod:
        angle = twist2
        ct, st = np.cos(angle), np.sin(angle)
        rotated = np.array([v[0]*ct - v[1]*st, v[0]*st + v[1]*ct, v[2]])
        points.append(rotated * r2)
    
    # Shell 3: additional icosahedron-like shell
    r3 = 2.5
    twist3 = rng.uniform(-0.2, 0.2)
    for v in ico:
        angle = twist3
        ct, st = np.cos(angle), np.sin(angle)
        rotated = np.array([v[0]*ct - v[1]*st, v[0]*st + v[1]*ct, v[2]])
        points.append(rotated * r3)
    
    # Shell 4: cube-like shell
    r4 = 3.1
    twist4 = rng.uniform(-0.2, 0.2)
    for v in cube:
        angle = twist4
        ct, st = np.cos(angle), np.sin(angle)
        rotated = np.array([v[0]*ct - v[1]*st, v[0]*st + v[1]*ct, v[2]])
        points.append(rotated * r4)
    
    # Shell 5: another dodecahedron
    r5 = 3.6
    twist5 = rng.uniform(-0.2, 0.2)
    for v in dod:
        angle = twist5
        ct, st = np.cos(angle), np.sin(angle)
        rotated = np.array([v[0]*ct - v[1]*st, v[0]*st + v[1]*ct, v[2]])
        points.append(rotated * r5)
    
    # Shell 6: icosahedron again
    r6 = 4.1
    twist6 = rng.uniform(-0.2, 0.2)
    for v in ico:
        angle = twist6
        ct, st = np.cos(angle), np.sin(angle)
        rotated = np.array([v[0]*ct - v[1]*st, v[0]*st + v[1]*ct, v[2]])
        points.append(rotated * r6)
    
    points = np.array(points)
    
    # Truncate to n atoms
    if len(points) >= n:
        points = points[:n]
    else:
        # Add more random points on a sphere
        while len(points) < n:
            p = rng.normal(size=3)
            p = p / np.linalg.norm(p) * rng.uniform(3.5, 5.0)
            points = np.vstack([points, p])
    
    return points


def _fix_minimum_distances(pos, min_dist=0.5, max_iter=1000, rng=None):
    """Fix any pairs closer than min_dist by pushing them apart."""
    if rng is None:
        rng = np.random.default_rng()
    pos = pos.copy()
    for _ in range(max_iter):
        n = len(pos)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(r, np.inf)
        mask = r < min_dist
        if not np.any(mask):
            break
        # Find violating pairs
        i, j = np.where(mask)
        # Only process first violation to avoid complexity
        i0, j0 = i[0], j[0]
        d = pos[i0] - pos[j0]
        dist = np.linalg.norm(d)
        if dist < 1e-10:
            d = rng.normal(size=3)
            dist = np.linalg.norm(d)
        d = d / dist
        # Push apart
        push = (min_dist - dist) / 2.0 + 0.01
        pos[i0] += d * push
        pos[j0] -= d * push
    return pos


def _basin_hopping_step(pos, rng, potential, parameter):
    """One basin hopping step: perturb, relax, accept if better."""
    n = len(pos)
    # Perturbation: random displacement
    step_size = rng.uniform(0.3, 0.8)
    perturbed = pos + rng.normal(scale=step_size, size=pos.shape)
    # Fix minimum distances
    perturbed = _fix_minimum_distances(perturbed, min_dist=0.5, rng=rng)
    # Relax
    energy, relaxed = relax(perturbed, potential, parameter)
    return energy, relaxed


def _construct_instance(instance):
    """Construct a single instance."""
    global _START_TIME
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(n * 12345)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse icosahedral seeds
    n_seeds = 8
    seeds = []
    for i in range(n_seeds):
        twist = rng.uniform(-0.5, 0.5) if i % 2 == 0 else None
        seed = _generate_icosahedral_cluster(n, twist=twist, rng=rng)
        seed = _fix_minimum_distances(seed, min_dist=0.5, rng=rng)
        seeds.append(seed)
    
    # Also add some random compact seeds
    for i in range(4):
        radius = 1.1 * n ** (1.0 / 3.0) * 0.8
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                    points.append(p)
        if len(points) == n:
            points = _fix_minimum_distances(np.array(points), min_dist=0.5, rng=rng)
            seeds.append(points)
    
    # Relax all seeds
    for seed in seeds:
        if _remaining() < 5.0:
            break
        try:
            energy, positions = relax(seed, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # Basin hopping from best position
    if best_positions is not None and _remaining() > 10.0:
        current_energy = best_energy
        current_positions = best_positions.copy()
        
        # Basin hopping loop
        bh_iter = 0
        max_bh = 30
        while bh_iter < max_bh and _remaining() > 15.0:
            new_energy, new_positions = _basin_hopping_step(
                current_positions, rng, potential, parameter
            )
            if new_energy < current_energy:
                current_energy = new_energy
                current_positions = new_positions
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_positions = new_positions
            bh_iter += 1
    
    if best_positions is None:
        # Fallback: simple random
        points = rng.normal(size=(n, 3)) * 2.0
        points = _fix_minimum_distances(points, min_dist=0.5, rng=rng)
        best_positions = points
    
    return best_positions


def construct(instance):
    """Entry point: construct cluster for given instance."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Each instance gets a fair share of time
    # 6 instances total, ~45s each
    instance_deadline = _START_TIME + 45.0
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(n * 98765)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse icosahedral seeds
    n_seeds = 6
    seeds = []
    for i in range(n_seeds):
        twist = rng.uniform(-0.4, 0.4) if i % 2 == 0 else None
        seed = _generate_icosahedral_cluster(n, twist=twist, rng=rng)
        seed = _fix_minimum_distances(seed, min_dist=0.5, rng=rng)
        seeds.append(seed)
    
    # Add random compact seeds
    for i in range(3):
        radius = 1.1 * n ** (1.0 / 3.0) * 0.8
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                    points.append(p)
        if len(points) == n:
            points = _fix_minimum_distances(np.array(points), min_dist=0.5, rng=rng)
            seeds.append(points)
    
    # Relax all seeds
    for seed in seeds:
        if time.time() > instance_deadline - 5.0:
            break
        try:
            energy, positions = relax(seed, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # Basin hopping from best position
    if best_positions is not None and time.time() < instance_deadline - 10.0:
        current_energy = best_energy
        current_positions = best_positions.copy()
        
        bh_iter = 0
        max_bh = 20
        while bh_iter < max_bh and time.time() < instance_deadline - 15.0:
            new_energy, new_positions = _basin_hopping_step(
                current_positions, rng, potential, parameter
            )
            if new_energy < current_energy:
                current_energy = new_energy
                current_positions = new_positions
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_positions = new_positions
            bh_iter += 1
    
    if best_positions is None:
        # Fallback
        points = rng.normal(size=(n, 3)) * 2.0
        points = _fix_minimum_distances(points, min_dist=0.5, rng=rng)
        best_positions = points
    
    return best_positions
# EVOLVE-BLOCK-END
