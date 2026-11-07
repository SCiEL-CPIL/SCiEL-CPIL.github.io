"""Portfolio-search program for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 290.0  # safety margin


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    if positions is None:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Check all pair distances
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _make_fcc(n):
    """Generate n atoms in FCC-like arrangement."""
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 3
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                # FCC lattice: (i,j,k) where i+j+k is even, or face-centered
                # Actually let's do a simple cubic with FCC-like density
                pts.append(np.array([i, j, k], dtype=float))
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
        if len(pts) >= n:
            break
    pts = pts[:n]
    if len(pts) < n:
        # Fill with random points near existing ones
        rng = np.random.default_rng(42)
        while len(pts) < n:
            base = pts[rng.integers(len(pts))]
            p = base + rng.normal(0, 0.3, 3)
            pts.append(p)
    return np.array(pts)


def _make_bcc(n):
    """Generate n atoms in BCC-like arrangement."""
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 3
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                # BCC: corners + body center
                pts.append(np.array([i, j, k], dtype=float))
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
        if len(pts) >= n:
            break
    pts = pts[:n]
    if len(pts) < n:
        rng = np.random.default_rng(43)
        while len(pts) < n:
            base = pts[rng.integers(len(pts))]
            p = base + rng.normal(0, 0.3, 3)
            pts.append(p)
    return np.array(pts)


def _make_icosahedral(n):
    """Generate n atoms in icosahedral shell structure."""
    pts = []
    # Center
    pts.append(np.array([0.0, 0.0, 0.0]))
    
    # Icosahedral vertices (12)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    ico_verts = np.array(ico_verts, dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])  # normalize to unit sphere
    
    # First shell: 12 icosahedral vertices
    for v in ico_verts:
        if len(pts) < n:
            pts.append(v * 1.0)
    
    # Second shell: 30 vertices (edge midpoints of icosahedron)
    if len(pts) < n:
        # Generate points on edges
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                if d < 1.1:  # close enough to be an edge
                    edges.append((i, j))
        for i, j in edges:
            if len(pts) < n:
                mid = (ico_verts[i] + ico_verts[j]) / 2
                mid /= np.linalg.norm(mid)
                pts.append(mid * 1.4)
    
    # Third shell: more points
    if len(pts) < n:
        rng = np.random.default_rng(44)
        while len(pts) < n:
            # Random point on sphere, radius ~1.8
            dir = rng.normal(0, 1, 3)
            dir /= np.linalg.norm(dir)
            pts.append(dir * 1.8)
    
    return np.array(pts[:n])


def _make_random_compact(n, seed):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    spacing = 0.72
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    pts = []
    for _ in range(10000):
        if len(pts) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in pts):
                pts.append(p)
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)


def _perturb(positions, sigma, seed):
    """Add Gaussian noise to positions."""
    rng = np.random.default_rng(seed)
    return positions + rng.normal(0, sigma, positions.shape)


def construct(instance):
    """Multi-start portfolio search with structured and random initial geometries."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting configurations
    starts = []
    
    # Structured starts
    starts.append(("fcc", _make_fcc(size)))
    starts.append(("bcc", _make_bcc(size)))
    starts.append(("ico", _make_icosahedral(size)))
    
    # Random compact starts with different seeds
    for s in range(6):
        starts.append((f"rand_{s}", _make_random_compact(size, 100 + s)))
    
    # Perturbed versions of structured starts
    for s, (name, pos) in enumerate(starts[:3]):
        for sigma in [0.05, 0.1, 0.15]:
            starts.append((f"{name}_pert_{sigma}", _perturb(pos, sigma, 200 + s * 10 + int(sigma * 100))))
    
    # Try each start
    for idx, (name, pos) in enumerate(starts):
        if _time_left() < 15.0:
            break
        
        if not _is_valid(pos):
            continue
        
        try:
            energy, positions = relax(pos, potential, kind)
        except Exception:
            continue
        
        if not _is_valid(positions):
            continue
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
    
    # If we have time, do more random restarts
    if best_positions is not None and _time_left() > 30.0:
        for s in range(20):
            if _time_left() < 10.0:
                break
            pos = _make_random_compact(size, 500 + s)
            if not _is_valid(pos):
                continue
            try:
                energy, positions = relax(pos, potential, kind)
            except Exception:
                continue
            if not _is_valid(positions):
                continue
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
    
    # Final check
    if best_positions is None or not _is_valid(best_positions):
        # Fallback: simple random
        rng = np.random.default_rng(0)
        best_positions = rng.uniform(-2, 2, (size, 3))
        # Ensure validity
        for _ in range(100):
            if _is_valid(best_positions):
                break
            best_positions = rng.uniform(-2, 2, (size, 3))
    
    return best_positions
# EVOLVE-BLOCK-END
