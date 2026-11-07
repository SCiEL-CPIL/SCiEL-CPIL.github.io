"""Sutton-Chen transition-metal cluster global minima search.
Uses a portfolio approach: FCC/BCC seeds, random compact seeds, and
iterative local relaxation with diverse starting points."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START = time.time()
_DEADLINE = 295.0  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _START)


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
    mc = int(np.ceil(n ** (1.0 / 3.0))) + 2
    a = 1.0  # lattice constant
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                # FCC positions: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if len(pts) < n:
                        pts.append(p)
    if len(pts) < n:
        # Fill remaining with nearby points
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.uniform(-mc * a, mc * a, 3)
            pts.append(p)
    return np.array(pts[:n])


def _make_bcc(n):
    """Generate a compact BCC-like cluster of n atoms."""
    pts = []
    mc = int(np.ceil(n ** (1.0 / 3.0))) + 2
    a = 1.0
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                p = np.array([i, j, k]) * a
                if len(pts) < n:
                    pts.append(p)
                # BCC center
                pc = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                if len(pts) < n:
                    pts.append(pc)
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.uniform(-mc * a, mc * a, 3)
            pts.append(p)
    return np.array(pts[:n])


def _make_random_compact(n, seed, spacing=0.72):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If we couldn't place all, fill remaining randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _make_mackay(n):
    """Generate a Mackay icosahedron-like cluster."""
    pts = []
    # Mackay numbers: 1, 13, 55, 147, 309, ...
    # Build by shells
    # Center
    pts.append(np.array([0.0, 0.0, 0.0]))
    
    # icosahedron vertices (12 vertices)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ]
    ico = np.array(ico, dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Shell 1: 12 atoms at icosahedron vertices
    for v in ico:
        if len(pts) < n:
            pts.append(v * 1.0)
    
    # Shell 2: 42 atoms (Mackay second shell)
    # For simplicity, add more atoms on a sphere if needed
    if len(pts) < n:
        # Add atoms at midpoints of icosahedron edges
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(ico[i] - ico[j])
                if 0.9 < d < 1.1:  # adjacent vertices
                    mid = (ico[i] + ico[j]) / 2.0
                    mid = mid / np.linalg.norm(mid) * 1.5
                    edges.append(mid)
        for e in edges:
            if len(pts) < n:
                pts.append(e)
    
    if len(pts) < n:
        # Fill remaining on a larger sphere
        rng = np.random.default_rng(123)
        while len(pts) < n:
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0, np.pi)
            r = 2.0
            p = np.array([r * np.sin(phi) * np.cos(theta),
                         r * np.sin(phi) * np.sin(theta),
                         r * np.cos(phi)])
            pts.append(p)
    
    return np.array(pts[:n])


def _make_deformed_fcc(n, seed, deform=0.3):
    """FCC with random deformation."""
    pts = _make_fcc(n)
    rng = np.random.default_rng(seed)
    pts = pts + rng.normal(0, deform, pts.shape)
    return pts


def construct(instance):
    size = instance["n"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting points
    starts = []
    
    # FCC-based
    starts.append(_make_fcc(size))
    starts.append(_make_deformed_fcc(size, seed=1, deform=0.2))
    starts.append(_make_deformed_fcc(size, seed=2, deform=0.4))
    
    # BCC-based
    starts.append(_make_bcc(size))
    starts.append(_make_deformed_fcc(size, seed=3, deform=0.3))  # reuse as random-ish
    
    # Random compact
    for s in range(5):
        starts.append(_make_random_compact(size, seed=100 + s))
    
    # Mackay-like
    starts.append(_make_mackay(size))
    
    # Additional random starts
    for s in range(5, 12):
        starts.append(_make_random_compact(size, seed=200 + s))
    
    # Relax each starting point
    for idx, pos in enumerate(starts):
        if _time_left() < 10.0:
            break
        
        if not _is_valid(pos):
            continue
        
        try:
            energy, relaxed = relax(pos, "sutton_chen", kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed.copy()
        except Exception:
            continue
    
    # If we have time, do additional refinements
    if best_positions is not None and _time_left() > 30.0:
        # Try perturbing the best solution and re-relaxing
        rng = np.random.default_rng(999)
        for attempt in range(20):
            if _time_left() < 15.0:
                break
            
            perturbation = rng.normal(0, 0.15, best_positions.shape)
            perturbed = best_positions + perturbation
            
            if not _is_valid(perturbed):
                continue
            
            try:
                energy, relaxed = relax(perturbed, "sutton_chen", kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = relaxed.copy()
            except Exception:
                continue
    
    if best_positions is None:
        # Fallback: return a valid simple configuration
        best_positions = _make_fcc(size)
        try:
            _, best_positions = relax(best_positions, "sutton_chen", kind)
        except Exception:
            pass
    
    # Final validation
    if not _is_valid(best_positions):
        # Last resort: spread atoms evenly
        best_positions = np.zeros((size, 3))
        for i in range(size):
            best_positions[i] = [i * 0.5, (i // 10) * 0.5, (i % 10) * 0.5]
    
    return best_positions
# EVOLVE-BLOCK-END
