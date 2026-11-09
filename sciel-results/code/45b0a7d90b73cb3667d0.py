"""Dzugutov cluster global minima search with icosahedral seeding."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

# Global time budget management
_TOTAL_BUDGET = 290.0  # seconds (300s limit with 10s safety margin)
_START_TIME = None
_INSTANCE_ORDER = [38, 45, 50, 55, 57, 60]
_INSTANCE_RESULTS = {}


def _elapsed():
    return time.time() - _START_TIME


def _remaining():
    return _TOTAL_BUDGET - _elapsed()


def _generate_icosahedron_shell(radius, n_atoms, rng):
    """Generate points on an icosahedron surface with n_atoms vertices."""
    # Icosahedron vertices (12 vertices)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize to unit edge length
    # Edge length of this icosahedron: distance between [0,1,phi] and [0,1,-phi] = 2*phi
    edge = 2 * phi
    verts = verts / edge  # now edge length = 1
    # Scale to desired radius (circumradius of unit-edge icosahedron)
    circumradius = np.sqrt(1 + phi**2) / (2 * phi)  # for unit edge
    verts = verts * (radius / circumradius)
    
    if n_atoms <= 12:
        return verts[:n_atoms]
    
    # For more atoms, add points on faces or use Fibonacci sphere
    # Use Fibonacci sphere for uniform distribution
    points = [verts]
    remaining = n_atoms - 12
    if remaining > 0:
        # Add points on a slightly larger sphere
        outer_radius = radius * 1.15
        fib_points = _fibonacci_sphere(remaining, outer_radius, rng)
        points.append(fib_points)
    
    return np.vstack(points)


def _fibonacci_sphere(n, radius, rng):
    """Generate n points on a sphere using Fibonacci spiral."""
    if n == 0:
        return np.empty((0, 3))
    points = np.zeros((n, 3))
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        y = 1 - 2 * i / (n - 1) if n > 1 else 0
        r = np.sqrt(1 - y**2)
        theta = golden_angle * i + rng.uniform(0, 2 * np.pi)
        points[i] = np.array([np.cos(theta) * r, y, np.sin(theta) * r]) * radius
    return points


def _generate_mackay_icosahedron(n_atoms, spacing=1.1, rng=None):
    """Generate a Mackay-type icosahedral cluster."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Core: 1 atom at center
    points = [np.array([0.0, 0.0, 0.0])]
    
    # First shell: 12 atoms (icosahedron vertices) at radius ~spacing
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    edge = 2 * phi
    verts = verts / edge * spacing
    points.extend([v.copy() for v in verts])
    
    if n_atoms <= 13:
        return np.array(points[:n_atoms])
    
    # Second shell: 42 atoms (octahedral + tetrahedral interstices) at radius ~2*spacing
    # For simplicity, use a larger sphere
    remaining = n_atoms - 13
    if remaining > 0:
        outer_radius = spacing * 1.9
        fib_points = _fibonacci_sphere(remaining, outer_radius, rng)
        points.extend([p.copy() for p in fib_points])
    
    return np.array(points[:n_atoms])


def _generate_compact_random(n_atoms, spacing=1.1, rng=None):
    """Generate a compact random cluster."""
    if rng is None:
        rng = np.random.default_rng(123)
    
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.8
    points = []
    min_dist = 0.55 * spacing
    
    # Start with a central atom
    points.append(np.array([0.0, 0.0, 0.0]))
    
    attempts = 0
    max_attempts = 10000
    while len(points) < n_atoms and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        if valid:
            points.append(p)
    
    # If we couldn't place all atoms, fill remaining randomly
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n_atoms])


def _generate_packed_lattice(n_atoms, spacing=1.1, rng=None):
    """Generate a compact BCC-like lattice cluster."""
    if rng is None:
        rng = np.random.default_rng(77)
    
    # BCC lattice points within a sphere
    points = []
    a = spacing
    max_coord = int(np.ceil(n_atoms ** (1.0/3.0) / 2)) + 2
    
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                # Simple cubic
                p = np.array([ix, iy, iz]) * a
                if len(points) < n_atoms:
                    points.append(p)
                # BCC center
                p2 = (np.array([ix, iy, iz]) + 0.5) * a
                if len(points) < n_atoms:
                    points.append(p2)
    
    # Sort by distance from center and take closest n_atoms
    points = np.array(points[:n_atoms * 2])
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n_atoms]]
    
    return points


def _relax_and_score(positions, potential, parameter, time_budget):
    """Relax positions and return energy, positions."""
    try:
        energy, relaxed = relax(positions, potential, parameter)
        return energy, relaxed
    except Exception:
        return np.inf, positions


def construct(instance):
    """Find low-energy Dzugutov cluster using icosahedral seeding and multi-start relaxation."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time budget for this instance
    time_per_instance = _remaining() / max(1, len([s for s in _INSTANCE_ORDER if s >= size]))
    time_per_instance = min(time_per_instance, 45.0)  # cap at 45s per instance
    
    rng = np.random.default_rng(size * 1000 + 7)
    
    # Generate diverse initial structures
    candidates = []
    
    # 1. Mackay icosahedron
    try:
        mackay = _generate_mackay_icosahedron(size, spacing=1.1, rng=rng)
        candidates.append(mackay)
    except Exception:
        pass
    
    # 2. Compact random
    try:
        rand = _generate_compact_random(size, spacing=1.1, rng=rng)
        candidates.append(rand)
    except Exception:
        pass
    
    # 3. BCC lattice
    try:
        bcc = _generate_packed_lattice(size, spacing=1.1, rng=rng)
        candidates.append(bcc)
    except Exception:
        pass
    
    # 4. Icosahedron shell with different radius
    try:
        shell = _generate_icosahedron_shell(1.1 * size ** (1.0/3.0) * 0.7, size, rng)
        candidates.append(shell)
    except Exception:
        pass
    
    # 5. Another random with different seed
    try:
        rand2 = _generate_compact_random(size, spacing=1.15, rng=np.random.default_rng(size * 2000 + 13))
        candidates.append(rand2)
    except Exception:
        pass
    
    # 6. BCC with different spacing
    try:
        bcc2 = _generate_packed_lattice(size, spacing=1.05, rng=np.random.default_rng(size * 3000 + 17))
        candidates.append(bcc2)
    except Exception:
        pass
    
    # Relax each candidate and keep the best
    best_energy = np.inf
    best_positions = None
    
    for i, cand in enumerate(candidates):
        if _remaining() < 5.0:
            break
        
        # Add small random perturbation to break symmetry
        perturbed = cand + rng.normal(0, 0.05, cand.shape)
        
        energy, positions = _relax_and_score(perturbed, potential, parameter, time_per_instance)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
    
    # If we still have time, do additional random restarts
    extra_time = time_per_instance - _elapsed() + (time.time() - _START_TIME) - _START_TIME  # approximate
    if best_positions is not None and _remaining() > 10.0:
        n_extra = min(4, int(_remaining() / 10.0))
        for j in range(n_extra):
            if _remaining() < 5.0:
                break
            extra = _generate_compact_random(size, spacing=1.1 + 0.1 * j, rng=np.random.default_rng(size * 5000 + j * 100))
            perturbed = extra + rng.normal(0, 0.08, extra.shape)
            energy, positions = _relax_and_score(perturbed, potential, parameter, time_per_instance)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
    
    # Final safety: ensure valid output
    if best_positions is None or len(best_positions) != size:
        # Fallback: simple random
        radius = 1.1 * size ** (1.0 / 3.0)
        best_positions = rng.uniform(-radius, radius, (size, 3))
    
    return best_positions
# EVOLVE-BLOCK-END
