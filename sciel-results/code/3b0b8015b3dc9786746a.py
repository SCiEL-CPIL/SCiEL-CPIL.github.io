"""Sutton-Chen transition-metal cluster global minima search with structured templates."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron, normalized to unit radius."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ], dtype=np.float64)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _fibonacci_sphere(n_pts, radius=1.0, rng=None):
    """Generate n_pts points uniformly on a sphere of given radius using Fibonacci lattice."""
    if n_pts <= 0:
        return np.zeros((0, 3))
    if n_pts == 1:
        return np.array([[0.0, 0.0, radius]])
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Fibonacci sphere
    i = np.arange(n_pts)
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n_pts)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * i  # golden angle
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])


def _fcc_cluster(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
    if n <= 1:
        return np.zeros((n, 3))
    
    # Generate FCC lattice points
    points = []
    # Central atom
    points.append(np.array([0.0, 0.0, 0.0]))
    
    # FCC neighbors in shells
    # First shell: 12 nearest neighbors at distance spacing
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = _icosahedron_vertices()
    for v in ico_verts:
        points.append(spacing * v)
    
    # If we need more, add second shell (FCC structure)
    if n > 13:
        # Second shell in FCC: atoms at (1,1,0) type positions
        second_shell = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    # FCC: all even coordinates or all odd with specific pattern
                    if (i + j + k) % 2 == 0:
                        p = np.array([i, j, k], dtype=np.float64) * spacing / np.sqrt(2)
                        # Avoid duplicates
                        if not any(np.linalg.norm(p - q) < 1e-10 for q in points):
                            second_shell.append(p)
        
        # Sort by distance from center
        second_shell.sort(key=lambda p: np.linalg.norm(p))
        for p in second_shell:
            if len(points) < n:
                points.append(p)
    
    # If still not enough, add more via Fibonacci sphere
    while len(points) < n:
        # Add on a sphere with radius increasing
        extra_n = n - len(points)
        radius = spacing * (1.5 + 0.3 * (len(points) - 13) / 10.0)
        extra = _fibonacci_sphere(extra_n, radius)
        for p in extra:
            if len(points) < n:
                points.append(p)
    
    return np.array(points[:n])


def _hcp_cluster(n, spacing=1.0):
    """Generate an HCP-like cluster of n atoms."""
    if n <= 1:
        return np.zeros((n, 3))
    
    points = []
    # Central atom
    points.append(np.array([0.0, 0.0, 0.0]))
    
    # First shell: 12 neighbors
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = _icosahedron_vertices()
    for v in ico_verts:
        points.append(spacing * v)
    
    # Second shell: HCP structure
    if n > 13:
        # HCP second shell has different arrangement
        # Use a hexagonal close-packed pattern
        hex_shell = []
        a = spacing / np.sqrt(2)
        c = spacing * np.sqrt(2/3)
        
        # Layer 1 (z = c/2)
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = a * (i + 0.5 * j)
                y = a * np.sqrt(3) / 2 * j
                z = c / 2
                p = np.array([x, y, z])
                if not any(np.linalg.norm(p - q) < 1e-10 for q in points):
                    hex_shell.append(p)
        
        # Layer 2 (z = -c/2)
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = a * (i + 0.5 * j)
                y = a * np.sqrt(3) / 2 * j
                z = -c / 2
                p = np.array([x, y, z])
                if not any(np.linalg.norm(p - q) < 1e-10 for q in points):
                    hex_shell.append(p)
        
        hex_shell.sort(key=lambda p: np.linalg.norm(p))
        for p in hex_shell:
            if len(points) < n:
                points.append(p)
    
    while len(points) < n:
        extra_n = n - len(points)
        radius = spacing * (1.5 + 0.3 * (len(points) - 13) / 10.0)
        extra = _fibonacci_sphere(extra_n, radius)
        for p in extra:
            if len(points) < n:
                points.append(p)
    
    return np.array(points[:n])


def _random_packed(n, spacing=0.72, rng=None):
    """Generate a randomly packed cluster with minimum distance constraints."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    min_dist = 0.7 * spacing
    
    points = [np.array([0.0, 0.0, 0.0])]
    attempts = 0
    max_attempts = n * 100
    
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    
    # If we didn't get enough, fill with Fibonacci sphere
    while len(points) < n:
        extra_n = n - len(points)
        radius = spacing * (1.5 + 0.3 * (len(points) - 1) / max(1, n - 1))
        extra = _fibonacci_sphere(extra_n, radius, rng)
        for p in extra:
            if len(points) < n:
                points.append(p)
    
    return np.array(points[:n])


def _perturb(positions, rng, scale=0.1):
    """Add random perturbation to positions."""
    return positions + rng.normal(0, scale, positions.shape)


def construct(instance):
    """Find low-energy Sutton-Chen cluster configurations using structured templates and relaxation."""
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s limit
    
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    # Get potential parameters
    params = {
        '12-6': (12, 6, 144.41),
        '9-6': (9, 6, 39.432),
        '10-8': (10, 8, 34.408)
    }
    n_exp, m_exp, c = params.get(kind, (12, 6, 144.41))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate initial templates
    templates = []
    
    # Icosahedral-based
    rng_ico = np.random.default_rng(100 + n)
    ico_base = _icosahedron_vertices()
    
    # Build icosahedral shells
    ico_points = [np.array([0.0, 0.0, 0.0])]
    for v in ico_base:
        ico_points.append(0.72 * v)
    
    # Add more shells if needed
    while len(ico_points) < n:
        shell_radius = 0.72 * (1.0 + 0.5 * (len(ico_points) - 13) / 10.0)
        n_shell = min(12, n - len(ico_points))
        shell_pts = _fibonacci_sphere(n_shell, shell_radius, rng_ico)
        ico_points.extend([p for p in shell_pts])
    
    templates.append(np.array(ico_points[:n]))
    
    # FCC template
    templates.append(_fcc_cluster(n, spacing=0.72))
    
    # HCP template
    templates.append(_hcp_cluster(n, spacing=0.72))
    
    # Random packed (multiple)
    for seed in range(5):
        rng = np.random.default_rng(seed * 1000 + n)
        templates.append(_random_packed(n, spacing=0.72, rng=rng))
    
    # Perturbed versions of best templates
    for i, tmpl in enumerate(templates[:4]):
        rng = np.random.default_rng(200 + i * 100 + n)
        templates.append(_perturb(tmpl, rng, scale=0.05))
    
    # Relax each template
    for idx, tmpl in enumerate(templates):
        if time.time() > deadline:
            break
        
        try:
            energy, positions = relax(tmpl, potential, kind)
            
            # Verify validity
            if not np.all(np.isfinite(positions)):
                continue
            
            # Check minimum distance
            min_dist = np.inf
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    d = np.linalg.norm(positions[i] - positions[j])
                    min_dist = min(min_dist, d)
            
            if min_dist < 0.35:
                continue
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        
        except Exception:
            continue
    
    # If we have time, do additional local optimization from best position
    if best_positions is not None and time.time() < deadline - 30:
        try:
            # Try to further optimize
            energy, positions = relax(best_positions, potential, kind)
            if np.all(np.isfinite(positions)) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
    
    # Final fallback if nothing worked
    if best_positions is None:
        # Simple random
        rng = np.random.default_rng(n)
        radius = 0.72 * n ** (1.0 / 3.0) * 0.8
        positions = rng.uniform(-radius, radius, (n, 3))
        try:
            energy, positions = relax(positions, potential, kind)
            if not np.all(np.isfinite(positions)):
                positions = rng.uniform(-radius, radius, (n, 3))
        except Exception:
            pass
        best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
