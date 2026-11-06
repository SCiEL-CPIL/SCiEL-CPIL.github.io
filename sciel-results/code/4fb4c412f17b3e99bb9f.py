"""Rewritten program for Dzugutov cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time
from itertools import combinations

def _dzugutov_energy(positions, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """Compute Dzugutov potential energy."""
    n = len(positions)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < a:
                E += A * (r ** (-m) - B) * np.exp(c / (r - a))
            if r < b:
                E += B * np.exp(d / (r - b))
    return E

def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

def _get_mackay_icosahedron(n_atoms, spacing=1.0):
    """
    Generate a Mackay icosahedron cluster with n_atoms atoms.
    Shell structure:
      - Core: 1 atom
      - Shell 1 (12 atoms): icosahedron vertices
      - Shell 2 (42 atoms): second icosahedral shell
      - Shell 3 (92 atoms): third icosahedral shell
    Total: 1 + 12 + 42 + 92 = 147 atoms for full 3-shell Mackay icosahedron
    """
    ico_verts = _get_icosahedron_vertices()
    
    # Mackay icosahedron shell radii (relative to shell 1)
    phi = (1 + np.sqrt(5)) / 2
    r1 = spacing
    r2 = spacing * np.sqrt(1 + 2/phi)  # ≈ 1.401 * spacing
    r3 = spacing * np.sqrt(1 + 4/phi + 4/phi**2)  # ≈ 1.754 * spacing
    
    positions = [np.zeros(3)]  # Core atom
    
    # Shell 1: 12 atoms at icosahedron vertices
    for v in ico_verts:
        positions.append(v * r1)
    
    # Shell 2: 42 atoms
    if n_atoms > 13:
        # Generate shell 2 positions
        # Shell 2 of Mackay icosahedron consists of:
        # - 6 atoms at octahedral directions (face centers of icosahedron)
        # - 30 atoms at edge midpoints
        # - 6 atoms at additional positions
        
        # Octahedral directions
        oct_verts = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], dtype=float)
        for v in oct_verts:
            positions.append(v * r2)
        
        # Edge midpoints: icosahedron has 30 edges
        # Find edges by distance
        edge_len = np.linalg.norm(ico_verts[0] - ico_verts[1])
        edge_mids = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                if abs(d - edge_len) < 0.01:
                    mid = (ico_verts[i] + ico_verts[j]) / 2
                    mid = mid / np.linalg.norm(mid)
                    edge_mids.append(mid)
        
        for v in edge_mids:
            positions.append(v * r2)
        
        # Add 6 more positions to complete 42 in shell 2
        # These are at the "face centers" of the 20 triangular faces
        # Actually, the 6 octahedral vertices already cover some of these
        # Let's add 6 more at different positions
        for k in range(6):
            angle = k * 2 * np.pi / 6
            v = np.array([np.cos(angle), np.sin(angle), 0.5])
            v = v / np.linalg.norm(v)
            positions.append(v * r2)
    
    # Shell 3: 92 atoms (only needed for n > 55)
    if n_atoms > 55:
        # For shell 3, we need 92 positions
        # This is complex; for n up to 60, we need at most 5 atoms from shell 3
        # Place them at r3 in reasonable positions
        n_shell3 = n_atoms - len(positions)
        if n_shell3 > 0:
            # Distribute n_shell3 atoms on a sphere at radius r3
            for k in range(n_shell3):
                # Use Fibonacci sphere distribution
                i = k + 1
                golden_angle = np.pi * (3 - np.sqrt(5))
                z = 1 - 2 * i / n_shell3
                radius_xy = np.sqrt(1 - z**2)
                theta = golden_angle * i
                v = np.array([radius_xy * np.cos(theta), radius_xy * np.sin(theta), z])
                v = v / np.linalg.norm(v)
                positions.append(v * r3)
    
    # Trim to n_atoms
    positions = positions[:n_atoms]
    return np.array(positions, dtype=float)

def _defect_seeded_initialization(n_target, n_base=55, spacing=1.0):
    """
    Generate diverse candidates by removing atoms from a 55-atom Mackay icosahedron
    or by displacing specific high-symmetry vertices.
    """
    candidates = []
    
    # Start with 55-atom Mackay icosahedron
    base_cluster = _get_mackay_icosahedron(n_base, spacing)
    
    if n_target == n_base:
        # For 55 atoms, just return the base cluster
        candidates.append(base_cluster.copy())
        return candidates
    
    if n_target < n_base:
        # Remove atoms from the base cluster
        n_remove = n_base - n_target
        # Remove atoms from outer shells first (less critical for structure)
        # Sort by distance from center (outer atoms first)
        distances = np.linalg.norm(base_cluster, axis=1)
        sorted_indices = np.argsort(distances)[::-1]  # Farthest first
        
        # Remove n_remove atoms, starting from the outermost
        remove_indices = sorted_indices[:n_remove]
        keep_indices = np.setdiff1d(np.arange(n_base), remove_indices)
        candidates.append(base_cluster[keep_indices].copy())
        
        # Also try removing atoms from different positions
        # Remove some inner atoms
        sorted_indices_inner = np.argsort(distances)  # Closest first
        remove_indices_inner = sorted_indices_inner[:n_remove // 2]
        keep_indices_inner = np.setdiff1d(np.arange(n_base), remove_indices_inner)
        if len(keep_indices_inner) == n_target:
            candidates.append(base_cluster[keep_indices_inner].copy())
        
        # Random removals
        rng = np.random.default_rng(n_target)
        for _ in range(3):
            remove_indices_random = rng.choice(n_base, size=n_remove, replace=False)
            keep_indices_random = np.setdiff1d(np.arange(n_base), remove_indices_random)
            candidates.append(base_cluster[keep_indices_random].copy())
    
    else:
        # n_target > n_base: add atoms
        n_add = n_target - n_base
        # Add atoms at positions that maintain icosahedral symmetry
        # Place them at r3 positions
        ico_verts = _get_icosahedron_vertices()
        phi = (1 + np.sqrt(5)) / 2
        r3 = spacing * np.sqrt(1 + 4/phi + 4/phi**2)
        
        # Generate candidate positions on shell 3
        new_positions = []
        for k in range(n_add):
            angle = k * 2 * np.pi / n_add
            v = np.array([np.cos(angle), np.sin(angle), 0.3])
            v = v / np.linalg.norm(v)
            new_positions.append(v * r3)
        
        candidates.append(np.vstack([base_cluster, np.array(new_positions)]).copy())
        
        # Also try adding atoms at different positions
        rng = np.random.default_rng(n_target + 100)
        for _ in range(3):
            new_positions_random = []
            for _ in range(n_add):
                # Random point on sphere at r3
                theta = rng.uniform(0, 2 * np.pi)
                phi_angle = rng.uniform(0, np.pi)
                v = np.array([
                    np.sin(phi_angle) * np.cos(theta),
                    np.sin(phi_angle) * np.sin(theta),
                    np.cos(phi_angle)
                ])
                new_positions_random.append(v * r3)
            candidates.append(np.vstack([base_cluster, np.array(new_positions_random)]).copy())
    
    return candidates

def _perturbed_icosahedron(n, spacing=1.0, noise_scale=0.1, rng=None):
    """Generate a perturbed icosahedral cluster."""
    if rng is None:
        rng = np.random.default_rng(n)
    
    # Generate base icosahedral cluster
    base = _get_mackay_icosahedron(n, spacing)
    
    # Add random perturbations
    noise = rng.normal(0, noise_scale, base.shape)
    perturbed = base + noise
    
    # Ensure minimum distance constraint
    for i in range(len(perturbed)):
        for j in range(i + 1, len(perturbed)):
            d = np.linalg.norm(perturbed[i] - perturbed[j])
            if d < 0.5:
                # Push apart
                diff = perturbed[i] - perturbed[j]
                if np.linalg.norm(diff) > 1e-10:
                    diff = diff / np.linalg.norm(diff)
                else:
                    diff = np.array([1.0, 0, 0])
                perturbed[i] += diff * (0.5 - d) / 2
                perturbed[j] -= diff * (0.5 - d) / 2
    
    return perturbed

def _random_compact_cluster(n, spacing=1.0, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng(n)
    
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = n * 100
    
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check minimum distance
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
    
    # If we didn't get enough points, add more randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n], dtype=float)

def construct(instance):
    """
    Find low-energy structures for Dzugutov potential clusters.
    Uses defect-seeded initialization from Mackay icosahedra plus perturbed and random restarts.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    def try_relax(positions, label=""):
        nonlocal best_energy, best_positions
        if time.time() > deadline:
            return
        try:
            energy, relaxed = relax(positions, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except Exception:
            pass
    
    # Strategy 1: Defect-seeded initialization
    candidates = _defect_seeded_initialization(size, n_base=55, spacing=1.0)
    for i, cand in enumerate(candidates):
        if time.time() > deadline:
            break
        try_relax(cand, f"defect_{i}")
    
    # Strategy 2: Perturbed icosahedral structures
    for i in range(8):
        if time.time() > deadline:
            break
        noise_scale = 0.05 + 0.05 * i
        perturbed = _perturbed_icosahedron(size, spacing=1.0, noise_scale=noise_scale, rng=rng)
        try_relax(perturbed, f"perturbed_{i}")
    
    # Strategy 3: Random compact restarts
    for i in range(6):
        if time.time() > deadline:
            break
        random_cluster = _random_compact_cluster(size, spacing=1.0, rng=rng)
        try_relax(random_cluster, f"random_{i}")
    
    # Strategy 4: Basin hopping on the best structure found
    if best_positions is not None and time.time() < deadline - 30:
        current = best_positions.copy()
        current_energy = best_energy
        for step in range(20):
            if time.time() > deadline:
                break
            # Perturb
            perturbation_scale = 0.2 + 0.1 * np.random.rand()
            perturbed = current + rng.normal(0, perturbation_scale, current.shape)
            # Relax
            try:
                energy, relaxed = relax(perturbed, potential, parameter)
                if energy < current_energy:
                    current_energy = energy
                    current = relaxed
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = relaxed
            except Exception:
                pass
    
    if best_positions is None:
        # Fallback: return a random compact cluster
        best_positions = _random_compact_cluster(size, spacing=1.0, rng=rng)
    
    return best_positions
# EVOLVE-BLOCK-END
