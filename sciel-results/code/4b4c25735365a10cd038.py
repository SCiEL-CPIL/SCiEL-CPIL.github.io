"""Morse cluster global minima search: structured starts + gradient polishing + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline: evaluator kills after 330s wall-clock.
# Use 305s as working limit for safety across all six construct() calls.
_DEADLINE = time.time() + 305.0

def _time_left():
    return _DEADLINE - time.time()

def _morse_energy(pos, rho):
    """Compute total Morse energy efficiently."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    x = rho * (1.0 - r)
    ex = np.exp(x)
    return np.sum(ex * (ex - 2.0))

def _build_fibonacci_sphere(n, radius=1.0):
    """Generate n points evenly distributed on a sphere using the Fibonacci spiral."""
    if n == 0:
        return np.zeros((0, 3))
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + np.sqrt(5.0)) * i
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])

def _build_icosahedral_core(n_atoms, spacing=1.0):
    """Build a configuration with an icosahedral core and outer shells."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])

    points = [np.zeros(3)]  # center

    # First shell: icosahedral vertices
    shell1 = ico_verts * spacing
    for v in shell1:
        if len(points) < n_atoms:
            points.append(v)

    # Second shell: larger icosahedral shell
    if len(points) < n_atoms:
        shell2 = ico_verts * spacing * 1.8
        for v in shell2:
            if len(points) < n_atoms:
                points.append(v)

    # Third shell: Fibonacci sphere
    if len(points) < n_atoms:
        n_remaining = n_atoms - len(points)
        shell3 = _build_fibonacci_sphere(n_remaining, spacing * 2.6)
        for v in shell3:
            if len(points) < n_atoms:
                points.append(v)

    # Fill any remaining with random points in a larger shell
    rng = np.random.default_rng(42)
    while len(points) < n_atoms:
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        r = spacing * (3.0 + rng.uniform(0, 0.5))
        p = np.array([r * np.sin(phi) * np.cos(theta),
                      r * np.sin(phi) * np.sin(theta),
                      r * np.cos(phi)])
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)

    return np.array(points[:n_atoms])

def _build_fcc_lattice(n_atoms, spacing=1.0):
    """Build an FCC-like lattice configuration."""
    points = []
    rng = np.random.default_rng(123)

    # Generate FCC lattice points in a cube
    side = int(np.ceil(n_atoms ** (1.0 / 3.0))) + 2
    fcc_offsets = np.array([
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]
    ], dtype=float)

    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for off in fcc_offsets:
                    p = np.array([i, j, k], dtype=float) * spacing + off * spacing
                    if len(points) < n_atoms:
                        points.append(p)
                if len(points) >= n_atoms:
                    break
            if len(points) >= n_atoms:
                break
        if len(points) >= n_atoms:
            break

    points = np.array(points[:n_atoms])

    # Center the cluster
    points -= np.mean(points, axis=0)

    # Add small random perturbation to break symmetry
    points += rng.normal(0, 0.05, points.shape)

    return points

def _build_random_compact(n_atoms, spacing=1.0, seed=0):
    """Build a random compact cluster using rejection sampling."""
    rng = np.random.default_rng(seed)
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    max_attempts = n_atoms * 200

    while len(points) < n_atoms and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)

    # If not enough, add more loosely
    while len(points) < n_atoms:
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        r = radius * (1.0 + rng.uniform(0, 0.5))
        p = np.array([r * np.sin(phi) * np.cos(theta),
                      r * np.sin(phi) * np.sin(theta),
                      r * np.cos(phi)])
        points.append(p)

    return np.array(points[:n_atoms])

def _basin_hop(pos, rho, rng, n_hops=8, step_scale=0.4):
    """Basin hopping: perturb, relax, accept if lower energy."""
    current_pos = pos.copy()
    current_energy = _morse_energy(current_pos, rho)

    for _ in range(n_hops):
        if _time_left() < 3:
            break

        # Perturb
        perturbed = current_pos + rng.normal(0, step_scale, current_pos.shape)

        # Relax
        try:
            energy, relaxed = relax(perturbed, "morse", rho)
        except Exception:
            continue

        # Check minimum separation
        if len(relaxed) >= 2:
            diff = relaxed[:, np.newaxis, :] - relaxed[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            if np.min(dist) < 0.49:
                continue

        # Accept if lower energy
        if energy < current_energy:
            current_energy = energy
            current_pos = relaxed.copy()

    return current_energy, current_pos

def construct(instance):
    """Find low-energy Morse cluster using structured starts, basin hopping, and gradient polishing."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))

    if potential != "morse":
        spacing = 1.0
        rng = np.random.default_rng(size)
        radius = spacing * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        energy, positions = relax(np.array(points), potential, parameter)
        return positions

    rho = parameter
    spacing = 1.0

    best_energy = np.inf
    best_positions = None

    # Generate diverse initial configurations
    initial_configs = []

    # 1. Icosahedral core with shells
    if _time_left() > 10:
        initial_configs.append(_build_icosahedral_core(size, spacing))

    # 2. FCC lattice
    if _time_left() > 10:
        initial_configs.append(_build_fcc_lattice(size, spacing))

    # 3-6. Random compact with different seeds
    for seed in [0, 1, 2, 3]:
        if _time_left() > 10:
            initial_configs.append(_build_random_compact(size, spacing, seed))

    # Relax each configuration and apply basin hopping
    rng = np.random.default_rng(999)
    for i, config in enumerate(initial_configs):
        if _time_left() < 5:
            break

        try:
            # Initial relaxation
            energy, positions = relax(config, "morse", rho)

            # Check minimum separation
            if len(positions) >= 2:
                diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dist, np.inf)
                if np.min(dist) < 0.49:
                    continue

            # Apply basin hopping
            bh_energy, bh_positions = _basin_hop(positions, rho, rng, n_hops=6, step_scale=0.35)

            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_positions.copy()
            elif energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue

    # If we still have time, try a few more random restarts with basin hopping
    extra_attempts = 0
    while _time_left() > 15 and extra_attempts < 6:
        extra_attempts += 1
        config = _build_random_compact(size, spacing, 100 + extra_attempts)
        try:
            energy, positions = relax(config, "morse", rho)

            if len(positions) >= 2:
                diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dist, np.inf)
                if np.min(dist) < 0.49:
                    continue

            bh_energy, bh_positions = _basin_hop(positions, rho, rng, n_hops=5, step_scale=0.4)

            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_positions.copy()
            elif energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue

    # Final gradient polishing on the best configuration
    if best_positions is not None and _time_left() > 5:
        try:
            # Use the provided gradient function for a final polish
            for _ in range(3):
                if _time_left() < 2:
                    break
                energy, polished = relax(best_positions, "morse", rho)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished.copy()
        except Exception:
            pass

    if best_positions is None:
        # Fallback
        config = _build_random_compact(size, spacing, 0)
        energy, best_positions = relax(config, "morse", rho)

    return best_positions
# EVOLVE-BLOCK-END
