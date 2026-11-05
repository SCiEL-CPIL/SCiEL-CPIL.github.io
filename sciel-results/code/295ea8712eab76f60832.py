"""
Strategy: Multi-start optimization with diverse initializations for Morse clusters.
Uses crystalline templates (FCC, BCC, HCP-like) and random starts, each refined
with L-BFGS-B using the provided gradient. Deadline guard ensures completion.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START = None
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s kill


def _get_remaining_time():
    global _GLOBAL_START
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()
    return _DEADLINE_SECONDS - (time.time() - _GLOBAL_START)


def _morse_energy(pos, rho):
    """Compute Morse energy for a configuration."""
    n = len(pos)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Mask out self-interactions
    mask = ~np.eye(n, dtype=bool)
    dist = dist[mask]
    # Morse potential: exp(rho*(1-r)) * (exp(rho*(1-r)) - 2)
    x = rho * (1.0 - dist)
    ex = np.exp(x)
    energy = np.sum(ex * (ex - 2.0))
    return energy


def _morse_energy_and_grad(pos, rho):
    """Compute energy and gradient for L-BFGS-B."""
    energy, grad = morse_energy_gradient(pos, rho)
    return energy, grad


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like cluster with n atoms."""
    # FCC lattice vectors
    a = spacing
    # Generate FCC lattice points
    points = []
    # Use a cubic chunk of FCC lattice
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of a
    basis = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
    ]
    # Estimate needed lattice constant
    # Volume per atom in FCC is a^3/4, so for n atoms, volume ~ n*a^3/4
    # We want a cluster of roughly n^(1/3) atoms per side
    side = int(math.ceil(n ** (1.0 / 3.0))) + 2
    lattice_points = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i + b[0], j + b[1], k + b[2]]) * a
                    lattice_points.append(p)
    lattice_points = np.array(lattice_points)
    # Center of mass
    com = np.mean(lattice_points, axis=0)
    lattice_points -= com
    # Sort by distance from center
    dists = np.linalg.norm(lattice_points, axis=1)
    order = np.argsort(dists)
    lattice_points = lattice_points[order]
    # Take first n
    if len(lattice_points) >= n:
        return lattice_points[:n].copy()
    else:
        # Pad with random if not enough
        remaining = n - len(lattice_points)
        extra = np.random.randn(remaining, 3) * spacing * 0.5
        return np.vstack([lattice_points, extra])


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like cluster with n atoms."""
    a = spacing
    basis = [
        [0, 0, 0],
        [0.5, 0.5, 0.5],
    ]
    side = int(math.ceil(n ** (1.0 / 3.0))) + 2
    lattice_points = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i + b[0], j + b[1], k + b[2]]) * a
                    lattice_points.append(p)
    lattice_points = np.array(lattice_points)
    com = np.mean(lattice_points, axis=0)
    lattice_points -= com
    dists = np.linalg.norm(lattice_points, axis=1)
    order = np.argsort(dists)
    lattice_points = lattice_points[order]
    if len(lattice_points) >= n:
        return lattice_points[:n].copy()
    else:
        remaining = n - len(lattice_points)
        extra = np.random.randn(remaining, 3) * spacing * 0.5
        return np.vstack([lattice_points, extra])


def _generate_hcp(n, spacing=1.0):
    """Generate an HCP-like cluster with n atoms."""
    a = spacing
    c = math.sqrt(8.0 / 3.0) * a  # Ideal HCP c/a ratio
    # HCP basis in a hexagonal cell
    # Layer A: (0,0,0), (1,0,0), (0,1,0), (1,1,0) in hex coordinates
    # Convert to Cartesian
    def hex_to_cart(hx, hy, hz):
        x = a * (hx + 0.5 * hy)
        y = a * (math.sqrt(3) / 2.0) * hy
        z = c * hz
        return np.array([x, y, z])
    
    points = []
    side = int(math.ceil(n ** (1.0 / 3.0))) + 2
    # Generate layers
    for layer in range(-side, side + 1):
        # Alternate A and B layers
        if layer % 2 == 0:
            # A layer
            for i in range(-side, side + 1):
                for j in range(-side, side + 1):
                    p = hex_to_cart(i, j, layer)
                    points.append(p)
        else:
            # B layer: shifted
            for i in range(-side, side + 1):
                for j in range(-side, side + 1):
                    p = hex_to_cart(i + 1.0 / 3.0, j + 2.0 / 3.0, layer)
                    points.append(p)
    
    points = np.array(points)
    com = np.mean(points, axis=0)
    points -= com
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order]
    if len(points) >= n:
        return points[:n].copy()
    else:
        remaining = n - len(points)
        extra = np.random.randn(remaining, 3) * spacing * 0.5
        return np.vstack([points, extra])


def _generate_random(n, spacing=1.0, seed=None):
    """Generate a random cluster with n atoms."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check minimum distance
            if len(points) > 0:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.5 * spacing:
                    points.append(p)
            else:
                points.append(p)
    return np.array(points)


def _jitter_crystal(points, jitter_amount, seed=None):
    """Add small random jitter to a crystal structure."""
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, jitter_amount, points.shape)
    return points + jitter


def construct(instance):
    """Find low-energy arrangements of atoms bound by Morse potentials."""
    global _GLOBAL_START
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Generate diverse initializations
    inits = []
    
    # Crystalline templates
    inits.append(_generate_fcc(size, spacing=1.0))
    inits.append(_generate_bcc(size, spacing=1.0))
    inits.append(_generate_hcp(size, spacing=1.0))
    
    # Jittered versions
    inits.append(_jitter_crystal(_generate_fcc(size, spacing=1.0), 0.1, seed=42))
    inits.append(_jitter_crystal(_generate_bcc(size, spacing=1.0), 0.1, seed=43))
    inits.append(_jitter_crystal(_generate_hcp(size, spacing=1.0), 0.1, seed=44))
    
    # Random starts
    inits.append(_generate_random(size, spacing=1.0, seed=100))
    inits.append(_generate_random(size, spacing=1.0, seed=200))
    inits.append(_generate_random(size, spacing=1.0, seed=300))
    inits.append(_generate_random(size, spacing=1.0, seed=400))
    
    # More random with different seeds
    for seed in [500, 600, 700, 800]:
        inits.append(_generate_random(size, spacing=1.0, seed=seed))
    
    # Optimize each initialization
    best_energy = np.inf
    best_positions = None
    
    for i, init in enumerate(inits):
        remaining = _get_remaining_time()
        if remaining < 5.0:
            break
        
        # Ensure valid minimum distances
        # If any pair is too close, jitter apart
        n = len(init)
        for _ in range(10):
            min_dist = np.inf
            for a in range(n):
                for b in range(a + 1, n):
                    d = np.linalg.norm(init[a] - init[b])
                    if d < min_dist:
                        min_dist = d
            if min_dist >= 0.5:
                break
            # Jitter the closest pair
            for a in range(n):
                for b in range(a + 1, n):
                    d = np.linalg.norm(init[a] - init[b])
                    if d < 0.5:
                        direction = init[a] - init[b]
                        if np.linalg.norm(direction) < 1e-6:
                            direction = np.random.randn(3)
                        direction /= np.linalg.norm(direction)
                        init[a] += direction * (0.5 - d + 0.01)
                        init[b] -= direction * (0.5 - d + 0.01)
        
        # Use L-BFGS-B for optimization
        try:
            result = minimize(
                _morse_energy_and_grad,
                init,
                args=(rho,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            if result.success or result.fun < best_energy:
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_positions = result.x.copy()
        except Exception:
            # Fallback to the provided relax function
            try:
                energy, positions = relax(init, potential, rho)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
    
    # If no optimization succeeded, use the provided relax on the best init
    if best_positions is None:
        # Use the first initialization
        init = inits[0]
        energy, best_positions = relax(init, potential, rho)
        best_energy = energy
    
    return best_positions
# EVOLVE-BLOCK-END
