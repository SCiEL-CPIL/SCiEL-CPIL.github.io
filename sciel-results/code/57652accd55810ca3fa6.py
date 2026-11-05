"""Dzugutov cluster global minima search using polytetrahedral/icosahedral initial guesses."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A_POT = 5.82
B_POT = 1.28
a_POT = 1.87
b_POT = 1.94
c_POT = 1.10
d_POT = 0.27
m_POT = 16.0

# Global deadline guard
_START_TIME = time.time()
_TOTAL_BUDGET = 330.0
_SAFETY_MARGIN = 10.0


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-12:
                continue
            if r < a_POT:
                energy += A_POT * (r ** (-m_POT) - B_POT) * np.exp(c_POT / (r - a_POT))
            elif r < b_POT:
                energy += B_POT * np.exp(d_POT / (r - b_POT))
    return energy


def _refine(positions, potential, parameter, max_iter=2000, tol=1e-10):
    """Gradient-based refinement using L-BFGS-B."""
    from scipy.optimize import minimize
    
    def objective(x):
        pos = x.reshape(-1, 3)
        return _dzugutov_energy(pos)
    
    def gradient(x):
        pos = x.reshape(-1, 3)
        grad = dzugutov_energy_gradient(pos)
        return grad.flatten()
    
    x0 = positions.flatten()
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
    return result.fun, result.x.reshape(-1, 3)


def _icosahedron_vertices(radius=1.0):
    """Generate 12 vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    # Normalize to unit radius
    verts = verts / np.linalg.norm(verts[0]) * radius
    return verts


def _build_icosahedral_cluster(n_atoms, radius=1.1, rng=None):
    """Build a cluster starting from an icosahedral core and add atoms."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Start with icosahedral core (13 atoms: 1 center + 12 vertices)
    ico = _icosahedron_vertices(radius)
    core = np.vstack([np.zeros(3), ico])
    positions = [core]
    current_n = 13
    
    # Add more atoms by placing them near existing atoms
    while current_n < n_atoms:
        # Pick a random existing atom and add a new one nearby
        base_idx = rng.integers(0, current_n)
        base_pos = positions[0][base_idx]
        
        # Try to place new atom at random direction from base
        for attempt in range(50):
            direction = rng.normal(0, 1, 3)
            direction /= np.linalg.norm(direction)
            new_pos = base_pos + direction * 1.1
            
            # Check minimum distance
            valid = True
            for p in positions[0]:
                if np.linalg.norm(new_pos - p) < 0.5:
                    valid = False
                    break
            
            if valid:
                positions[0] = np.vstack([positions[0], new_pos])
                current_n += 1
                break
    
    return positions[0][:n_atoms]


def _build_polytetrahedral(n_atoms, rng=None):
    """Build a polytetrahedral cluster with icosahedral symmetry."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Start with icosahedron
    ico = _icosahedron_vertices(1.1)
    center = np.zeros(3)
    positions = [center] + list(ico)
    
    # Add atoms in layers
    for target_n in range(13, n_atoms + 1):
        # Find position that maximizes distance to existing atoms
        best_pos = None
        best_score = -np.inf
        
        for _ in range(100):
            # Random point in a shell
            r = 1.1 * (1 + 0.1 * rng.random())
            theta = rng.random() * 2 * np.pi
            phi = np.arccos(2 * rng.random() - 1)
            new_pos = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])
            
            # Check minimum distance
            min_dist = min(np.linalg.norm(new_pos - p) for p in positions)
            if min_dist < 0.5:
                continue
            
            # Score based on local environment
            score = min_dist
            if score > best_score:
                best_score = score
                best_pos = new_pos
        
        if best_pos is not None:
            positions.append(best_pos)
    
    return np.array(positions[:n_atoms])


def _build_random_compact(n_atoms, rng=None):
    """Build a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng()
    
    radius = 1.1 * n_atoms ** (1.0 / 3.0) * 0.7
    points = []
    for _ in range(n_atoms):
        for attempt in range(1000):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
                break
    
    if len(points) < n_atoms:
        # Fallback: add more points
        while len(points) < n_atoms:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    
    return np.array(points[:n_atoms])


def construct(instance):
    """Find low-energy structures for Dzugutov clusters using multiple initial guesses."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = _START_TIME + _TOTAL_BUDGET - _SAFETY_MARGIN
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial guesses
    rng = np.random.default_rng(size)
    
    # 1. Icosahedral-based clusters
    for i in range(3):
        if time.time() > deadline:
            break
        try:
            pos = _build_icosahedral_cluster(size, radius=1.0 + 0.1 * i, rng=rng)
            energy, refined_pos = _refine(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = refined_pos
        except Exception:
            pass
    
    # 2. Polytetrahedral clusters
    for i in range(2):
        if time.time() > deadline:
            break
        try:
            pos = _build_polytetrahedral(size, rng=rng)
            energy, refined_pos = _refine(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = refined_pos
        except Exception:
            pass
    
    # 3. Random compact clusters
    for i in range(4):
        if time.time() > deadline:
            break
        try:
            pos = _build_random_compact(size, rng=rng)
            energy, refined_pos = _refine(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = refined_pos
        except Exception:
            pass
    
    # 4. Local search around best solution
    if best_positions is not None and time.time() < deadline:
        for scale in [0.1, 0.05, 0.02]:
            if time.time() > deadline:
                break
            for _ in range(5):
                perturbed = best_positions + rng.normal(0, scale, best_positions.shape)
                try:
                    energy, refined_pos = _refine(perturbed, potential, parameter)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = refined_pos
                except Exception:
                    pass
    
    # Final safety: if no valid result, use simple random
    if best_positions is None:
        best_positions = _build_random_compact(size, rng=rng)
    
    return best_positions
# EVOLVE-BLOCK-END
