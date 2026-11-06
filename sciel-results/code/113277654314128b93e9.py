# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy for a set of positions."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = dists[i, j]
            if r < 1e-8:
                continue
            if r < a:
                energy += A * (r**(-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
    return energy

def _generate_bcc_cluster(n):
    """Generate a BCC-like cluster structure."""
    points = []
    # BCC lattice points
    lattice_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                lattice_points.append(np.array([i, j, k]))
                lattice_points.append(np.array([i + 0.5, j + 0.5, k + 0.5]))
    
    # Sort by distance from center and take n closest
    lattice_points = np.array(lattice_points)
    centers = np.linalg.norm(lattice_points, axis=1)
    sorted_idx = np.argsort(centers)
    points = lattice_points[sorted_idx[:n]]
    
    # Center the cluster
    points = points - np.mean(points, axis=0)
    return points

def _generate_fcc_cluster(n):
    """Generate an FCC-like cluster structure."""
    points = []
    # FCC lattice points
    lattice_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                lattice_points.append(np.array([i, j, k]))
                lattice_points.append(np.array([i, j + 0.5, k + 0.5]))
                lattice_points.append(np.array([i + 0.5, j, k + 0.5]))
                lattice_points.append(np.array([i + 0.5, j + 0.5, k]))
    
    lattice_points = np.array(lattice_points)
    centers = np.linalg.norm(lattice_points, axis=1)
    sorted_idx = np.argsort(centers)
    points = lattice_points[sorted_idx[:n]]
    
    points = points - np.mean(points, axis=0)
    return points

def _generate_icosahedral_cluster(n):
    """Generate an icosahedral-like cluster structure with proper polytetrahedral ordering."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Build layers: center, inner icosahedron, outer icosahedron
    points = [np.zeros(3)]
    
    # Inner shell (12 atoms) - icosahedral coordination
    if n > 1:
        inner_r = 1.05
        points.extend([v * inner_r for v in verts])
    
    # Outer shell - use icosahedral symmetry for remaining atoms
    remaining = n - len(points)
    if remaining > 0:
        outer_r = 2.1
        # Use all 12 vertices at outer radius
        outer_verts = [v * outer_r for v in verts]
        
        # If we need more than 12, use a second shell at larger radius
        # with vertices of a dodecahedron-like arrangement
        if remaining > 12:
            # Add vertices at midpoints of icosahedron edges (30 edge midpoints)
            edge_midpoints = []
            for i in range(12):
                for j in range(i+1, 12):
                    mid = (verts[i] + verts[j]) / 2
                    mid = mid / np.linalg.norm(mid)
                    edge_midpoints.append(mid * (outer_r * 1.15))
            edge_midpoints = np.array(edge_midpoints)
            # Sort by distance and take what we need
            idx = np.argsort(np.linalg.norm(edge_midpoints, axis=1))
            for idx_i in idx[:min(remaining - 12, len(idx))]:
                points.append(edge_midpoints[idx_i])
        
        # Fill remaining with outer icosahedral vertices
        for v in outer_verts[:max(0, remaining - (len(points) - 13))]:
            points.append(v)
    
    # Trim to n
    points = np.array(points[:n])
    points = points - np.mean(points, axis=0)
    return points

def _generate_random_compact(n, rng):
    """Generate a random compact cluster."""
    radius = 1.1 * n**(1.0/3.0) * 0.75
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
            points.append(p)
    return np.array(points)

def _local_relax(positions, potential, max_steps=300, tol=1e-7):
    """Perform local relaxation using gradient descent with L-BFGS-like line search."""
    pos = positions.copy()
    energy, grad = dzugutov_energy_gradient(pos, potential)
    for step in range(max_steps):
        if np.max(np.abs(grad)) < tol:
            break
        # Adaptive step size with backtracking
        lr = 0.05
        for _ in range(15):
            new_pos = pos - lr * grad
            new_energy = _dzugutov_energy(new_pos)
            if new_energy < energy:
                break
            lr *= 0.5
        pos = new_pos
        energy = new_energy
        grad = dzugutov_energy_gradient(pos, potential)[1]
    return _dzugutov_energy(pos), pos

def _simulated_annealing(positions, potential, n_restarts=3, max_steps=500, T_start=1.0, T_end=0.01):
    """Perform simulated annealing from multiple restarts."""
    best_energy = np.inf
    best_pos = positions.copy()
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        pos = positions.copy()
        if restart > 0:
            # Add perturbation
            pos = pos + rng.normal(0, 0.3, pos.shape)
        
        T = T_start
        for step in range(max_steps):
            T = T_start * (T_end / T_start) ** (step / max_steps)
            energy, grad = dzugutov_energy_gradient(pos, potential)
            
            # Propose move
            delta = rng.normal(0, 0.1, pos.shape)
            new_pos = pos + delta
            new_energy = _dzugutov_energy(new_pos)
            
            # Acceptance criterion
            if new_energy < energy or rng.uniform() < np.exp(-(new_energy - energy) / T):
                pos = new_pos
        
        if _dzugutov_energy(pos) < best_energy:
            best_energy = _dzugutov_energy(pos)
            best_pos = pos.copy()
    
    return best_energy, best_pos

def construct(instance):
    """Find low-energy Dzugutov cluster structures."""
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 295.0  # 5 second safety margin before 300s limit
    
    # Generate diverse initial structures
    initial_structures = []
    
    # BCC
    try:
        bcc = _generate_bcc_cluster(n)
        initial_structures.append(bcc)
    except:
        pass
    
    # FCC
    try:
        fcc = _generate_fcc_cluster(n)
        initial_structures.append(fcc)
    except:
        pass
    
    # Icosahedral (most important for Dzugutov)
    try:
        ico = _generate_icosahedral_cluster(n)
        initial_structures.append(ico)
    except:
        pass
    
    # Random compact structures
    rng = np.random.default_rng(n)
    for _ in range(3):
        try:
            rand = _generate_random_compact(n, rng)
            initial_structures.append(rand)
        except:
            pass
    
    # Relax each structure and keep the best
    best_energy = np.inf
    best_positions = None
    
    for i, init_pos in enumerate(initial_structures):
        if time.time() > deadline - 10:  # Safety margin
            break
        
        # First do local relaxation using the provided relax function
        try:
            energy, relaxed_pos = relax(init_pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
        except:
            pass
        
        # Then try simulated annealing from this structure
        if time.time() > deadline - 30:
            break
        
        try:
            sa_energy, sa_pos = _simulated_annealing(
                relaxed_pos if 'relaxed_pos' in locals() else init_pos,
                potential,
                n_restarts=2,
                max_steps=150,
                T_start=0.5,
                T_end=0.05
            )
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_positions = sa_pos.copy()
        except:
            pass
    
    # Final high-precision polish pass
    if best_positions is not None and time.time() < deadline - 5:
        try:
            final_energy, final_pos = _local_relax(best_positions, potential, max_steps=200, tol=1e-8)
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos
        except:
            pass
    
    # Final check
    if best_positions is None:
        # Fallback: return first relaxed structure
        energy, positions = relax(initial_structures[0], potential, parameter)
        return positions
    
    return best_positions
# EVOLVE-BLOCK-END
