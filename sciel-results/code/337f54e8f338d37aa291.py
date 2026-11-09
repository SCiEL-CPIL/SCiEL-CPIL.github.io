"""Morse cluster global minima search with time-budget guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline: evaluator kills after 330s wall-clock.
# We use a shared deadline across all six construct() calls.
# The first call sets the deadline; subsequent calls check against it.
_DEADLINE = time.time() + 310.0  # 20s safety margin

def _time_left():
    return _DEADLINE - time.time()

def _morse_energy(pos, rho):
    """Compute total Morse energy: sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij))-2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    # Only upper triangle (i<j)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    e = rho * (1.0 - d)
    val = np.exp(e)
    energy = np.sum(val * (val - 2.0))
    return float(energy)

def _morse_energy_gradient(pos, rho):
    """Compute gradient of Morse energy."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    # Force on each atom: -dE/dr * r_hat
    # E_pair = exp(rho(1-r))(exp(rho(1-r))-2)
    # dE_pair/dr = exp(rho(1-r)) * (-rho) * (exp(rho(1-r))-2) + exp(rho(1-r)) * (-rho) * exp(rho(1-r))
    #           = -rho * exp(rho(1-r)) * (exp(rho(1-r))-2) - rho * exp(2*rho(1-r))
    #           = -rho * exp(rho(1-r)) * (exp(rho(1-r))-2+exp(rho(1-r)))
    #           = -rho * exp(rho(1-r)) * (2*exp(rho(1-r))-2)
    #           = -2*rho*exp(rho(1-r))*(exp(rho(1-r))-1)
    e = rho * (1.0 - dist)
    val = np.exp(e)
    # dE_pair/dr = -2*rho*val*(val-1)
    dEdr = -2.0 * rho * val * (val - 1.0)
    # Force = -dE/dr * (r_i - r_j)/|r_i - r_j|
    # For atom i: sum_j -dE_pair/dr_ij * (r_i - r_j)/r_ij
    # = sum_j dEdr_ij * (r_j - r_i)/r_ij  ... wait let me redo
    # F_i = -dE/dr_i = -sum_{j!=i} dE_pair/dr_ij * dr_ij/dr_i
    # dr_ij/dr_i = (r_i - r_j)/r_ij
    # So F_i = -sum_j dEdr_ij * (r_i - r_j)/r_ij
    # Since dEdr is the derivative w.r.t. r (scalar distance), and r_ij = |r_i - r_j|
    # dr_ij/dr_i = (r_i - r_j)/r_ij
    # F_i = -sum_j dEdr_ij * (r_i - r_j)/r_ij
    # Note: dEdr_ij is symmetric in i,j
    # Let's compute: contribution from pair (i,j) to atom i: -dEdr_ij * (r_i - r_j)/r_ij
    # = dEdr_ij * (r_j - r_i)/r_ij
    
    # Build force matrix
    # force[i] = sum_{j != i} dEdr[i,j] * (pos[j] - pos[i]) / dist[i,j]
    # dEdr[i,j] = -2*rho*exp(rho(1-d))*(exp(rho(1-d))-1)
    
    # Avoid division by zero (diagonal is inf, so 1/inf = 0)
    inv_dist = np.zeros_like(dist)
    mask = dist < 1e10
    inv_dist[mask] = 1.0 / dist[mask]
    
    # force[i] = sum_j dEdr[i,j] * inv_dist[i,j] * (pos[j] - pos[i])
    # = sum_j dEdr[i,j] * inv_dist[i,j] * pos[j] - (sum_j dEdr[i,j] * inv_dist[i,j]) * pos[i]
    
    # dEdr is symmetric, diagonal is 0 (since dist=inf -> e=-inf -> val=0 -> dEdr=0)
    # Actually for diagonal: dist=inf, e=rho(1-inf)=-inf, val=0, dEdr=-2*rho*0*(0-1)=0. Good.
    
    coeff = dEdr * inv_dist  # shape (n,n), symmetric, zero diagonal
    # force[i] = sum_j coeff[i,j] * pos[j] - (sum_j coeff[i,j]) * pos[i]
    force = coeff @ pos - coeff.sum(axis=1, keepdims=True) * pos
    return force

def _generate_icosahedral_shell(n_atoms, rho):
    """Generate atoms on an icosahedral-like shell structure."""
    # Use icosahedral coordinates as base
    # Icosahedron vertices (unit):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # For n atoms, distribute on sphere using Fibonacci spiral
    rng = np.random.default_rng(42)
    points = []
    
    if n_atoms <= 12:
        # Use subset of icosahedron vertices
        idx = rng.choice(12, size=min(n_atoms, 12), replace=False)
        points = verts[idx].copy()
        # If fewer than 12, just use first n_atoms
    else:
        # Fibonacci sphere for remaining
        n_fib = n_atoms
        golden = (1.0 + np.sqrt(5.0)) / 2.0
        for i in range(n_fib):
            z = 1.0 - 2.0 * (i + 0.5) / n_fib
            r = np.sqrt(max(0.0, 1.0 - z*z))
            theta = 2.0 * np.pi * i / golden
            points.append([r * np.cos(theta), r * np.sin(theta), z])
        points = np.array(points)
    
    # Scale to appropriate radius
    # For Morse with equilibrium r=1, a compact cluster of n atoms
    # has roughly radius ~ n^(1/3) * 1.0
    radius = max(1.5, n_atoms ** (1.0/3.0) * 0.8)
    points *= radius
    
    return points

def _generate_bcc(n_atoms, rho):
    """Generate a body-centered cubic-like arrangement."""
    rng = np.random.default_rng(123)
    points = []
    
    # BCC unit cell: corners + center
    # For n atoms, create a grid
    # Simple approach: place in a cubic grid with BCC-like density
    side = int(np.ceil(n_atoms ** (1.0/3.0)))
    
    # Generate BCC lattice points
    lattice = []
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                # Corner
                lattice.append([2*ix, 2*iy, 2*iz])
                # Body center
                lattice.append([2*ix+1, 2*iy+1, 2*iz+1])
    
    lattice = np.array(lattice, dtype=float)
    
    # Select n_atoms closest to center
    center = lattice.mean(axis=0)
    dists = np.linalg.norm(lattice - center, axis=1)
    idx = np.argsort(dists)[:n_atoms]
    points = lattice[idx].copy()
    
    # Center and scale
    points -= points.mean(axis=0)
    # Scale so nearest neighbor distance ~ 1
    if len(points) > 1:
        # Compute nearest neighbor distance
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        d, _ = tree.query(points, k=2)
        nn_dist = d[:, 1].mean()
        if nn_dist > 0:
            points /= nn_dist
    
    return points

def _generate_compact_random(n_atoms, rho, seed=0):
    """Generate a compact random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    spacing = 0.55  # Slightly above 0.5 minimum
    radius = spacing * n_atoms ** (1.0/3.0) * 1.2
    
    points = []
    max_attempts = n_atoms * 200
    attempts = 0
    while len(points) < n_atoms and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= spacing for q in points):
            points.append(p)
    
    # If we couldn't place all atoms, add remaining at random positions (will be fixed by relaxation)
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)

def _local_relax(positions, potential, rho, max_iter=500):
    """Local minimization using L-BFGS-B with gradient."""
    from scipy.optimize import minimize
    
    def obj(x):
        return _morse_energy(x.reshape(-1, 3), rho)
    
    def grad(x):
        return _morse_energy_gradient(x.reshape(-1, 3), rho).reshape(-1)
    
    x0 = positions.reshape(-1)
    result = minimize(obj, x0, jac=grad, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
    
    final_pos = result.x.reshape(-1, 3)
    final_energy = _morse_energy(final_pos, rho)
    
    return final_energy, final_pos

def _refine_with_perturbation(positions, rho, n_perturb=3, perturb_scale=0.1):
    """Try small perturbations of the current best to escape local minima."""
    best_energy, best_pos = _local_relax(positions, 'morse', rho, max_iter=300)
    
    for _ in range(n_perturb):
        if _time_left() < 10:
            break
        rng = np.random.default_rng()
        perturbed = best_pos + rng.normal(0, perturb_scale, size=best_pos.shape)
        # Ensure minimum distance
        e, p = _local_relax(perturbed, 'morse', rho, max_iter=200)
        if e < best_energy:
            best_energy, best_pos = e, p
    
    return best_energy, best_pos

def construct(instance):
    """Find low-energy Morse cluster arrangement."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = _DEADLINE
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: Generate diverse initial configurations
    initial_configs = []
    
    # Icosahedral shell
    try:
        cfg = _generate_icosahedral_shell(n, rho)
        if len(cfg) == n:
            initial_configs.append(('ico', cfg))
    except:
        pass
    
    # BCC-like
    try:
        cfg = _generate_bcc(n, rho)
        if len(cfg) == n:
            initial_configs.append(('bcc', cfg))
    except:
        pass
    
    # Multiple random compact configs with different seeds
    for seed in [0, 1, 2, 3, 4, 5]:
        try:
            cfg = _generate_compact_random(n, rho, seed=seed)
            if len(cfg) == n:
                initial_configs.append((f'rand_{seed}', cfg))
        except:
            pass
    
    # Random sphere (uniform in ball)
    for seed in [10, 11]:
        try:
            rng = np.random.default_rng(seed)
            radius = n ** (1.0/3.0) * 0.9
            cfg = rng.uniform(-radius, radius, (n, 3))
            # Filter to sphere
            norms = np.linalg.norm(cfg, axis=1)
            cfg = cfg[norms <= radius]
            while len(cfg) < n:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius:
                    cfg = np.vstack([cfg, p])
            cfg = cfg[:n]
            initial_configs.append((f'sphere_{seed}', cfg))
        except:
            pass
    
    # Relax each initial configuration
    for name, cfg in initial_configs:
        if _time_left() < 15:
            break
        try:
            energy, positions = _local_relax(cfg, 'morse', rho, max_iter=400)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except:
            pass
    
    # If we have a good result, try to improve with perturbation
    if best_positions is not None and _time_left() > 20:
        try:
            energy, positions = _refine_with_perturbation(best_positions, rho, n_perturb=5, perturb_scale=0.08)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            pass
    
    # If still time, try more diverse starts
    if best_positions is not None and _time_left() > 30:
        for seed in range(100, 120):
            if _time_left() < 15:
                break
            try:
                cfg = _generate_compact_random(n, rho, seed=seed)
                energy, positions = _local_relax(cfg, 'morse', rho, max_iter=300)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except:
                pass
    
    # Final fallback: if no valid result, use a simple linear arrangement
    if best_positions is None:
        # Place atoms in a line with spacing 1.0
        best_positions = np.zeros((n, 3))
        for i in range(n):
            best_positions[i, 0] = i * 1.0
    
    return best_positions
# EVOLVE-BLOCK-END
