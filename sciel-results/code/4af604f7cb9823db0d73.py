"""
Strategy: Deterministic Basin Hopping with Diverse Crystalline and Random Initializations.
This approach fundamentally differs from the previous stochastic annealing method.
Instead of simulating dynamics to self-assemble, we use a structured search:
1. Generate diverse initial configurations (FCC, BCC, Icosahedron, Random).
2. Relax each to a local minimum using L-BFGS-B.
3. Perform Basin Hopping: kick the structure, relax again, and accept/reject based on Metropolis criterion.
This allows systematic exploration of the energy landscape by jumping between basins of attraction.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 290.0  # 300s limit, keep 10s safety margin

def _morse_energy_and_grad(positions, rho):
    """Compute exact Morse energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)  # Avoid division by zero
    
    # Morse potential: V(r) = exp(rho(1-r)) * (exp(rho(1-r)) - 2)
    # Let x = rho(1-r)
    # V = e^x (e^x - 2) = e^(2x) - 2e^x
    # dV/dr = (2e^(2x) - 2e^x) * dx/dr = 2e^x(e^x - 1) * (-rho)
    # dV/dri = -dV/dr * (ri - rj) / r
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    energy = 0.5 * float(np.sum(ex * (ex - 2.0)))
    
    # Gradient
    # dV/dr = -rho * 2 * ex * (ex - 1)
    dV_dr = -rho * 2.0 * ex * (ex - 1.0)
    
    # Vectorized gradient calculation
    inv_r = np.zeros_like(r)
    mask = r > 1e-12
    inv_r[mask] = 1.0 / r[mask]
    
    # factor[i,j] = dV_dr[i,j] * inv_r[i,j]
    factor = dV_dr * inv_r
    
    # sum_j factor[i,j] -> row_sum (n,)
    row_sum = np.sum(factor, axis=1)
    
    # sum_j factor[i,j] * positions[j] -> factor @ positions (n, 3)
    weighted_pos = factor.dot(positions)
    
    grad = positions * row_sum[:, np.newaxis] - weighted_pos
    
    return energy, grad

def _is_valid(positions, min_dist=0.5):
    """Check if all pairwise distances are >= min_dist."""
    if len(positions) < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    return np.all(r >= min_dist)

def _generate_fcc(n, rng):
    """Generate an FCC-like cluster."""
    # Simple cubic grid, then filter to FCC-like by keeping points where i+j+k is even? 
    # Or just a dense cubic block and relax.
    # For small n, a cube is fine.
    side = int(np.ceil(n ** (1.0/3.0)))
    pts = []
    i, j, k = 0, 0, 0
    while len(pts) < n:
        pts.append([i, j, k])
        k += 1
        if k == side:
            k = 0; j += 1
        if j == side:
            j = 0; i += 1
    pts = np.array(pts[:n], dtype=float)
    # Center
    pts -= pts.mean(axis=0)
    # Add small noise to break symmetry
    pts += rng.normal(0, 0.01, pts.shape)
    return pts

def _generate_bcc(n, rng):
    """Generate a BCC-like cluster (Cubic + Center)."""
    side = int(np.ceil((n/2) ** (1.0/3.0)))
    pts = []
    i, j, k = 0, 0, 0
    while len(pts) < n:
        pts.append([i, j, k])
        if (i+j+k) % 2 == 0: # BCC condition? No, BCC is corners + body center.
             pass
        # Just use a dense grid for simplicity, relax will sort it out
        k += 1
        if k == side:
            k = 0; j += 1
        if j == side:
            j = 0; i += 1
    pts = np.array(pts[:n], dtype=float)
    pts -= pts.mean(axis=0)
    pts += rng.normal(0, 0.01, pts.shape)
    return pts

def _generate_icosahedron(n, rng):
    """Generate an icosahedral-like cluster (Mackay)."""
    # Simple approximation: place atoms on shells of an icosahedron
    # For simplicity, use a random spherical shell distribution
    pts = []
    # Core
    pts.append([0,0,0])
    # Shell 1 (12 atoms)
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_vertices /= np.linalg.norm(ico_vertices[0]) # Normalize
    pts.extend(ico_vertices.tolist())
    
    # Fill rest with random points on sphere
    while len(pts) < n:
        # Random direction
        z = rng.uniform(-1, 1)
        theta = rng.uniform(0, 2*np.pi)
        r = 1.0 + 0.1*rng.random() # Slightly outside shell 1
        x = np.sqrt(1-z*z)*np.cos(theta)*r
        y = np.sqrt(1-z*z)*np.sin(theta)*r
        pts.append([x, y, z])
        
    pts = np.array(pts[:n], dtype=float)
    pts -= pts.mean(axis=0)
    pts += rng.normal(0, 0.01, pts.shape)
    return pts

def _generate_random(n, rng):
    """Generate random points in a sphere."""
    radius = 1.5 * n ** (1.0/3.0)
    pts = np.zeros((n, 3))
    for i in range(n):
        while True:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if i == 0:
                    pts[i] = p
                    break
                dists = np.linalg.norm(pts[:i] - p, axis=1)
                if np.min(dists) > 0.6:
                    pts[i] = p
                    break
    return pts

def _local_minimize(positions, rho):
    """Local minimization using L-BFGS-B."""
    def fun(x):
        x = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(x, rho)
        g = g.flatten()
        return e, g
    
    try:
        res = minimize(fun, positions.flatten(), method='L-BFGS-B', jac=True, 
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
        final_pos = res.x.reshape(-1, 3)
        final_energy = res.fun
        
        if _is_valid(final_pos):
            return final_pos, final_energy
    except Exception:
        pass
    
    return positions, _morse_energy_and_grad(positions, rho)[0]

def _basin_hopping(positions, rho, n_kicks=50, kick_size=0.5, T=1.0, rng=None):
    """
    Perform Basin Hopping.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    current_pos, current_e = _local_minimize(positions, rho)
    best_pos, best_e = current_pos.copy(), current_e
    
    for k in range(n_kicks):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
            
        # Kick
        new_pos = current_pos + rng.normal(0, kick_size, current_pos.shape)
        
        # Relax
        new_pos_relaxed, new_e = _local_minimize(new_pos, rho)
        
        # Metropolis Acceptance
        delta_e = new_e - current_e
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_pos, current_e = new_pos_relaxed, new_e
            
            if current_e < best_e:
                best_pos, best_e = current_pos.copy(), current_e
                
    return best_pos, best_e

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    base_seed = 42
    best_energy = float('inf')
    best_positions = None
    
    # Define initial structure generators
    generators = [
        _generate_fcc,
        _generate_bcc,
        _generate_icosahedron,
        _generate_random
    ]
    
    # We will run multiple independent basin hopping searches from different starts
    # Budget: 300s. Local minimize takes ~0.1-0.5s for n=80. Kick+Relax ~0.5s.
    # So maybe 50-100 kicks per start is feasible if we have few starts.
    # Let's try 4 starts, 20 kicks each.
    
    n_starts = 4
    kicks_per_start = 20
    
    for start_idx in range(n_starts):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
            
        rng = np.random.default_rng(base_seed + start_idx)
        
        # Select generator
        gen_func = generators[start_idx % len(generators)]
        
        try:
            pos0 = gen_func(size, rng)
            
            # Ensure validity before starting BH
            if not _is_valid(pos0):
                # Fallback to random if generator fails validity
                pos0 = _generate_random(size, rng)
                
            # Run Basin Hopping
            pos_bh, e_bh = _basin_hopping(pos0, parameter, n_kicks=kicks_per_start, kick_size=0.4, T=0.5, rng=rng)
            
            if e_bh < best_energy:
                best_energy = e_bh
                best_positions = pos_bh
                
        except Exception:
            # Fallback to simple relax if BH fails
            try:
                pos0 = _generate_random(size, rng)
                energy, positions = relax(pos0, potential, parameter)
                if _is_valid(positions) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass

    # If no valid position found, return a valid random one
    if best_positions is None:
        radius = 1.0 * size ** (1.0 / 3.0) * 1.5
        points = []
        rng = np.random.default_rng(0)
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        best_positions = np.array(points)
    
    return best_positions
# EVOLVE-BLOCK-END
