"""
Morse cluster global minima search using Basin Hopping with L-BFGS-B optimization.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import morse_energy_gradient

# Global time tracking
_START_TIME = time.time()
_TIME_LIMIT = 310.0  # Safety margin: 330s limit, leave 20s buffer

def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _morse_energy(pos, rho):
    """
    Compute the total Morse potential energy.
    E = sum_{i<j} exp(rho(1-r_ij)) * (exp(rho(1-r_ij)) - 2)
    """
    n = len(pos)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(d ** 2, axis=2))
    
    # Avoid division by zero or self-interaction issues by masking diagonal
    np.fill_diagonal(r, np.inf)
    
    # Only consider upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    
    # Morse potential term
    u = rho * (1.0 - r_ij)
    e = np.exp(u)
    energy = np.sum(e * (e - 2.0))
    
    return energy

def _morse_gradient(pos, rho):
    """
    Compute the gradient of the Morse potential energy.
    """
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
        
    d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(d ** 2, axis=2))
    np.fill_diagonal(r, 1.0) # Avoid div by 0 for gradient calculation mask
    
    # Gradient of V(r) = e^u (e^u - 2) where u = rho(1-r)
    # dV/dr = e^u * (2e^u - 2) * (-rho) = -rho * e^u * (2e^u - 2)
    # dV/dpos_i = sum_{j != i} (dV/dr_ij) * (pos_i - pos_j) / r_ij
    
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    
    u = rho * (1.0 - r_ij)
    e = np.exp(u)
    dV_dr = -rho * e * (2.0 * e - 2.0)
    
    # Vector of distances between pairs
    vecs = pos[iu[0]] - pos[iu[1]]
    
    # Normalize vectors
    # Avoid division by zero
    safe_r = np.where(r_ij < 1e-8, 1e-8, r_ij)
    unit_vecs = vecs / safe_r[:, np.newaxis]
    
    # Gradient contribution for each pair
    grad_pairs = dV_dr[:, np.newaxis] * unit_vecs
    
    # Accumulate gradients
    grad = np.zeros_like(pos)
    np.add.at(grad, iu[0], grad_pairs)
    np.add.at(grad, iu[1], -grad_pairs)
    
    return grad

def _local_minimize(pos, rho, max_iter=1000):
    """
    Perform local optimization using L-BFGS-B.
    """
    if _remaining_time() < 1.0:
        return pos, _morse_energy(pos, rho)
        
    result = minimize(
        _morse_energy,
        pos,
        args=(rho,),
        jac=_morse_gradient,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'maxls': 50}
    )
    
    if result.success or result.fun < _morse_energy(pos, rho):
        return result.x, result.fun
    else:
        return pos, _morse_energy(pos, rho)

def _generate_geometric_starts(n, rho, rng):
    """
    Generate high-quality initial configurations based on geometric motifs.
    """
    starts = []
    
    # 1. Icosahedral shells (good for small n)
    if n <= 13:
        # Center + 12 vertices of icosahedron
        t = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
            [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
            [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
        ])
        ico = ico / np.linalg.norm(ico[0])
        
        if n == 1:
            starts.append(np.array([[0.0, 0.0, 0.0]]))
        elif n == 2:
            starts.append(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        elif n == 13:
            starts.append(np.vstack([np.zeros(3), ico]))
        else:
            # Select n-1 points from icosahedron + center
            idx = rng.choice(12, size=n-1, replace=False)
            starts.append(np.vstack([np.zeros(3), ico[idx]]))
            
    # 2. FCC fragment (good for larger n)
    if n > 13:
        # Generate points in a small FCC lattice
        a = 1.0
        points = []
        # Simple cubic grid with FCC offsets
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    points.append([i*a, j*a, k*a])
                    points.append([(i+0.5)*a, (j+0.5)*a, k*a])
                    points.append([(i+0.5)*a, j*a, (k+0.5)*a])
                    points.append([i*a, (j+0.5)*a, (k+0.5)*a])
        
        points = np.array(points)
        # Remove duplicates and filter by distance to center to get compact cluster
        # Just take first n points for simplicity, they will be relaxed
        if len(points) >= n:
            idx = rng.choice(len(points), size=n, replace=False)
            starts.append(points[idx])
            
    # 3. Random compact cluster
    radius = 1.0 * (n ** (1.0/3.0)) * 0.8
    pts = []
    for _ in range(n):
        p = rng.uniform(-radius, radius, 3)
        # Check min distance
        if len(pts) == 0:
            pts.append(p)
        else:
            dists = np.linalg.norm(np.array(pts) - p, axis=1)
            if np.min(dists) > 0.8:
                pts.append(p)
            else:
                # Retry or force add if stuck
                pts.append(p + rng.uniform(-0.1, 0.1, 3))
    starts.append(np.array(pts))
    
    return starts

def construct(instance):
    """
    Find the global minimum for the given Morse cluster instance.
    """
    n = instance['n']
    rho = instance['rho']
    
    # Deadline check
    if _remaining_time() < 5.0:
        # Fallback: simple random start + 1 relaxation
        pos = np.random.randn(n, 3) * 0.5
        pos, _ = _local_minimize(pos, rho, max_iter=50)
        return pos

    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = _generate_geometric_starts(n, rho, rng)
    
    # Add a few random perturbations to the best geometric start
    best_start = starts[0]
    for _ in range(3):
        perturbed = best_start + rng.normal(0, 0.1, (n, 3))
        starts.append(perturbed)
    
    best_pos = None
    best_energy = np.inf
    
    # Basin Hopping Loop
    # Time adaptive: run until time runs out
    current_pos = starts[0].copy()
    
    # First, relax the best geometric start
    current_pos, current_energy = _local_minimize(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    # Iterate through other starts and perform hopping
    hop_temp = 0.1
    hop_step = 0.5
    
    for i in range(1, len(starts)):
        if _remaining_time() < 2.0:
            break
            
        # Relax this start
        pos, energy = _local_minimize(starts[i], rho)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            # Reset hopping from this new best
            current_pos = pos.copy()
            current_energy = energy
        else:
            # Metropolis acceptance for hopping exploration
            # We don't strictly need Metropolis if we are just trying to find global min,
            # but it helps escape local traps.
            dE = energy - best_energy
            accept = (dE < 0) or (np.random.rand() < np.exp(-dE / hop_temp))
            
            if accept:
                current_pos = pos.copy()
                current_energy = energy
            else:
                # Keep current best, but maybe perturb it for next hop
                pass
        
        # Perform a few hops from the current best
        hops_per_start = 5
        for h in range(hops_per_start):
            if _remaining_time() < 2.0:
                break
                
            # Perturb
            noise = rng.normal(0, hop_step, (n, 3))
            new_pos = current_pos + noise
            
            # Relax
            new_pos, new_energy = _local_minimize(new_pos, rho)
            
            # Acceptance
            dE = new_energy - current_energy
            if dE < 0 or np.random.rand() < np.exp(-dE / hop_temp):
                current_pos = new_pos.copy()
                current_energy = new_energy
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
            else:
                # Try a smaller perturbation
                pass
        
        # Decay temperature and step size
        hop_temp *= 0.9
        hop_step *= 0.95
        
    # Final check: ensure minimum distance constraint is roughly satisfied
    # The optimizer should handle this, but let's verify.
    # If atoms are too close, the energy is huge, so it should have moved them.
    
    return best_pos

# EVOLVE-BLOCK-END
