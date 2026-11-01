"""Lennard-Jones 55-atom cluster: search approach with Mackay icosahedron + relaxation."""
import numpy as np
import time
from itertools import combinations

N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def lj_energy(pos):
    """Compute total LJ energy for N atoms."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def lj_energy_grad(pos):
    """Compute LJ energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def make_mackay_icosahedron():
    """Generate a Mackay icosahedral cluster with 55 atoms.
    
    Structure: 1 (center) + 12 (icosahedron shell 1) + 30 (edge-midpoint shell 2) + 12 (vertex directions, outer shell)
    For N=55: 1+12+30+12 = 55.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find edges of icosahedron
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len, atol=1e-10):
                edges.append((i, j))
    
    # 30 edge midpoints (normalized)
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    # Radii: for LJ, optimal nearest-neighbor distance is 2^(1/6) ≈ 1.12246
    # Use radii that give reasonable inter-shell distances
    r1 = 1.0
    r2 = 2.0  
    r3 = 2.8  # Slightly less than 3 to avoid being too far
    
    inner = np.array([[0.0, 0.0, 0.0]])
    shell1 = verts * r1
    shell2 = midpoints * r2
    shell3 = verts * r3  # 12 atoms at icosahedron directions at outer radius
    
    pos = np.vstack([inner, shell1, shell2, shell3])
    return pos

def make_cubic_lattice():
    """Generate a cubic lattice fragment."""
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    return np.array(points[:N], dtype=float) * 1.12

def make_random_cluster(seed):
    """Generate a random cluster."""
    rng = np.random.RandomState(seed)
    pos = rng.randn(N, 3) * 2.0
    return pos

def make_perturbed_mackay(seed, scale=0.1):
    """Generate a perturbed Mackay icosahedron."""
    base = make_mackay_icosahedron()
    rng = np.random.RandomState(seed)
    pos = base + rng.randn(N, 3) * scale
    return pos

def relax(pos, maxiter=500):
    """Relax using L-BFGS-B with tight tolerance."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True, 
                         options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, lj_energy(pos)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    best_pos = make_mackay_icosahedron()
    best_e = lj_energy(best_pos)
    
    # Generate candidate structures
    candidates = []
    
    # 1. Mackay icosahedron base (most promising)
    candidates.append(("mackay_base", make_mackay_icosahedron()))
    
    # 2. Perturbed Mackay with small perturbations (multiple seeds)
    for s in range(15):
        candidates.append((f"perturbed_mackay_small_{s}", make_perturbed_mackay(s, scale=0.05)))
    
    # 3. Perturbed Mackay with medium perturbations
    for s in range(10):
        candidates.append((f"perturbed_mackay_med_{s}", make_perturbed_mackay(s, scale=0.15)))
    
    # 4. Cubic lattice
    candidates.append(("cubic", make_cubic_lattice()))
    
    # 5. Random clusters
    for s in range(5):
        candidates.append((f"random_{s}", make_random_cluster(s)))
    
    # Evaluate and relax each candidate within time budget
    # Priority: relax Mackay-based candidates first (most likely to converge to good minimum)
    for name, pos in candidates:
        if time.time() > deadline:
            break
        
        # Skip if already better than current best
        e_raw = lj_energy(pos)
        
        # Try to relax - use more iterations for promising candidates
        if time.time() < deadline - 10:  # Leave 10s safety margin
            # Use more iterations for Mackay-based candidates
            if "mackay" in name:
                pos_relaxed, e_relaxed = relax(pos, maxiter=500)
            else:
                pos_relaxed, e_relaxed = relax(pos, maxiter=300)
            
            if e_relaxed < best_e:
                best_e = e_relaxed
                best_pos = pos_relaxed.copy()
        
        if e_raw < best_e:
            best_e = e_raw
            best_pos = pos.copy()
    
    # Final polish: one more relaxation on the best candidate
    if time.time() < deadline - 5:
        pos_final, e_final = relax(best_pos, maxiter=300)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final
    
    return best_pos

# EVOLVE-BLOCK-END
