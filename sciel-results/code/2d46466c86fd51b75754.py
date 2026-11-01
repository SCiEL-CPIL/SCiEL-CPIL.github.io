# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 55

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

def get_mackay_directions():
    """Returns directions for the 43-atom Mackay icosahedron (1 center, 12 shell1, 30 shell2)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len, atol=1e-10):
                edges.append((i, j))
    
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    return verts, midpoints

def make_mackay_55_optimized():
    """
    Create a 55-atom cluster based on Mackay icosahedron geometry.
    Structure: 
    - 1 atom at center
    - 12 atoms at icosahedron vertices (radius r1)
    - 30 atoms at edge midpoints (radius r2)
    - 12 atoms at icosahedron vertices (radius r3) -> This creates a 3rd shell of 12 atoms.
    
    We optimize r1, r2, r3 to minimize energy of this symmetric configuration.
    """
    verts, mids = get_mackay_directions()
    
    def symmetric_energy(params):
        r1, r2, r3 = params
        # Ensure positive radii
        if r1 < 0.5 or r2 < 0.5 or r3 < 0.5:
            return 1e10
        pos = np.vstack([
            np.array([[0.0, 0.0, 0.0]]),
            verts * r1,
            mids * r2,
            verts * r3
        ])
        return lj_energy(pos)

    # Initial guess
    x0 = np.array([1.0, 2.0, 2.8])
    
    # Optimize radii
    res = minimize(symmetric_energy, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    r1, r2, r3 = res.x
    # Clip to reasonable values
    r1 = max(0.5, r1)
    r2 = max(0.5, r2)
    r3 = max(0.5, r3)
    
    pos = np.vstack([
        np.array([[0.0, 0.0, 0.0]]),
        verts * r1,
        mids * r2,
        verts * r3
    ])
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
    base = make_mackay_55_optimized()
    rng = np.random.RandomState(seed)
    pos = base + rng.randn(N, 3) * scale
    return pos

def relax(pos, maxiter=500, gtol=1e-6):
    """Relax using L-BFGS-B."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True, 
                         options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-10})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, lj_energy(pos)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    best_pos = make_mackay_55_optimized()
    best_e = lj_energy(best_pos)
    
    candidates = []
    
    # 1. Optimized Mackay base
    candidates.append(("mackay_opt", make_mackay_55_optimized()))
    
    # 2. Perturbed Optimized Mackay
    for s in range(10):
        candidates.append((f"pert_mackay_{s}", make_perturbed_mackay(s, scale=0.05)))
        
    # 3. Larger perturbations to escape local minima
    for s in range(5):
        candidates.append((f"pert_mackay_large_{s}", make_perturbed_mackay(s, scale=0.2)))
        
    # 4. Cubic lattice
    candidates.append(("cubic", make_cubic_lattice()))
    
    # 5. Random
    for s in range(5):
        candidates.append((f"random_{s}", make_random_cluster(s)))

    # Relax candidates
    for name, pos in candidates:
        if time.time() > deadline:
            break
            
        e_raw = lj_energy(pos)
        
        # Skip if raw energy is already much worse than best? 
        # No, relaxation might improve it. But if it's > best_e + 10, maybe skip?
        # Let's just relax everything in time.
        
        if time.time() < deadline - 10:
            # Use tighter tolerance for the final best, but for search, use faster settings
            pos_relaxed, e_relaxed = relax(pos, maxiter=300, gtol=1e-5)
            
            if e_relaxed < best_e:
                best_e = e_relaxed
                best_pos = pos_relaxed.copy()
        
        if e_raw < best_e:
            best_e = e_raw
            best_pos = pos.copy()

    # Final high-precision polish
    if time.time() < deadline - 5:
        pos_final, e_final = relax(best_pos, maxiter=1000, gtol=1e-8)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final
            
    return best_pos

# EVOLVE-BLOCK-END
