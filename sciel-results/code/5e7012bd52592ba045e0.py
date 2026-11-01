"""Lennard-Jones 55-atom cluster: coarse-grained radius optimization + multi-start polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
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
    """Get the 55 unit direction vectors for a Mackay icosahedron."""
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
    
    # Return directions and shell assignments
    directions = np.zeros((55, 3))
    shell_ids = np.zeros(55, dtype=int)
    
    # Atom 0: center
    directions[0] = [0.0, 0.0, 0.0]
    shell_ids[0] = 0
    
    # Atoms 1-12: Shell 1 (icosahedron vertices)
    for i in range(12):
        directions[1+i] = verts[i]
        shell_ids[1+i] = 1
    
    # Atoms 13-42: Shell 2 (edge midpoints)
    for i in range(30):
        directions[13+i] = midpoints[i]
        shell_ids[13+i] = 2
    
    # Atoms 43-54: Shell 3 (icosahedron vertices, outer)
    for i in range(12):
        directions[43+i] = verts[i]
        shell_ids[43+i] = 3
    
    return directions, shell_ids

def make_mackay_from_radii(radii):
    """Generate Mackay icosahedron positions given shell radii [r1, r2, r3]."""
    directions, shell_ids = get_mackay_directions()
    
    pos = np.zeros((55, 3))
    for i in range(1, 55):
        shell = shell_ids[i]
        if shell == 1:
            pos[i] = directions[i] * radii[0]
        elif shell == 2:
            pos[i] = directions[i] * radii[1]
        elif shell == 3:
            pos[i] = directions[i] * radii[2]
    
    return pos

def relax_positions(pos, maxiter=300, gtol=1e-6):
    """Relax atomic positions using L-BFGS-B."""
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

def optimize_radii_with_relaxation(radii_init, deadline, maxiter_nm=80, maxiter_lbfgs=150):
    """
    Optimize shell radii using Nelder-Mead where each evaluation
    includes a full L-BFGS relaxation. This gives a smooth-ish landscape
    in the 3D radius space.
    """
    def objective(radii):
        if time.time() > deadline:
            return 1e10
        radii = np.maximum(radii, 0.3)
        pos = make_mackay_from_radii(radii)
        pos_relaxed, e = relax_positions(pos, maxiter=maxiter_lbfgs, gtol=1e-7)
        return e
    
    try:
        res = minimize(objective, radii_init, method='Nelder-Mead', 
                       options={'maxiter': maxiter_nm, 'xatol': 1e-5, 'fatol': 1e-6,
                                'adaptive': True})
        best_radii = res.x
        best_pos = make_mackay_from_radii(best_radii)
        best_pos, best_e = relax_positions(best_pos, maxiter=maxiter_lbfgs, gtol=1e-7)
        return best_pos, best_e, best_radii
    except Exception:
        pos = make_mackay_from_radii(radii_init)
        pos, e = relax_positions(pos, maxiter=maxiter_lbfgs, gtol=1e-7)
        return pos, e, radii_init

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s kill
    
    # Phase 1: Coarse-grained radius optimization
    # Multiple initial radius guesses to cover different basins in radius space
    radius_inits = [
        np.array([1.0, 1.8, 2.6]),
        np.array([1.1, 1.9, 2.7]),
        np.array([0.95, 1.75, 2.55]),
        np.array([1.05, 1.85, 2.65]),
        np.array([1.0, 2.0, 2.8]),
        np.array([0.9, 1.6, 2.4]),
    ]
    
    best_pos = None
    best_e = np.inf
    
    for radii_init in radius_inits:
        if time.time() > deadline - 20:
            break
        try:
            pos, e, radii = optimize_radii_with_relaxation(radii_init, deadline, maxiter_nm=60, maxiter_lbfgs=120)
            if e < best_e:
                best_e = e
                best_pos = pos.copy()
        except Exception:
            continue
    
    # Fallback: if radius optimization failed, use direct relaxation
    if best_pos is None:
        base_pos = make_mackay_from_radii(np.array([1.0, 1.8, 2.6]))
        best_pos, best_e = relax_positions(base_pos, maxiter=300, gtol=1e-7)
    
    # Phase 2: Multi-start perturbation from the best relaxed Mackay
    rng = np.random.RandomState(42)
    noise_scales = [0.02, 0.04, 0.06, 0.10, 0.15, 0.20, 0.30, 0.45]
    
    for scale in noise_scales:
        if time.time() > deadline - 15:
            break
        n_perturb = 3 if scale < 0.1 else 2
        for trial in range(n_perturb):
            if time.time() > deadline - 15:
                break
            noise = rng.randn(N, 3) * scale
            noise[0] = 0.0
            perturbed = best_pos + noise
            
            # Quick energy check
            e_perturbed = lj_energy(perturbed)
            if e_perturbed > best_e + 30:
                continue
            
            pos_relaxed, e_relaxed = relax_positions(perturbed, maxiter=200, gtol=1e-7)
            if e_relaxed < best_e:
                best_e = e_relaxed
                best_pos = pos_relaxed.copy()
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5:
        pos_final, e_final = relax_positions(best_pos, maxiter=500, gtol=1e-10)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final
    
    return best_pos

# EVOLVE-BLOCK-END
