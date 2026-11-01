"""Lennard-Jones 55-atom cluster: global optimization with Basin-Hopping."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE = 140.0  # Safety margin before 150s kill

def lj_energy(pos):
    """Compute total LJ energy for N atoms (vectorized)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def lj_energy_grad(pos):
    """Compute LJ energy and gradient (vectorized)."""
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
    
    # Total: 1 (center) + 12 (shell 1) + 30 (shell 2) + 12 (shell 3) = 55
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

def relax_positions(pos, maxiter=200, gtol=1e-5):
    """Relax atomic positions using L-BFGS-B."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True, 
                         options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-8, 'maxls': 50})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, lj_energy(pos)

def basin_hopping_step(pos, scale, rng, maxiter_relax=150):
    """
    One Basin-Hopping step: rigid-body displacement + L-BFGS relaxation.
    Returns new position and energy.
    """
    # Rigid-body displacement: translate entire cluster
    displacement = rng.randn(3) * scale
    perturbed = pos + displacement
    
    # Also add small per-atom perturbations to break symmetry
    small_pert = rng.randn(*pos.shape) * (scale * 0.3)
    perturbed = perturbed + small_pert
    
    # Relax to nearest local minimum
    relaxed, e = relax_positions(perturbed, maxiter=maxiter_relax, gtol=1e-5)
    return relaxed, e

def optimize_radii_and_relax(radii_init, maxiter_nm=50, maxiter_lbfgs=120):
    """
    Optimize shell radii using Nelder-Mead with L-BFGS relaxation.
    """
    def objective(radii):
        radii = np.maximum(radii, 0.5)
        pos = make_mackay_from_radii(radii)
        pos_relaxed, e = relax_positions(pos, maxiter=maxiter_lbfgs, gtol=1e-5)
        return e
    
    try:
        res = minimize(objective, radii_init, method='Nelder-Mead', 
                       options={'maxiter': maxiter_nm, 'xatol': 1e-4, 'fatol': 1e-4})
        best_radii = res.x
        best_pos = make_mackay_from_radii(best_radii)
        best_pos, best_e = relax_positions(best_pos, maxiter=200, gtol=1e-6)
        return best_pos, best_e, best_radii
    except Exception:
        return None, np.inf, radii_init

def construct():
    start_time = time.time()
    
    def remaining():
        return DEADLINE - (time.time() - start_time)
    
    # Phase 1: Optimize radii from multiple starting points
    radius_candidates = [
        np.array([1.0, 1.8, 2.6]),
        np.array([1.1, 1.9, 2.7]),
        np.array([0.9, 1.7, 2.5]),
        np.array([1.0, 2.0, 2.8]),
        np.array([1.05, 1.85, 2.65]),
        np.array([1.0, 1.75, 2.55]),
    ]
    
    best_pos = None
    best_e = np.inf
    
    for radii_init in radius_candidates:
        if remaining() < 20:
            break
        try:
            pos, e, radii = optimize_radii_and_relax(radii_init, maxiter_nm=40, maxiter_lbfgs=100)
            if pos is not None and e < best_e:
                best_e = e
                best_pos = pos.copy()
        except Exception:
            continue
    
    # Fallback if no valid result
    if best_pos is None:
        best_pos = make_mackay_from_radii(np.array([1.0, 1.8, 2.6]))
        best_e = lj_energy(best_pos)
        best_pos, best_e = relax_positions(best_pos, maxiter=200, gtol=1e-6)
    
    # Phase 2: Basin-Hopping global optimization
    rng = np.random.RandomState(42)
    bh_steps = 30
    current_pos = best_pos.copy()
    current_e = best_e
    
    # Adaptive step size
    step_scales = [0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    
    for step in range(bh_steps):
        if remaining() < 10:
            break
        
        # Choose step scale: start small, increase periodically
        scale_idx = min(step // 3, len(step_scales) - 1)
        scale = step_scales[scale_idx]
        
        try:
            new_pos, new_e = basin_hopping_step(current_pos, scale, rng, maxiter_relax=120)
            
            # Metropolis acceptance: always accept if lower, accept with probability exp(-delta/kT) if higher
            delta = new_e - current_e
            if delta < 0:
                # Accept (lower energy)
                current_pos = new_pos
                current_e = new_e
                if current_e < best_e:
                    best_e = current_e
                    best_pos = current_pos.copy()
            else:
                # Metropolis criterion with temperature ~0.1
                kT = 0.1
                if rng.rand() < np.exp(-delta / kT):
                    current_pos = new_pos
                    current_e = new_e
        except Exception:
            continue
    
    # Phase 3: Final high-precision polish
    if remaining() > 5:
        try:
            final_pos, final_e = relax_positions(best_pos, maxiter=500, gtol=1e-8)
            if final_e < best_e:
                best_e = final_e
                best_pos = final_pos
        except Exception:
            pass
    
    return best_pos
