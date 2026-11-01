"""Lennard-Jones 55-atom cluster: global optimization of Mackay geometry."""
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
    
    # 12 outer shell vertices (same directions as inner shell)
    # Total: 1 (center) + 12 (shell 1) + 30 (shell 2) + 12 (shell 3) = 55
    
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
    # Atom 0 is at center (radius 0)
    for i in range(1, 55):
        shell = shell_ids[i]
        if shell == 1:
            pos[i] = directions[i] * radii[0]
        elif shell == 2:
            pos[i] = directions[i] * radii[1]
        elif shell == 3:
            pos[i] = directions[i] * radii[2]
    
    return pos

def relax_positions(pos, maxiter=300):
    """Relax atomic positions using L-BFGS-B."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True, 
                         options={'maxiter': maxiter, 'gtol': 1e-6, 'ftol': 1e-8})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, lj_energy(pos)

def optimize_radii_and_relax(radii_init, maxiter_nm=100, maxiter_lbfgs=200):
    """
    Stage 1: Optimize shell radii using Nelder-Mead.
    Stage 2: For each radius configuration, relax positions with L-BFGS.
    """
    def objective(radii):
        # Ensure radii are positive
        radii = np.maximum(radii, 0.5)
        pos = make_mackay_from_radii(radii)
        # Relax positions
        pos_relaxed, e = relax_positions(pos, maxiter=maxiter_lbfgs)
        return e
    
    # Optimize radii
    res = minimize(objective, radii_init, method='Nelder-Mead', 
                   options={'maxiter': maxiter_nm, 'xatol': 1e-4, 'fatol': 1e-4})
    
    best_radii = res.x
    best_pos = make_mackay_from_radii(best_radii)
    best_pos, best_e = relax_positions(best_pos, maxiter=maxiter_lbfgs)
    
    return best_pos, best_e, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Initial guess for radii (based on typical LJ cluster dimensions)
    # Nearest neighbor distance ~ 2^(1/6) ≈ 1.122
    # Shell 1 radius ~ 1.0, Shell 2 ~ 1.8, Shell 3 ~ 2.6
    initial_radii = np.array([1.0, 1.8, 2.6])
    
    best_pos = make_mackay_from_radii(initial_radii)
    best_e = lj_energy(best_pos)
    
    # Try multiple initial radius configurations
    radius_candidates = [
        np.array([1.0, 1.8, 2.6]),
        np.array([1.1, 1.9, 2.7]),
        np.array([0.9, 1.7, 2.5]),
        np.array([1.0, 2.0, 2.8]),  # Original program's guess
        np.array([1.05, 1.85, 2.65]),
    ]
    
    for radii_init in radius_candidates:
        if time.time() > deadline - 10:
            break
        
        try:
            pos, e, radii = optimize_radii_and_relax(radii_init, maxiter_nm=50, maxiter_lbfgs=150)
            if e < best_e:
                best_e = e
                best_pos = pos.copy()
        except Exception:
            continue
    
    # Final polish: one more relaxation on the best candidate
    if time.time() < deadline - 5:
        pos_final, e_final = relax_positions(best_pos, maxiter=300)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final
    
    return best_pos

# EVOLVE-BLOCK-END
