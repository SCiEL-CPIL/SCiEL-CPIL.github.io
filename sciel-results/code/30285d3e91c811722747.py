import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 55
DEADLINE_SECONDS = 145.0  # Safety margin from 150s
_start_time = time.time()

def _remaining():
    return DEADLINE_SECONDS - (time.time() - _start_time)

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for N atoms."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0) # Avoid division by zero
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def _lj_energy_grad(pos):
    """Compute gradient of Lennard-Jones energy w.r.t. Cartesian coords."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    # grad_i = sum_j 4(-12 r_ij^-14 + 6 r_ij^-8) * (r_i - r_j)
    coeff = 4.0 * (-12.0 * r2**(-7) + 6.0 * r2**(-4))
    np.fill_diagonal(coeff, 0.0)
    sum_coeff = np.sum(coeff, axis=1, keepdims=True)
    sum_coeff_pos = np.sum(coeff[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_coeff - sum_coeff_pos
    return grad

def _make_mackay_icosahedron():
    """
    Generate the Mackay icosahedron coordinates for 55 atoms.
    Structure: 1 center, 12 inner shell (vertices), 30 middle shell (edge midpoints), 12 outer shell (vertices).
    Radii are optimized to minimize energy.
    """
    # Icosahedral vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    v_unit = verts / norms
    
    # Find edge midpoints
    # Edges connect vertices that are closest to each other
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(v_unit[i] - v_unit[j])
            dists[i, j] = d
            dists[j, i] = d
    min_dist = np.min(dists[dists > 0])
    edge_mask = dists < min_dist * 1.001
    
    v_edge = []
    for i in range(12):
        for j in range(i + 1, 12):
            if edge_mask[i, j]:
                mid = (v_unit[i] + v_unit[j]) / 2.0
                mid_unit = mid / np.linalg.norm(mid)
                v_edge.append(mid_unit)
    v_edge = np.array(v_edge)
    
    # Optimize radii for the Mackay structure
    # Typical radii: r1 ~ 1.04, r2 ~ 1.76, r3 ~ 1.98
    # We will refine these with a quick Nelder-Mead optimization
    def energy_from_radii(radii):
        r1, r2, r3 = radii
        pos = np.vstack([
            np.zeros((1, 3)),
            v_unit * r1,
            v_edge * r2,
            v_unit * r3
        ])
        # Check min distance
        d = pdist(pos)
        if d.min() < 0.5:
            return 1e10
        return _lj_energy(pos)
    
    # Initial guess
    initial_radii = np.array([1.04, 1.76, 1.98])
    
    # Quick optimization of radii
    try:
        res = minimize(
            energy_from_radii,
            initial_radii,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        best_radii = res.x
    except Exception:
        best_radii = initial_radii
        
    r1, r2, r3 = best_radii
    pos = np.vstack([
        np.zeros((1, 3)),
        v_unit * r1,
        v_edge * r2,
        v_unit * r3
    ])
    
    # Center the structure
    pos -= np.mean(pos, axis=0)
    
    return pos

def _relax_lbfgs(pos, maxiter=1000, gtol=1e-12):
    """Relax the structure using L-BFGS."""
    if _remaining() < 5.0:
        return pos, _lj_energy(pos)
        
    def energy_func(x):
        p = x.reshape(-1, 3)
        return _lj_energy(p)
        
    def grad_func(x):
        p = x.reshape(-1, 3)
        return _lj_energy_grad(p).flatten()
        
    try:
        result = minimize(
            energy_func,
            pos.flatten(),
            jac=grad_func,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-14, 'maxls': 50}
        )
        new_pos = result.x.reshape(-1, 3)
        return new_pos, result.fun
    except Exception:
        return pos, _lj_energy(pos)

def _multi_stage_polish(pos):
    """
    Multi-stage polish: progressively tighten convergence tolerance.
    Stage 1: Moderate tolerance, many iterations
    Stage 2: Tight tolerance, fewer iterations  
    Stage 3: Very tight tolerance, final polish
    """
    if _remaining() < 10.0:
        return pos, _lj_energy(pos)
    
    # Stage 1: Moderate polish
    pos1, e1 = _relax_lbfgs(pos, maxiter=500, gtol=1e-10)
    if _remaining() < 15.0:
        return pos1, e1
    
    # Stage 2: Tighter polish
    pos2, e2 = _relax_lbfgs(pos1, maxiter=300, gtol=1e-13)
    if _remaining() < 20.0:
        return pos2, e2
    
    # Stage 3: Very tight final polish
    pos3, e3 = _relax_lbfgs(pos2, maxiter=200, gtol=1e-15)
    
    return pos3, e3

def construct():
    global _start_time
    _start_time = time.time()
    
    # Step 1: Generate initial Mackay Icosahedron with optimized radii
    pos_init = _make_mackay_icosahedron()
    energy_init = _lj_energy(pos_init)
    
    # Step 2: Multi-stage polish for high-precision convergence
    pos_final, energy_final = _multi_stage_polish(pos_init)
    
    # Safety check: ensure minimum distance constraint
    d = pdist(pos_final)
    if d.min() < 0.5:
        # Fallback: try initial position if it was valid
        d_init = pdist(pos_init)
        if d_init.min() >= 0.5:
            pos_final = pos_init
            energy_final = energy_init
        else:
            # Scale to fix constraint
            scale = 0.5 / d.min() * 0.999
            pos_final = pos_final * scale
            energy_final = _lj_energy(pos_final)
            
    return pos_final
