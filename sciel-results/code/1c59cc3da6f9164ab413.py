import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE_SECONDS = 110.0

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for N atoms."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    r2 = np.where(r2 < 1e-12, 1.0, r2)
    r6 = r2**(-3)
    r12 = r6**2
    energy = 4.0 * np.sum(r12 - r6) - 4.0 * n
    return energy

def _lj_energy_grad(pos):
    """Compute gradient of Lennard-Jones energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    r2 = np.where(r2 < 1e-12, 1.0, r2)
    r6 = r2**(-3)
    coeff = 4.0 * (-12.0 * r2**(-7) + 6.0 * r2**(-4))
    np.fill_diagonal(coeff, 0.0)
    grad = 2.0 * np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad

def _get_icosahedral_directions():
    """Get 12 vertex directions and 30 edge-midpoint directions of regular icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            dists[i, j] = d
            dists[j, i] = d
    
    min_dist = np.min(dists[dists > 0])
    edge_mask = dists < min_dist * 1.001
    
    edge_midpoints = []
    for i in range(12):
        for j in range(i+1, 12):
            if edge_mask[i, j]:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid_unit = mid / np.linalg.norm(mid)
                edge_midpoints.append(mid_unit)
    
    edge_midpoints = np.array(edge_midpoints)
    
    return verts_unit, edge_midpoints

def _mackay_icosahedron_55(r1, r2, r3):
    """
    Construct 55-atom Mackay icosahedron with independent radii.
    r1: radius for inner 12 atoms
    r2: radius for middle 30 atoms
    r3: radius for outer 12 atoms
    """
    verts_unit, edge_midpoints = _get_icosahedral_directions()
    
    points = [np.zeros(3)]  # center atom
    
    # Inner shell: 12 vertices at radius r1
    for v in verts_unit:
        points.append(r1 * v)
    
    # Middle shell: 30 edge-midpoint directions at radius r2
    for v in edge_midpoints:
        points.append(r2 * v)
    
    # Outer shell: 12 vertex directions at radius r3
    for v in verts_unit:
        points.append(r3 * v)
    
    return np.array(points, dtype=float)

def _optimize_radii():
    """Optimize the three shell radii to minimize LJ energy."""
    def energy_from_radii(params):
        r1, r2, r3 = params
        if r1 < 0.3 or r2 < 0.5 or r3 < 0.5 or r2 < r1 or r3 < r2:
            return 1e10
        pos = _mackay_icosahedron_55(r1, r2, r3)
        return _lj_energy(pos)
    
    def grad_from_radii(params):
        r1, r2, r3 = params
        if r1 < 0.3 or r2 < 0.5 or r3 < 0.5 or r2 < r1 or r3 < r2:
            return np.array([1.0, 1.0, 1.0])
        
        pos = _mackay_icosahedron_55(r1, r2, r3)
        grad_cart = _lj_energy_grad(pos)
        
        # grad_r1 = sum of grad_cart[i] dot (pos[i] / r1) for i in inner shell
        # Indices:
        # 0: center
        # 1-12: inner shell (r1)
        # 13-42: middle shell (r2)
        # 43-54: outer shell (r3)
        
        grad_r1 = np.sum(np.sum(grad_cart[1:13] * pos[1:13], axis=1) / r1)
        grad_r2 = np.sum(np.sum(grad_cart[13:43] * pos[13:43], axis=1) / r2)
        grad_r3 = np.sum(np.sum(grad_cart[43:55] * pos[43:55], axis=1) / r3)
        
        return np.array([grad_r1, grad_r2, grad_r3])
    
    # Initial guess
    r0 = [1.05, 1.75, 2.0]
    
    try:
        result = minimize(
            energy_from_radii, 
            r0, 
            jac=grad_from_radii,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-16}
        )
        return result.x, result.fun
    except Exception:
        return r0, energy_from_radii(r0)

def _relax(pos0, maxiter=5000, gtol=1e-12, ftol=1e-16):
    """Apply L-BFGS relaxation to a starting position."""
    def energy_func(x):
        return _lj_energy(x.reshape(-1, 3))
    
    def grad_func(x):
        return _lj_energy_grad(x.reshape(-1, 3)).flatten()
    
    try:
        result = minimize(
            energy_func, 
            pos0.flatten(), 
            jac=grad_func,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol}
        )
        return result.x.reshape(-1, 3), result.fun
    except Exception:
        return pos0, _lj_energy(pos0)

def construct():
    start_time = time.time()
    
    # Step 1: Optimize shell radii
    try:
        radii, radius_energy = _optimize_radii()
    except Exception:
        radii = [1.05, 1.75, 2.0]
        pos_tmp = _mackay_icosahedron_55(*radii)
        radius_energy = _lj_energy(pos_tmp)
    
    pos0 = _mackay_icosahedron_55(*radii)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 20:
        return pos0
    
    # Step 2: Primary L-BFGS relaxation
    pos_relaxed, energy_relaxed = _relax(pos0, maxiter=10000)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 10:
        return pos_relaxed
    
    # Step 3: Final high-precision polish
    # Use tighter tolerances to ensure we are at the minimum
    pos_final, energy_final = _relax(pos_relaxed, maxiter=20000, gtol=1e-14, ftol=1e-18)
    
    if energy_final < energy_relaxed:
        pos_relaxed = pos_final
        energy_relaxed = energy_final
    
    return pos_relaxed
