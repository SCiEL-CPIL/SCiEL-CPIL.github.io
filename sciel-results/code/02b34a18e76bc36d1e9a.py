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
        # Check min distance constraint softly
        d = pdist(p)
        if d.min() < 0.5:
            return 1e10 + (0.5 - d.min())**2
        return _lj_energy(p)
        
    def grad_func(x):
        p = x.reshape(-1, 3)
        d = pdist(p)
        if d.min() < 0.5:
            # Gradient of penalty
            # This is complex to compute analytically for the penalty, 
            # so we rely on L-BFGS finite differences if penalty is active, 
            # but we expect it to be inactive for valid structures.
            # For safety, if constraint is violated, return zero gradient to stop?
            # Better: just return the LJ gradient and hope the optimizer steps away.
            # Or use a large penalty that dominates.
            pass
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
        # Final check
        d = pdist(new_pos)
        if d.min() < 0.5:
             # If constraint violated, revert or fix
             # For now, return original if it was better
             if _lj_energy(pos) < result.fun:
                 return pos, _lj_energy(pos)
        return new_pos, result.fun
    except Exception:
        return pos, _lj_energy(pos)

def _basin_hopping(pos_init, max_steps=50, step_scale=0.1):
    """
    Perform basin hopping starting from pos_init.
    """
    if _remaining() < 10.0:
        return pos_init, _lj_energy(pos_init)
        
    current_pos = pos_init.copy()
    current_energy = _lj_energy(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(42)
    
    for step in range(max_steps):
        if _remaining() < 15.0:
            break
            
        # Perturb the structure
        # Random translation, rotation, and internal noise
        perturbed_pos = current_pos.copy()
        
        # Random translation
        trans = rng.normal(0, step_scale, size=(3,))
        perturbed_pos += trans
        
        # Random rotation (small)
        # Generate a random rotation matrix
        axis = rng.normal(0, 1, size=(3,))
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-step_scale, step_scale)
        # Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        perturbed_pos = perturbed_pos @ R.T
        
        # Internal noise (break symmetry slightly to escape local minima)
        noise = rng.normal(0, step_scale * 0.5, size=perturbed_pos.shape)
        perturbed_pos += noise
        
        # Relax the perturbed structure
        relaxed_pos, relaxed_energy = _relax_lbfgs(perturbed_pos, maxiter=200, gtol=1e-10)
        
        # Accept if better (Hill Climbing)
        # For global min search, simple hill climbing is often sufficient if starting near global min.
        # Metropolis can help escape, but for this specific problem, we want the lowest energy.
        if relaxed_energy < current_energy:
            current_pos = relaxed_pos
            current_energy = relaxed_energy
            if current_energy < best_energy:
                best_pos = current_pos.copy()
                best_energy = current_energy
        else:
            # Optional: Metropolis acceptance with temperature
            # But since we want the global min, and we start from a very good guess,
            # we can just reject worse moves to stay in the basin.
            pass
            
    return best_pos, best_energy

def construct():
    global _start_time
    _start_time = time.time()
    
    # Step 1: Generate initial Mackay Icosahedron
    pos_init = _make_mackay_icosahedron()
    energy_init = _lj_energy(pos_init)
    
    # Step 2: Initial Relaxation
    pos_relaxed, energy_relaxed = _relax_lbfgs(pos_init, maxiter=500, gtol=1e-12)
    
    # Step 3: Basin Hopping
    # Use a smaller number of steps to save time, as the initial guess is very good
    pos_bh, energy_bh = _basin_hopping(pos_relaxed, max_steps=30, step_scale=0.05)
    
    # Step 4: Final High-Precision Polish
    # Use very tight tolerances
    pos_final, energy_final = _relax_lbfgs(pos_bh, maxiter=2000, gtol=1e-15)
    
    # Safety check
    d = pdist(pos_final)
    if d.min() < 0.5:
        # Fallback to previous best
        if energy_relaxed < energy_final:
            pos_final = pos_relaxed
            energy_final = energy_relaxed
        else:
            # Try to fix by scaling if needed
            scale = 0.5 / d.min() * 0.99
            pos_final = pos_final * scale
            energy_final = _lj_energy(pos_final)
            
    return pos_final
