import numpy as np
from scipy.optimize import minimize
import time

N = 55

# EVOLVE-BLOCK-START
def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized, numerically stable)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)
    
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _get_icosahedron_vectors():
    """
    Returns the 12 unit vectors to the vertices of a regular icosahedron
    and the 30 unit vectors to the midpoints of the edges.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts_unit = verts / norms[:, np.newaxis]
    
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts[i] - verts[j])
    
    min_dist = np.min(dists[dists > 1e-6])
    
    edge_mids = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                mid = (verts[i] + verts[j]) / 2.0
                norm_mid = np.linalg.norm(mid)
                if norm_mid > 1e-6:
                    edge_mids.append(mid / norm_mid)
                    
    edge_mids = np.array(edge_mids)
    return verts_unit, edge_mids

def _make_mackay_icosahedron(r1, r2, r3):
    """
    Construct a Mackay icosahedron for 55 atoms.
    1 center + 12 first shell + 30 second shell + 12 third shell
    """
    verts_unit, edge_mids = _get_icosahedron_vectors()
    
    center = np.array([[0.0, 0.0, 0.0]])
    first_shell = verts_unit * r1
    second_shell = edge_mids * r2
    third_shell = verts_unit * r3
    
    all_pos = np.vstack([center, first_shell, second_shell, third_shell])
    return all_pos

def construct():
    """Construct 55-atom LJ cluster near global minimum."""
    start_time = time.time()
    deadline = start_time + 145.0

    # Use known optimal radii for the 55-atom Mackay icosahedron
    # These are approximate but close to the minimum
    r1, r2, r3 = 1.08, 1.78, 2.00
    
    pos = _make_mackay_icosahedron(r1, r2, r3)
    
    # Add a tiny deterministic perturbation to avoid exact symmetry issues
    # that might cause the gradient to be zero or ill-conditioned
    # This ensures L-BFGS can take steps if the initial guess is slightly off
    perturbation = np.array([
        [0.001, 0.000, 0.000],
        [0.000, 0.001, 0.000],
        [0.000, 0.000, 0.001]
    ])
    # Apply perturbation to all atoms except the center (index 0)
    pos[1:] += perturbation
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    best_pos = pos
    best_energy = _lj_energy(pos)
    
    # Aggressive L-BFGS relaxation
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={
                'maxiter': 10000, 
                'gtol': 1e-12, 
                'maxls': 1000, 
                'ftol': 1e-16
            }
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    # Check time and return
    if time.time() > deadline - 5.0:
        return best_pos
        
    return best_pos
# EVOLVE-BLOCK-END
