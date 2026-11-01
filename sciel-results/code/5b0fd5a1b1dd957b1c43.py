import numpy as np
import time
import scipy.optimize

N = 55

def _lj_energy_and_grad(pos):
    """Compute Lennard-Jones energy and gradient."""
    pos = pos.reshape(-1, 3)
    N = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    r2[np.diag_indices_from(r2)] = 1.0 
    r6 = r2**3
    r12 = r6**2
    E = 4.0 * np.sum(r12 - r6) / 2.0
    
    inv_r2_4 = 1.0 / (r2**4)
    inv_r2_7 = 1.0 / (r2**7)
    factor = 24.0 * (inv_r2_4 - 2.0 * inv_r2_7)
    np.fill_diagonal(factor, 0.0)
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return E, grad

def _generate_mackay_icosahedron():
    """Generate initial positions for the 55-atom Mackay icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    r_inner = 1.12
    inner_atoms = verts * r_inner
    center = np.array([[0.0, 0.0, 0.0]])
    inner_atoms = np.vstack([center, inner_atoms])
    
    r_outer_v = 2.2
    outer_v_atoms = verts * r_outer_v
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, 2.0):
                edges.append((i, j))
                
    edge_mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        norm = np.linalg.norm(mid)
        edge_mids.append(mid / norm)
        
    edge_mids = np.array(edge_mids)
    
    r_outer_e = 2.2
    outer_e_atoms = edge_mids * r_outer_e
    
    all_atoms = np.vstack([inner_atoms, outer_v_atoms, outer_e_atoms])
    
    return all_atoms

def construct():
    start_time = time.time()
    
    pos = _generate_mackay_icosahedron()
    
    def objective(x):
        E, _ = _lj_energy_and_grad(x)
        return E
        
    def gradient(x):
        _, g = _lj_energy_and_grad(x)
        return g

    try:
        result = scipy.optimize.minimize(
            objective, 
            pos, 
            method='L-BFGS-B', 
            jac=gradient,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        pos = result.x
    except Exception:
        pass
        
    return pos
