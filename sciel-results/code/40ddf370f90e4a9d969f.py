"""Dzugutov cluster global minima: Icosahedral seeds + BFGS relaxation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

_START = time.time()
TIME_LIMIT = 290.0

def _generate_icosahedron_55():
    points = [[0.0, 0.0, 0.0]]
    phi = (1 + np.sqrt(5)) / 2
    r1 = 1.9
    
    v = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    v = np.array(v, dtype=float)
    norms = np.linalg.norm(v, axis=1)
    v = v * (r1 / norms[:, np.newaxis])
    points.extend(v.tolist())
    
    edges = []
    n_v = len(v)
    for i in range(n_v):
        for j in range(i+1, n_v):
            d = np.linalg.norm(v[i] - v[j])
            if d < 1.5 * r1:
                edges.append((i, j))
                
    shell2 = []
    r2 = r1 * 1.9
    
    for i in range(12):
        dir_vec = v[i] / np.linalg.norm(v[i])
        shell2.append(dir_vec * r2)
        
    for i, j in edges:
        dir_vec = (v[i] + v[j])
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        shell2.append(dir_vec * r2)
        
    points.extend(shell2)
    return np.array(points)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    ico_55 = _generate_icosahedron_55()
    
    if size <= 55:
        seed = ico_55[:size]
    else:
        seed = ico_55.copy()
        if size > 55:
            extra = size - 55
            outer_shell = seed[13:]
            rng = np.random.default_rng(size)
            for _ in range(extra):
                idx = rng.integers(0, len(outer_shell))
                base = outer_shell[idx]
                noise = rng.normal(0, 0.1, 3)
                new_pos = base + noise
                seed = np.vstack([seed, new_pos])
                
    def objective(x):
        pos = x.reshape(size, 3)
        e, g = dzugutov_energy_gradient(pos)
        return e
        
    def gradient(x):
        pos = x.reshape(size, 3)
        e, g = dzugutov_energy_gradient(pos)
        return g.flatten()
        
    x0 = seed.flatten()
    
    try:
        res = minimize(
            objective, 
            x0, 
            jac=gradient, 
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        final_pos = res.x.reshape(size, 3)
    except Exception as e:
        final_pos = seed
        
    return final_pos

# EVOLVE-BLOCK-END
