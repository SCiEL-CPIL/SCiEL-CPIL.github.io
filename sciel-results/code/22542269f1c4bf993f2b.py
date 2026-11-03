"""Pack 50 equal circles in a unit circle.

Uses Differential Evolution (DE) for global optimization and Nelder-Mead
for local refinement. This avoids gradient-based methods (L-BFGS) and
custom Simulated Annealing, using population-based stochastic search
instead.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 50

def _energy(c, r):
    """Calculate the total overlap energy and boundary violation energy."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu, ju = np.triu_indices(N, k=1)
    d = dists[iu, ju]
    
    ov = np.maximum(0.0, 2.0 * r - d)
    e_pairs = np.sum(ov ** 2)
    
    radii = np.sqrt(np.sum(c ** 2, axis=1))
    bv = np.maximum(0.0, radii + r - 1.0)
    e_bound = np.sum(bv ** 2)
    
    return e_pairs + 100.0 * e_bound

def construct():
    r_target = 0.1258254895304
    
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    def obj_flat(x):
        c = x.reshape(N, 2)
        return _energy(c, r_target)
    
    # DE Global Search
    de_res = differential_evolution(
        obj_flat, 
        bounds, 
        maxiter=30, 
        popsize=20, 
        tol=1e-10, 
        atol=1e-12,
        seed=42,
        polish=True
    )
    
    best_c = de_res.x.reshape(N, 2)
    
    # Local Polish with Nelder-Mead
    def obj_local(x):
        c = x.reshape(N, 2)
        return _energy(c, r_target)
        
    nm_res = minimize(
        obj_local, 
        best_c.flatten(), 
        method='Nelder-Mead',
        options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 2000}
    )
    current_c = nm_res.x.reshape(N, 2)
    current_r = r_target
    
    # Increase radius
    step = 1e-5
    for _ in range(30):
        new_r = current_r + step
        def obj_new(x):
            c = x.reshape(N, 2)
            return _energy(c, new_r)
            
        nm_res2 = minimize(
            obj_new, 
            current_c.flatten(), 
            method='Nelder-Mead',
            options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000}
        )
        
        new_c = nm_res2.x.reshape(N, 2)
        e_new = obj_new(new_c.flatten())
        
        if e_new < 1e-8:
            current_r = new_r
            current_c = new_c
        else:
            break
            
    # Final Polish
    def obj_final(x):
        c = x.reshape(N, 2)
        return _energy(c, current_r)
        
    nm_final = minimize(
        obj_final, 
        current_c.flatten(), 
        method='Nelder-Mead',
        options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 5000}
    )
    
    final_c = nm_final.x.reshape(N, 2)
    
    # Verify
    diff = final_c[:, np.newaxis, :] - final_c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu, ju = np.triu_indices(N, k=1)
    min_pair = np.min(dists[iu, ju]) / 2.0
    radii = np.sqrt(np.sum(final_c ** 2, axis=1))
    max_r = np.max(radii)
    boundary_r = 1.0 - max_r
    achievable = min(min_pair, boundary_r)
    
    if achievable < current_r:
        scale = achievable / current_r
        final_c = final_c * scale
        
    return final_c
