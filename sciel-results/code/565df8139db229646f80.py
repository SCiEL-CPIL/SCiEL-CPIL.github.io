import numpy as np
import time
from scipy.optimize import minimize

N = 38

def lj_energy(pos):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    r_inv6 = dist_sq**-3
    r_inv12 = r_inv6**2
    total = 4.0 * np.sum(r_inv12 - r_inv6)
    return 0.5 * total

def lj_grad(pos):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    r_inv8 = dist_sq**-2
    r_inv14 = dist_sq**-4
    coeff = 4.0 * (-12.0 * r_inv14 + 6.0 * r_inv8)
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad

def relax(pos):
    res = minimize(lj_energy, pos, jac=lj_grad, method='L-BFGS-B', 
                   options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
    return res.x

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_energy = np.inf
    best_pos = None
    
    # Run multiple independent basin hopping chains
    # 10 chains, roughly 10s each
    n_chains = 10
    
    for i in range(n_chains):
        if time.time() > deadline - 5.0:
            break
            
        # Random start in a box of size 6
        pos = np.random.rand(N, 3) * 6.0
        
        # Relax initial
        pos = relax(pos)
        e = lj_energy(pos)
        
        if e < best_energy:
            best_energy = e
            best_pos = pos.copy()
            
        # Basin Hopping loop
        while time.time() < deadline - 5.0:
             # Perturb
             # Scale 0.5 is reasonable
             pos_pert = pos + np.random.normal(0, 0.5, pos.shape)
             
             # Relax
             pos_rel = relax(pos_pert)
             e_new = lj_energy(pos_rel)
             
             if e_new < best_energy:
                 best_energy = e_new
                 best_pos = pos_rel.copy()
                 
             # Move to new state
             pos = pos_rel
             
    if best_pos is None:
        return np.random.rand(N, 3) * 5.0
        
    return best_pos
