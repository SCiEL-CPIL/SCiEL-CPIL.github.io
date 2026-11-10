"""
Strategy: Crystal Melting via BCC Initialization and Simulated Annealing.
We initialize atoms on a BCC lattice, which is the known ground state for 
long-range Morse potentials. We then use a simulated annealing approach 
with L-BFGS-B gradient-based local minimization to explore the energy 
landscape, cooling the "temperature" to settle into a low-energy state.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # 300s limit, keep 10s safety margin

def _morse_energy_and_grad(positions, rho):
    """Compute exact Morse energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf) # Avoid division by zero
    
    # E = sum exp(rho(1-r)) * (exp(rho(1-r)) - 2)
    x = rho * (1.0 - r)
    e = np.exp(x)
    energy = np.sum(e * (e - 2.0))
    
    # Gradient: dE/dr = -2 * rho * exp(2x)
    # dE/dri = dE/dr * (ri - rj) / r
    dEdr = -2.0 * rho * np.exp(2.0 * x)
    dEdr = dEdr / r
    
    # Gradient for each atom i is sum over j of dE/dr * (ri - rj)
    # Using broadcasting: dEdr is (n, n). We need sum_j dEdr_ij * (ri - rj)
    # diff is ri - rj.
    grad = np.sum(dEdr[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _generate_bcc(n):
    """Generate n atoms on a BCC lattice cropped to a sphere."""
    points = []
    # BCC lattice points: (0,0,0) and (0.5, 0.5, 0.5) in unit cube
    # We want spacing ~ 1.0
    # Generate points in a grid
    max_coord = int(np.ceil(n ** (1.0/3.0) * 0.8)) + 2
    
    # Create a list of potential BCC points
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            for z in range(-max_coord, max_coord + 1):
                # Corner
                candidates.append((x, y, z))
                # Body center
                candidates.append((x + 0.5, y + 0.5, z + 0.5))
                
    # Sort by distance from origin to pick the most compact ones
    candidates.sort(key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
    
    # Take the first n points
    for i in range(n):
        if i < len(candidates):
            points.append(candidates[i])
            
    if len(points) < n:
        # Fallback if not enough points (unlikely)
        rng = np.random.default_rng(n)
        while len(points) < n:
            p = rng.uniform(-max_coord, max_coord, 3)
            points.append(tuple(p))
            
    return np.array(points)

def _relax_lbfgs(positions, rho, max_iter=200):
    """Relax positions using L-BFGS-B."""
    def fun(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g.flatten()
        
    res = minimize(fun, positions.flatten(), method='L-BFGS-B', jac=True, 
                   options={'maxiter': max_iter, 'maxls': 50})
    return res.x.reshape(-1, 3), res.fun

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Initialize on BCC
    pos = _generate_bcc(size)
    
    # Initial relaxation
    pos, e = _relax_lbfgs(pos, parameter)
    
    best_pos = pos.copy()
    best_e = e
    
    # Simulated Annealing Loop
    # Start with a moderate temperature
    # Typical energy scales for these clusters are ~ -10 to -50
    T = 5.0
    cooling_rate = 0.95
    max_sa_iter = 500
    
    current_pos = pos
    current_e = e
    
    for i in range(max_sa_iter):
        if time.time() > _TIME_LIMIT:
            break
            
        # Perturbation
        # Add noise to positions
        # The noise scale should be related to T
        # For L-BFGS to work, we need to be in a basin, so noise shouldn't be too huge
        # But for SA, we need to explore.
        # Let's try a random kick
        rng = np.random.default_rng(i + size)
        noise = rng.normal(0, np.sqrt(T) * 0.1, size=(size, 3))
        new_pos = current_pos + noise
        
        # Relax the perturbed position
        new_pos, new_e = _relax_lbfgs(new_pos, parameter, max_iter=100)
        
        # Metropolis Criterion
        if new_e < current_e:
            current_pos = new_pos
            current_e = new_e
            if new_e < best_e:
                best_e = new_e
                best_pos = new_pos.copy()
        else:
            delta_e = new_e - current_e
            accept_prob = np.exp(-delta_e / T)
            if rng.random() < accept_prob:
                current_pos = new_pos
                current_e = new_e
                
        # Cool down
        T *= cooling_rate
        if T < 0.01:
            break
            
    return best_pos
# EVOLVE-BLOCK-END
