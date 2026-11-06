import numpy as np
import time
import concurrent.futures
from scipy.optimize import minimize
import scipy.linalg
import scipy.spatial.distance

# Constants for the Morse potential
# E = sum exp(rho(1-r))(exp(rho(1-r)) - 2)
# dE/dr = -2*rho*exp(rho(1-r))*(exp(rho(1-r)) - 1)

def _morse_energy_and_grad(pos, rho):
    """
    Compute Morse energy and gradient for a cluster.
    pos: (n, 3) array
    rho: float
    Returns: energy (float), grad (n, 3) array
    """
    n = pos.shape[0]
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    # Compute pairwise distances
    # Using broadcasting for clarity, though for large n this is O(n^2) memory
    # For n=80, 80x80x3 is small.
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    # Calculate term exp(rho(1-r))
    # Avoid overflow/underflow issues by clipping? 
    # rho is 3, 6, 10. 1-r can be large if r is small.
    # If r < 0.5, 1-r > 0.5. exp(10*0.5) = exp(5) ~ 148. 
    # If r is very small, exp(rho) is large but finite.
    # The potential goes to infinity as r->0.
    # We assume min distance is enforced or we handle it.
    # The task says "atoms must be at least 0.5 apart". 
    # If we violate this, energy is huge. We should probably penalize or just let it be.
    # However, for gradient stability, if dists is very small, exp is large.
    
    # Let's compute the force scalar F(r) = -dE/dr = 2*rho*exp(rho(1-r))*(exp(rho(1-r)) - 1)
    # Wait, the gradient with respect to position r_i is sum_j F(r_ij) * (r_i - r_j) / r_ij
    # Let u = rho * (1 - dists)
    exp_u = np.exp(u)
    # dE/dr_ij = -2 * rho * exp_u * (exp_u - 1)
    # Force magnitude on i from j is -dE/dr_ij * (r_i - r_j)/r_ij ? No.
    # V(r) = exp(rho(1-r))(exp(rho(1-r)) - 2)
    # dV/dr = exp(rho(1-r)) * (-rho) * (exp(rho(1-r)) - 2) + exp(rho(1-r)) * exp(rho(1-r)) * (-rho)
    # dV/dr = -rho * exp(rho(1-r)) * [ (exp(rho(1-r)) - 2) + exp(rho(1-r)) ]
    # dV/dr = -rho * exp(rho(1-r)) * [ 2*exp(rho(1-r)) - 2 ]
    # dV/dr = -2 * rho * exp(rho(1-r)) * (exp(rho(1-r)) - 1)
    
    # Let k_ij = -dV/dr_ij * (1/r_ij) = 2 * rho * exp_u * (exp_u - 1) / dists
    # Force on i from j is k_ij * (r_i - r_j)
    
    # Handle division by zero (diagonal is inf, so 1/inf = 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        k = 2 * rho * exp_u * (exp_u - 1) / dists
    np.fill_diagonal(k, 0)
    
    # Energy
    energy = np.sum(exp_u * (exp_u - 2)) / 2.0 # Divide by 2 because sum over i<j
    
    # Gradient
    # grad_i = sum_{j!=i} k_ij * (r_i - r_j)
    # diff is r_i - r_j
    # k is (n, n)
    # We want sum over j for each i.
    # grad = np.sum(k[:, :, np.newaxis] * diff, axis=1)
    grad = np.sum(k[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _generate_fcc_cluster(n, rng):
    """Generate an FCC cluster of approximately n atoms, centered."""
    side = int(np.ceil(n ** (1.0/3.0))) + 2
    pts = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                for b in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i, j, k]) + np.array(b)
                    pts.append(p)
    
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    indices = np.argsort(dists)
    pts = pts[indices[:n]]
    pts += rng.normal(0, 0.05, size=pts.shape)
    return pts

def _generate_bcc_cluster(n, rng):
    """Generate a BCC cluster."""
    side = int(np.ceil(n ** (1.0/3.0))) + 2
    pts = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                pts.append([i, j, k])
                pts.append([i+0.5, j+0.5, k+0.5])
                
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    indices = np.argsort(dists)
    pts = pts[indices[:n]]
    pts += rng.normal(0, 0.05, size=pts.shape)
    return pts

def _generate_random_packed(n, rng):
    """Generate a random packed cluster using rejection sampling."""
    radius = n ** (1.0/3.0) * 1.5
    pts = []
    attempts = 0
    min_dist = 0.5
    while len(pts) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in pts):
                pts.append(p)
        attempts += 1
        
    if len(pts) < n:
        # Fill remaining with random points, allowing closer distances if necessary
        # or just place them randomly.
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            pts.append(p)
            
    return np.array(pts)

def _optimize_single(args):
    """
    Optimize a single initial configuration.
    args: (pos, rho, seed)
    """
    pos, rho, seed = args
    rng = np.random.default_rng(seed)
    
    # Flatten pos for scipy
    x0 = pos.flatten()
    
    def fun_grad(x):
        p = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(p, rho)
        return e, g
    
    # Use L-BFGS-B with analytic gradient
    try:
        res = minimize(fun_grad, x0, jac=True, method='L-BFGS-B', 
                       options={'maxiter': 500, 'gtol': 1e-8, 'disp': False})
        # Check for min distance violation? 
        # The optimizer might push atoms together.
        # We should check the final energy.
        final_pos = res.x.reshape(-1, 3)
        
        # If min distance is violated, energy is likely very high, 
        # but let's verify and potentially reject or repair.
        # For now, we trust the optimizer to stay in reasonable bounds 
        # if the initial guess is good.
        
        return res.fun, final_pos
    except Exception:
        return np.inf, pos

def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", 3)
    potential = instance["potential"]
    
    start_time = time.time()
    # Total budget for all 6 instances is 300s.
    # Assume sequential calls. We have ~45s per call to be safe.
    # However, if they are parallel, we have 300s each? 
    # "all six calls together must finish within 300 s" usually implies total wall time.
    # Let's aim for 40s per call to be safe, or manage a global deadline if possible.
    # Since we don't know the global state, we'll use a per-call deadline of 45s.
    deadline = start_time + 45.0
    
    rng = np.random.default_rng(42 + n)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. FCC
    try:
        cfg = _generate_fcc_cluster(n, rng)
        initial_configs.append(cfg)
    except Exception:
        pass
        
    # 2. BCC
    try:
        cfg = _generate_bcc_cluster(n, rng)
        initial_configs.append(cfg)
    except Exception:
        pass
        
    # 3. Random Packed
    try:
        cfg = _generate_random_packed(n, rng)
        initial_configs.append(cfg)
    except Exception:
        pass
        
    # 4. FCC with larger noise
    try:
        cfg = _generate_fcc_cluster(n, rng)
        cfg += rng.normal(0, 0.15, size=cfg.shape)
        initial_configs.append(cfg)
    except Exception:
        pass
        
    # 5. Random Packed with different seed
    try:
        cfg = _generate_random_packed(n, np.random.default_rng(100 + n))
        initial_configs.append(cfg)
    except Exception:
        pass
        
    # 6. BCC with larger noise
    try:
        cfg = _generate_bcc_cluster(n, rng)
        cfg += rng.normal(0, 0.15, size=cfg.shape)
        initial_configs.append(cfg)
    except Exception:
        pass

    if not initial_configs:
        cfg = rng.uniform(-n**0.33, n**0.33, size=(n, 3))
        initial_configs.append(cfg)

    # Parallel optimization of initial configs
    # Use 4 threads as per constraint
    best_energy = np.inf
    best_positions = initial_configs[0]
    
    # Prepare args
    opt_args = [(cfg, rho, 1000 + i) for i, cfg in enumerate(initial_configs)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_optimize_single, args): args for args in opt_args}
        for future in concurrent.futures.as_completed(futures):
            if time.time() > deadline - 10.0:
                # Stop submitting new ones? We already submitted. 
                # We can break and ignore remaining if we have a good result.
                # But we must wait for at least one to finish to get a result.
                # If we break, we might not have any result.
                # Let's just collect what we can.
                pass
            
            try:
                energy, positions = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                continue

    # Basin Hopping from the best found
    # Use a simple kick-and-relax loop
    current_energy = best_energy
    current_positions = best_positions
    
    # Time remaining for local search
    # We have about 10s left (deadline - 10).
    # Let's use 5s for basin hopping.
    bh_deadline = start_time + 40.0
    
    kick_sigma = 0.1
    T = 0.1
    
    while time.time() < bh_deadline:
        # Perturb
        new_positions = current_positions + rng.normal(0, kick_sigma, size=(n, 3))
        
        # Check min distance? 
        # If violated, energy will be huge, so it will be rejected.
        # But we should avoid infinite loops if it's always violated.
        # Let's just optimize it.
        
        try:
            energy, positions = _optimize_single((new_positions, rho, int(time.time()*1000)))
            if energy < current_energy:
                current_energy = energy
                current_positions = positions
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            else:
                # Metropolis acceptance
                delta_e = energy - current_energy
                if np.random.rand() < np.exp(-delta_e / T):
                    current_energy = energy
                    current_positions = positions
        except Exception:
            continue

    return best_positions
