"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

def _dzugutov_energy_and_grad(pos):
    """
    Compute Dzugutov energy and gradient.
    V(r) = A(r^-m - B)exp(c/(r - a)) [r < a] + B exp(d/(r - b)) [r < b]
    A = 5.82, B = 1.28, a = 1.87, b = 1.94, c = 1.10, d = 0.27, m = 16
    """
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)

    # Compute pairwise distances
    # diff[i, j, :] = pos[i] - pos[j]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask for i < j to avoid double counting and self-interaction
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    rij = dists[mask]
    diff_masked = diff[mask] # shape (num_pairs, 3)
    
    energy = 0.0
    grad = np.zeros_like(pos)
    
    # Parameters
    r1 = a
    r2 = b
    
    # Region 1: r < a
    mask1 = rij < r1
    if np.any(mask1):
        r = rij[mask1]
        diff_r = diff_masked[mask1]
        
        # V(r) = A * (r^-m - B) * exp(c / (r - a))
        # Note: r - a is negative here.
        
        # dV/dr = A * [ -m * r^(-m-1) * exp(c/(r-a)) + (r^-m - B) * exp(c/(r-a)) * (-c * (r-a)^(-2)) ]
        # dV/dr = A * exp(c/(r-a)) * [ -m * r^(-m-1) - c * (r^-m - B) * (r-a)^(-2) ]
        
        term1 = -m * r**(-m-1)
        term2 = -c * (r**(-m) - B) * (r - a)**(-2)
        dVdr = A * np.exp(c / (r - a)) * (term1 + term2)
        
        energy += np.sum(A * (r**(-m) - B) * np.exp(c / (r - a)))
        
        # Gradient contribution: dE/dpos_i = sum_j dV/dr_ij * (pos_i - pos_j) / r_ij
        # dV/dr is the derivative w.r.t r.
        # Force on i from j is -dV/dr * (pos_i - pos_j) / r
        # So grad_i += dV/dr * (pos_i - pos_j) / r
        # Wait, standard Lennard Jones: F = -dU/dr * r_hat. 
        # U depends on r. dU/dpos_i = dU/dr * dr/dpos_i = dU/dr * (pos_i - pos_j)/r.
        # So grad = dV/dr * (pos_i - pos_j) / r.
        
        weights = dVdr / r
        # weights shape (num_pairs,)
        # diff_r shape (num_pairs, 3) where diff_r = pos_i - pos_j
        
        # We need to sum these vectors into the gradient array.
        # Indices for mask
        idx_i, idx_j = np.where(mask1)
        
        for k in range(len(idx_i)):
            i = idx_i[k]
            j = idx_j[k]
            w = weights[k]
            d = diff_r[k]
            grad[i] += w * d
            grad[j] -= w * d

    # Region 2: a <= r < b
    mask2 = (rij >= r1) & (rij < r2)
    if np.any(mask2):
        r = rij[mask2]
        diff_r = diff_masked[mask2]
        
        # V(r) = B * exp(d / (r - b))
        # dV/dr = B * exp(d/(r-b)) * (-d * (r-b)^(-2))
        
        dVdr = -B * d * np.exp(d / (r - b)) * (r - b)**(-2)
        
        energy += np.sum(B * np.exp(d / (r - b)))
        
        weights = dVdr / r
        idx_i, idx_j = np.where(mask2)
        
        for k in range(len(idx_i)):
            i = idx_i[k]
            j = idx_j[k]
            w = weights[k]
            d = diff_r[k]
            grad[i] += w * d
            grad[j] -= w * d

    # Region 3: r >= b
    # V(r) = 0, dV/dr = 0

    return energy, grad

def _repulsive_energy_and_grad(pos, k_rep, r_cut=1.5):
    """
    Auxiliary potential to encourage separation and polytetrahedral order.
    V_rep(r) = k_rep * (1/r - 1/r_cut)^2 for r < r_cut, else 0.
    This pushes atoms apart if they are too close, but is zero at r_cut.
    """
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)

    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    rij = dists[mask]
    diff_masked = diff[mask]
    
    energy = 0.0
    grad = np.zeros_like(pos)
    
    mask_rep = rij < r_cut
    if np.any(mask_rep):
        r = rij[mask_rep]
        diff_r = diff_masked[mask_rep]
        
        # V = k * (1/r - 1/r_cut)^2
        # dV/dr = 2 * k * (1/r - 1/r_cut) * (-1/r^2)
        
        term = (1.0/r - 1.0/r_cut)
        dVdr = -2.0 * k_rep * term / (r**2)
        
        energy += k_rep * np.sum(term**2)
        
        weights = dVdr / r
        idx_i, idx_j = np.where(mask_rep)
        
        for k in range(len(idx_i)):
            i = idx_i[k]
            j = idx_j[k]
            w = weights[k]
            d = diff_r[k]
            grad[i] += w * d
            grad[j] -= w * d

    return energy, grad

def _lbfgs_minimize(pos, energy_func, max_iter=200, tol=1e-5):
    """
    Simple L-BFGS implementation or use scipy.optimize.minimize.
    We will use scipy.optimize.minimize with L-BFGS-B for robustness.
    """
    from scipy.optimize import minimize
    
    def wrapper(x):
        pos = x.reshape(-1, 3)
        e, g = energy_func(pos)
        return e, g.flatten()
    
    x0 = pos.flatten()
    res = minimize(wrapper, x0, jac=True, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-5, 'maxls': 20})
    
    pos_opt = res.x.reshape(-1, 3)
    return pos_opt, res.fun

def _generate_initial_cluster(n):
    """
    Generate a random compact cluster with minimum distance 0.5.
    """
    rng = np.random.default_rng(42 + n)
    # Estimate radius
    # Volume ~ n * (0.5^3) / packing_fraction
    # Let's just use a sphere with radius ~ 1.2 * n^(1/3)
    radius = 1.5 * (n ** (1.0/3.0))
    
    points = []
    # Place center atom
    points.append(np.zeros(3))
    
    max_attempts = 5000
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
        # Sample from sphere
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        # Radius distribution: prefer outer shells? Or uniform?
        # Uniform in volume: r = R * u^(1/3)
        r = radius * rng.uniform(0.2, 1.0) ** (1.0/3.0)
        p = v * r
        
        # Check distance to existing points
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        
        if valid:
            points.append(p)
            
    if len(points) < n:
        # Fallback: add points on a grid or spiral if random fails
        # This is unlikely for n<60 with radius 1.5*n^(1/3)
        # For n=60, radius ~ 5.1. Volume ~ 550. 60 atoms * 0.125 vol ~ 7.5. Should be fine.
        pass
        
    return np.array(points)

def construct(instance):
    """
    Repulsive Annealing Strategy:
    1. Generate random initial structures.
    2. Optimize using a blend of Dzugutov potential and a repulsive auxiliary potential.
    3. Gradually reduce the weight of the repulsive potential to 0, allowing the system
       to settle into the true Dzugutov energy landscape.
    4. This helps escape local minima caused by the complex polytetrahedral interactions.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit
    
    size = instance["n"]
    potential = instance["potential"]
    
    best_energy = np.inf
    best_positions = None
    
    # Number of restarts
    n_restarts = 10
    
    # Annealing schedule
    # We will run a sequence of optimizations with decreasing k_rep
    # k_rep values: 10.0, 5.0, 1.0, 0.2, 0.0
    k_rep_schedule = [10.0, 5.0, 1.0, 0.2, 0.0]
    
    # We will try a few different initial configurations
    # To save time, we'll do full annealing for 3 restarts, and shorter annealing for others
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial structure
        # Use different seeds for diversity
        np.random.seed(1000 + restart)
        init_pos = _generate_initial_cluster(size)
        
        if init_pos is None or len(init_pos) != size:
            continue
            
        current_pos = init_pos.copy()
        
        # Annealing loop
        for k_rep in k_rep_schedule:
            if time.time() > deadline:
                break
                
            # Define the combined energy function
            def combined_energy(pos, k_rep_val=k_rep):
                # Dzugutov energy
                e_dzug, g_dzug = _dzugutov_energy_and_grad(pos)
                
                # Repulsive energy
                e_rep, g_rep = _repulsive_energy_and_grad(pos, k_rep_val, r_cut=1.5)
                
                e_total = e_dzug + e_rep
                g_total = g_dzug + g_rep
                
                return e_total, g_total
            
            # Minimize
            try:
                pos_opt, e_opt = _lbfgs_minimize(current_pos, combined_energy, max_iter=100, tol=1e-4)
                
                # Check validity (min distance)
                min_dist = np.inf
                # Quick check
                for i in range(size):
                    for j in range(i+1, size):
                        d = np.linalg.norm(pos_opt[i] - pos_opt[j])
                        if d < min_dist:
                            min_dist = d
                        if d < 0.5:
                            # Invalid, penalize or break
                            break
                    if min_dist < 0.5:
                        break
                        
                if min_dist < 0.5:
                    # If invalid, we might need to push them apart
                    # For now, skip this step if it becomes invalid
                    # Or just continue to next k_rep with current_pos
                    # Actually, if it becomes invalid, the repulsive term should have prevented it.
                    # If k_rep is 0, it might happen.
                    # Let's just keep current_pos if it becomes invalid
                    pass
                else:
                    current_pos = pos_opt
            except Exception:
                continue
        
        # After annealing, do a final pure Dzugutov relaxation using the provided relax function
        # This ensures we are in a local minimum of the true potential
        try:
            # The provided relax function might be more robust or tuned
            # However, we can also just use our own L-BFGS with pure Dzugutov
            # Let's use the provided relax for safety and consistency with evaluator expectations
            # But the provided relax might be slow or use a specific method.
            # Let's use our L-BFGS for the final step to ensure it's a local min of Dzugutov.
            final_pos, final_e = _lbfgs_minimize(current_pos, _dzugutov_energy_and_grad, max_iter=200, tol=1e-5)
            
            # Verify min distance
            min_dist = np.inf
            for i in range(size):
                for j in range(i+1, size):
                    d = np.linalg.norm(final_pos[i] - final_pos[j])
                    if d < min_dist:
                        min_dist = d
            if min_dist >= 0.5:
                if final_e < best_energy:
                    best_energy = final_e
                    best_positions = final_pos
        except Exception:
            # Fallback to current_pos if final minimization fails
            if current_pos is not None and len(current_pos) == size:
                e_cur, _ = _dzugutov_energy_and_grad(current_pos)
                # Check min dist
                min_dist = np.inf
                for i in range(size):
                    for j in range(i+1, size):
                        d = np.linalg.norm(current_pos[i] - current_pos[j])
                        if d < min_dist:
                            min_dist = d
                if min_dist >= 0.5 and e_cur < best_energy:
                    best_energy = e_cur
                    best_positions = current_pos

    # Fallback if no valid structure found
    if best_positions is None:
        # Generate a simple valid structure
        rng = np.random.default_rng(0)
        radius = 2.0 * (size ** (1.0/3.0))
        points = []
        points.append(np.zeros(3))
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
        best_positions = np.array(points)
        
        # Try to relax it at least once
        try:
            relaxed_pos, _ = relax(best_positions, potential, instance.get("rho", instance.get("kind")))
            best_positions = relaxed_pos
        except Exception:
            pass

    return best_positions
# EVOLVE-BLOCK-END
