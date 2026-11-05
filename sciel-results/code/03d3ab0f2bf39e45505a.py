"""Continuous annealing of Dzugutov potential softness parameters for global minima."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _dzugutov_params():
    """Return the Dzugutov potential parameters."""
    return {'A': 5.82, 'B': 1.28, 'a': 1.87, 'b': 1.94, 'c': 1.10, 'd': 0.27, 'm': 16}

def _dzugutov_energy(positions, params=None):
    """Compute the Dzugutov potential energy for a set of positions."""
    if params is None:
        params = _dzugutov_params()
    A, B, a, b, c, d, m = params['A'], params['B'], params['a'], params['b'], params['c'], params['d'], params['m']
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    dist = dist[mask]
    
    energy = 0.0
    mask_a = dist < a
    if np.any(mask_a):
        r_a = dist[mask_a]
        r_a_safe = np.maximum(r_a, 1e-10)
        energy += np.sum(A * (r_a_safe**(-m) - B) * np.exp(c / (r_a_safe - a)))
    
    mask_b = (dist >= a) & (dist < b)
    if np.any(mask_b):
        r_b = dist[mask_b]
        energy += np.sum(B * np.exp(d / (r_b - b)))
    
    return float(energy)

def _dzugutov_energy_gradient(positions, params=None):
    """Compute the gradient of the Dzugutov potential energy."""
    if params is None:
        params = _dzugutov_params()
    A, B, a, b, c, d, m = params['A'], params['B'], params['a'], params['b'], params['c'], params['d'], params['m']
    n = len(positions)
    grad = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(i+1, n):
            diff = positions[i] - positions[j]
            r = np.linalg.norm(diff)
            if r < 1e-10:
                continue
            if r < a:
                r_safe = max(r, 1e-10)
                dV_dr = A * (-m * r_safe**(-m-1)) * np.exp(c / (r_safe - a)) + \
                        A * (r_safe**(-m) - B) * np.exp(c / (r_safe - a)) * (-c / (r_safe - a)**2)
            elif r < b:
                dV_dr = B * np.exp(d / (r - b)) * (-d / (r - b)**2)
            else:
                continue
            
            factor = dV_dr / r
            grad[i] += factor * diff
            grad[j] -= factor * diff
    
    return grad

def _generate_polytetrahedral_seed(n_atoms, rng):
    """
    Generate a high-quality polytetrahedral seed using iterative tetrahedral growth.
    Starts with a regular tetrahedron and grows by adding atoms in tetrahedral
    coordination sites, favoring the polytetrahedral order of the Dzugutov potential.
    """
    # Optimal bond length for Dzugutov potential
    r0 = 1.92
    
    # Regular tetrahedron with edge r0
    # Vertices of regular tetrahedron centered at origin
    s = r0
    h = s * np.sqrt(2/3)
    
    # Tetrahedron vertices
    v0 = np.array([0, 0, h])
    v1 = np.array([s/np.sqrt(3), 0, -h/3])
    v2 = np.array([-s/(2*np.sqrt(3)), s/2, -h/3])
    v3 = np.array([-s/(2*np.sqrt(3)), -s/2, -h/3])
    
    points = [v0.copy(), v1.copy(), v2.copy(), v3.copy()]
    
    min_dist = 0.52
    max_attempts = 300
    
    # Precompute positions array for efficient distance checks
    while len(points) < n_atoms:
        added = False
        for attempt in range(max_attempts):
            # Choose a random existing atom as anchor
            anchor_idx = rng.integers(0, len(points))
            anchor = points[anchor_idx]
            
            # Generate random direction
            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(rng.uniform(-1, 1))
            direction = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            
            # Try to place new atom at tetrahedral distance from anchor
            # and check for tetrahedral coordination
            for dist_factor in [0.95, 1.0, 1.05, 1.1]:
                r = r0 * dist_factor
                candidate = anchor + r * direction
                
                # Check minimum distance to all existing points
                valid = True
                for p in points:
                    if np.linalg.norm(candidate - p) < min_dist:
                        valid = False
                        break
                
                if valid:
                    # Bonus: check if this creates tetrahedral coordination
                    # Count neighbors at ~r0 distance
                    neighbors = sum(1 for p in points if abs(np.linalg.norm(candidate - p) - r0) < 0.15)
                    if neighbors >= 2:
                        points.append(candidate)
                        added = True
                        break
                    elif rng.random() < 0.3:
                        # Accept with lower probability if not tetrahedral
                        points.append(candidate)
                        added = True
                        break
        
        if not added:
            # Fallback: random placement
            r = r0 * rng.uniform(0.8, 1.4)
            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(rng.uniform(-1, 1))
            candidate = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])
            points.append(candidate)
    
    return np.array(points[:n_atoms])

def _enforce_min_distance(positions, min_dist=0.5, max_iters=20):
    """Enforce minimum distance constraint via iterative projection."""
    pos = positions.copy()
    n = len(pos)
    
    for _ in range(max_iters):
        max_violation = 0.0
        for i in range(n):
            for j in range(i+1, n):
                diff = pos[i] - pos[j]
                r = np.linalg.norm(diff)
                if r < min_dist:
                    if r > 1e-12:
                        push = (min_dist - r) * diff / r
                        pos[i] += 0.5 * push
                        pos[j] -= 0.5 * push
                    else:
                        # Random displacement for overlapping atoms
                        pos[i] += np.random.uniform(-0.05, 0.05, 3)
                        pos[j] -= np.random.uniform(-0.05, 0.05, 3)
                    max_violation = max(max_violation, min_dist - r)
        
        if max_violation < 1e-8:
            break
    
    return pos

def _continuous_anneal(positions, n_atoms, c_start=4.0, d_start=1.2, 
                       c_end=1.10, d_end=0.27, n_steps=2000, 
                       lr_init=0.03, lr_end=0.005, rng=None):
    """
    Continuous annealing on Dzugutov softness parameters c and d.
    Uses a smooth exponential schedule to gradually sharpen the potential,
    allowing the system to flow into the global polytetrahedral basin.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    A, B, a, b, m = 5.82, 1.28, 1.87, 1.94, 16
    pos = positions.copy()
    
    # Adaptive learning rate schedule
    lr_schedule = np.linspace(lr_init, lr_end, n_steps)
    
    for step in range(n_steps):
        t = step / max(n_steps - 1, 1)
        
        # Smooth exponential annealing schedule
        # Use sigmoid-like transition for smoother behavior
        c_val = c_end + (c_start - c_end) * np.exp(-4.0 * t)
        d_val = d_end + (d_start - d_end) * np.exp(-4.0 * t)
        
        params = {'A': A, 'B': B, 'a': a, 'b': b, 'c': c_val, 'd': d_val, 'm': m}
        
        # Compute gradient
        grad = _dzugutov_energy_gradient(pos, params)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-14:
            break
        
        # Adaptive step size with damping
        lr = lr_schedule[step]
        # Scale down if gradient is large to avoid instability
        if grad_norm > 10.0:
            lr *= 10.0 / grad_norm
        
        # Take gradient step
        pos = pos - lr * grad
        
        # Enforce minimum distance constraint
        pos = _enforce_min_distance(pos, min_dist=0.5, max_iters=5)
    
    return pos

def construct(instance):
    """
    Construct a low-energy Dzugutov cluster using continuous annealing.
    
    The approach uses a single gradient-based trajectory with continuous
    annealing of the potential's softness parameters (c and d) to suppress
    local minima barriers and flow into the global polytetrahedral basin.
    """
    n_atoms = instance["n"]
    rng = np.random.default_rng(n_atoms * 7919 + 12345)
    
    start_time = time.time()
    time_budget = 42.0  # Leave margin for 6 instances in 300s
    
    # Generate high-quality polytetrahedral seed
    seed_positions = _generate_polytetrahedral_seed(n_atoms, rng)
    
    # Primary continuous annealing run
    annealed = _continuous_anneal(
        seed_positions, n_atoms,
        c_start=4.5, d_start=1.5,
        c_end=1.10, d_end=0.27,
        n_steps=2500,
        lr_init=0.035, lr_end=0.004,
        rng=rng
    )
    
    # Local refinement with L-BFGS-B
    def objective(x):
        pos = x.reshape(n_atoms, 3)
        return _dzugutov_energy(pos)
    
    def gradient(x):
        pos = x.reshape(n_atoms, 3)
        return _dzugutov_energy_gradient(pos).flatten()
    
    result = minimize(
        objective, annealed.flatten(), jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12, 'maxls': 50}
    )
    
    best_positions = result.x.reshape(n_atoms, 3)
    best_energy = result.fun
    
    # Second annealing run with different schedule if time allows
    elapsed = time.time() - start_time
    if elapsed < time_budget * 0.55:
        # Perturbed seed with different annealing parameters
        perturbed_seed = seed_positions + rng.normal(0, 0.08, seed_positions.shape)
        
        annealed2 = _continuous_anneal(
            perturbed_seed, n_atoms,
            c_start=3.8, d_start=1.1,
            c_end=1.10, d_end=0.27,
            n_steps=2000,
            lr_init=0.028, lr_end=0.006,
            rng=rng
        )
        
        result2 = minimize(
            objective, annealed2.flatten(), jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 400, 'gtol': 1e-10, 'ftol': 1e-12, 'maxls': 50}
        )
        
        if result2.fun < best_energy:
            best_positions = result2.x.reshape(n_atoms, 3)
            best_energy = result2.fun
    
    # Third run with very soft initial potential if time allows
    elapsed = time.time() - start_time
    if elapsed < time_budget * 0.75:
        # Even softer start for broader exploration
        very_soft_seed = seed_positions + rng.normal(0, 0.12, seed_positions.shape)
        
        annealed3 = _continuous_anneal(
            very_soft_seed, n_atoms,
            c_start=5.0, d_start=1.8,
            c_end=1.10, d_end=0.27,
            n_steps=1500,
            lr_init=0.04, lr_end=0.008,
            rng=rng
        )
        
        result3 = minimize(
            objective, annealed3.flatten(), jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-12, 'maxls': 50}
        )
        
        if result3.fun < best_energy:
            best_positions = result3.x.reshape(n_atoms, 3)
            best_energy = result3.fun
    
    # Final constraint enforcement
    best_positions = _enforce_min_distance(best_positions, min_dist=0.5, max_iters=30)
    
    return best_positions
# EVOLVE-BLOCK-END
