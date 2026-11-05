"""Rewritten program using continuous annealing of Dzugutov softness parameters."""
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
    # Compute pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    dist = dist[mask]
    
    energy = 0.0
    # For r < a: V = A*(r^-m - B)*exp(c/(r-a))
    # For a <= r < b: V = B*exp(d/(r-b))
    # For r >= b: V = 0
    
    # r < a
    mask_a = dist < a
    if np.any(mask_a):
        r_a = dist[mask_a]
        # Avoid division by zero or negative in exp
        r_a_safe = np.maximum(r_a, 1e-10)
        energy += np.sum(A * (r_a_safe**(-m) - B) * np.exp(c / (r_a_safe - a)))
    
    # a <= r < b
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
                # V = A*(r^-m - B)*exp(c/(r-a))
                # dV/dr = A * (-m*r^(-m-1)) * exp(c/(r-a)) + A*(r^-m - B) * exp(c/(r-a)) * (-c/(r-a)^2)
                r_safe = max(r, 1e-10)
                dV_dr = A * (-m * r_safe**(-m-1)) * np.exp(c / (r_safe - a)) + \
                        A * (r_safe**(-m) - B) * np.exp(c / (r_safe - a)) * (-c / (r_safe - a)**2)
            elif r < b:
                # V = B*exp(d/(r-b))
                # dV/dr = B*exp(d/(r-b)) * (-d/(r-b)^2)
                dV_dr = B * np.exp(d / (r - b)) * (-d / (r - b)**2)
            else:
                continue
            
            # grad_i += dV/dr * (pos_i - pos_j)/r
            # grad_j -= dV/dr * (pos_i - pos_j)/r
            factor = dV_dr / r
            grad[i] += factor * diff
            grad[j] -= factor * diff
    
    return grad

def _anneal_dzugutov(positions, n_atoms, c_start=2.0, d_start=0.5, c_end=1.10, d_end=0.27, n_steps=500, lr=0.05):
    """Anneal the Dzugutov potential softness parameters and follow the gradient."""
    A, B, a, b, m = 5.82, 1.28, 1.87, 1.94, 16
    pos = positions.copy()
    
    for step in range(n_steps):
        # Linear annealing schedule
        t = step / max(n_steps - 1, 1)
        c_val = c_start + (c_end - c_start) * t
        d_val = d_start + (d_end - d_start) * t
        params = {'A': A, 'B': B, 'a': a, 'b': b, 'c': c_val, 'd': d_val, 'm': m}
        
        # Compute gradient
        grad = _dzugutov_energy_gradient(pos, params)
        
        # Update positions with adaptive step size
        step_size = lr * (1.0 - 0.5 * t)  # Decrease step size as we anneal
        pos = pos - step_size * grad
        
        # Ensure minimum separation of 0.5
        # Simple rejection: if any pair is too close, push them apart
        for iter_fix in range(3):
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    diff = pos[i] - pos[j]
                    r = np.linalg.norm(diff)
                    if r < 0.5 and r > 1e-10:
                        push = (0.5 - r) * diff / r
                        pos[i] += 0.5 * push
                        pos[j] -= 0.5 * push
                    elif r <= 1e-10:
                        # Random small displacement
                        pos[i] += np.random.randn(3) * 0.01
                        pos[j] -= np.random.randn(3) * 0.01
    
    return pos

def _polytetrahedral_seed(n_atoms, rng):
    """Generate a rough polytetrahedral-like seed structure."""
    # Start with a small icosahedron or tetrahedron and add atoms
    # For simplicity, use a compact random cluster with polytetrahedral bias
    spacing = 1.1
    radius = spacing * n_atoms ** (1.0/3.0) * 0.8
    
    # Generate points with a bias toward polytetrahedral distances
    points = []
    # Start with a central atom
    points.append(np.array([0.0, 0.0, 0.0]))
    
    # Add atoms in a roughly spherical shell
    for _ in range(n_atoms - 1):
        for attempt in range(100):
            # Random direction
            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(rng.uniform(-1, 1))
            r = radius * (0.3 + 0.7 * rng.uniform(0, 1))
            p = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.9 * spacing for q in points):
                points.append(p)
                break
    
    # If we didn't get enough points, fill with random
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
            points.append(p)
        else:
            # Force add if stuck
            if len(points) < n_atoms:
                points.append(p + rng.uniform(-0.1, 0.1, 3))
    
    return np.array(points[:n_atoms])

def construct(instance):
    """Construct a low-energy Dzugutov cluster using annealing."""
    n_atoms = instance["n"]
    rng = np.random.default_rng(n_atoms)
    
    # Generate multiple initial seeds
    best_energy = np.inf
    best_positions = None
    
    # Time budget
    start_time = time.time()
    time_limit = 40  # Seconds per instance to stay within 300s for 6 instances
    
    for seed_idx in range(4):
        if time.time() - start_time > time_limit:
            break
        
        # Generate a seed
        if seed_idx == 0:
            # Polytetrahedral-like seed
            positions = _polytetrahedral_seed(n_atoms, rng)
        else:
            # Random compact seed
            spacing = 1.1
            radius = spacing * n_atoms ** (1.0/3.0) * 0.75
            points = []
            while len(points) < n_atoms:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                    points.append(p)
            positions = np.array(points)
        
        # Run annealing
        annealed = _anneal_dzugutov(positions, n_atoms, c_start=2.5, d_start=0.6, 
                                     c_end=1.10, d_end=0.27, n_steps=300, lr=0.03)
        
        # Local refinement with scipy
        def objective(x):
            pos = x.reshape(n_atoms, 3)
            return _dzugutov_energy(pos)
        
        def gradient(x):
            pos = x.reshape(n_atoms, 3)
            return _dzugutov_energy_gradient(pos).flatten()
        
        result = minimize(objective, annealed.flatten(), jac=gradient, method='L-BFGS-B',
                         options={'maxiter': 200, 'gtol': 1e-6})
        
        if result.success or result.fun < best_energy:
            final_pos = result.x.reshape(n_atoms, 3)
            final_energy = result.fun
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos
    
    # Ensure minimum separation
    if best_positions is not None:
        for _ in range(10):
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    diff = best_positions[i] - best_positions[j]
                    r = np.linalg.norm(diff)
                    if r < 0.5 and r > 1e-10:
                        push = (0.5 - r) * diff / r
                        best_positions[i] += 0.5 * push
                        best_positions[j] -= 0.5 * push
    
    return best_positions
# EVOLVE-BLOCK-END
