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

def _build_polytetrahedral_seed(n_atoms, rng):
    """
    Generate a polytetrahedral seed structure.
    Polytetrahedral structures are built from tetrahedra sharing faces.
    We start with a central tetrahedron and add atoms in shells.
    """
    # Tetrahedron edge length for Dzugutov potential
    # The minimum of the potential is around r ~ 1.9-2.0
    edge = 1.95
    
    # Central tetrahedron vertices
    # Regular tetrahedron with edge length `edge`
    s = edge
    h = s * np.sqrt(2/3)  # height of tetrahedron
    r_base = s / np.sqrt(6)  # distance from center to base vertex
    
    # Place tetrahedron centered at origin
    v0 = np.array([r_base, r_base, r_base]) * np.sqrt(2)
    v1 = np.array([r_base, -r_base, -r_base]) * np.sqrt(2)
    v2 = np.array([-r_base, r_base, -r_base]) * np.sqrt(2)
    v3 = np.array([-r_base, -r_base, r_base]) * np.sqrt(2)
    
    # Scale to get correct edge length
    # Distance between v0 and v1: sqrt((2*r_base*sqrt(2))^2 + (2*r_base*sqrt(2))^2 + 0) = 2*r_base*2 = 4*r_base
    # We want this to equal s, so scale factor = s / (4*r_base)
    scale = s / (4 * r_base)
    v0 *= scale
    v1 *= scale
    v2 *= scale
    v3 *= scale
    
    points = [v0.copy(), v1.copy(), v2.copy(), v3.copy()]
    
    # Add atoms in concentric shells with polytetrahedral bias
    # Polytetrahedral structures have coordination numbers of 4, 12, etc.
    # We'll add atoms in a way that favors tetrahedral coordination
    
    shell_radii = []
    # First shell: atoms at distance ~edge from existing atoms
    # For a tetrahedron, the next shell is at distance edge from each vertex
    # This creates a structure like a truncated tetrahedron or icosahedron-like
    
    target_dist = edge
    min_dist = 0.55  # slightly above 0.5 to be safe
    
    max_attempts_per_atom = 200
    
    while len(points) < n_atoms:
        # Try to add an atom that maintains polytetrahedral character
        added = False
        for attempt in range(max_attempts_per_atom):
            # Random direction
            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(rng.uniform(-1, 1))
            
            # Try various radii
            for r_frac in [0.9, 1.0, 1.1, 1.2, 1.3]:
                r = target_dist * r_frac
                p = np.array([
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    r * np.cos(phi)
                ])
                
                # Check minimum distance
                valid = True
                for q in points:
                    if np.linalg.norm(p - q) < min_dist:
                        valid = False
                        break
                
                if valid:
                    # Check that this doesn't create too-close pairs
                    points.append(p)
                    added = True
                    break
            
            if added:
                break
        
        if not added:
            # Force add with random position
            r = target_dist * rng.uniform(0.8, 1.5)
            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(rng.uniform(-1, 1))
            p = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])
            points.append(p)
    
    return np.array(points[:n_atoms])

def _anneal_dzugutov(positions, n_atoms, c_start=3.0, d_start=0.8, c_end=1.10, d_end=0.27, 
                     n_steps=800, lr_base=0.02, cooling_rate=0.999):
    """
    Anneal the Dzugutov potential softness parameters and follow the gradient.
    Uses exponential cooling for a smoother annealing schedule.
    """
    A, B, a, b, m = 5.82, 1.28, 1.87, 1.94, 16
    pos = positions.copy()
    
    # Track energy for adaptive step sizing
    prev_energy = None
    
    for step in range(n_steps):
        # Exponential annealing schedule
        t = step / max(n_steps - 1, 1)
        # Use exponential decay for smoother transition
        c_val = c_end + (c_start - c_end) * np.exp(-5.0 * t)
        d_val = d_end + (d_start - d_end) * np.exp(-5.0 * t)
        params = {'A': A, 'B': B, 'a': a, 'b': b, 'c': c_val, 'd': d_val, 'm': m}
        
        # Compute gradient
        grad = _dzugutov_energy_gradient(pos, params)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-12:
            break
        
        # Adaptive step size: larger when gradient is small (exploration), 
        # smaller when gradient is large (exploitation)
        # Also decrease step size as we anneal
        step_size = lr_base * (1.0 + 0.5 * np.exp(-2.0 * t)) / (1.0 + 0.1 * grad_norm)
        step_size = min(step_size, 0.1)  # Cap step size
        
        pos = pos - step_size * grad
        
        # Ensure minimum separation of 0.5 with iterative projection
        for iter_fix in range(5):
            max_violation = 0.0
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    diff = pos[i] - pos[j]
                    r = np.linalg.norm(diff)
                    if r < 0.5:
                        if r > 1e-10:
                            push = (0.5 - r) * diff / r
                            pos[i] += 0.5 * push
                            pos[j] -= 0.5 * push
                        else:
                            # Random small displacement
                            pos[i] += rng.uniform(-0.05, 0.05, 3)
                            pos[j] -= rng.uniform(-0.05, 0.05, 3)
                        max_violation = max(max_violation, 0.5 - r)
            
            if max_violation < 1e-6:
                break
    
    return pos

def construct(instance):
    """Construct a low-energy Dzugutov cluster using continuous annealing."""
    n_atoms = instance["n"]
    rng = np.random.default_rng(n_atoms * 12345)
    
    # Time budget
    start_time = time.time()
    time_limit = 45  # Seconds per instance to stay within 300s for 6 instances
    
    # Generate a high-quality polytetrahedral seed
    positions = _build_polytetrahedral_seed(n_atoms, rng)
    
    # Run continuous annealing
    annealed = _anneal_dzugutov(positions, n_atoms, c_start=3.5, d_start=1.0, 
                                 c_end=1.10, d_end=0.27, n_steps=1000, lr_base=0.015)
    
    # Local refinement with scipy
    def objective(x):
        pos = x.reshape(n_atoms, 3)
        return _dzugutov_energy(pos)
    
    def gradient(x):
        pos = x.reshape(n_atoms, 3)
        return _dzugutov_energy_gradient(pos).flatten()
    
    result = minimize(objective, annealed.flatten(), jac=gradient, method='L-BFGS-B',
                     options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-10})
    
    best_positions = result.x.reshape(n_atoms, 3)
    best_energy = result.fun
    
    # Try a second annealing run with different parameters if time allows
    if time.time() - start_time < time_limit * 0.6:
        # Slightly perturbed seed
        perturbed = positions + rng.normal(0, 0.05, positions.shape)
        annealed2 = _anneal_dzugutov(perturbed, n_atoms, c_start=2.8, d_start=0.7, 
                                      c_end=1.10, d_end=0.27, n_steps=800, lr_base=0.018)
        
        result2 = minimize(objective, annealed2.flatten(), jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10})
        
        if result2.fun < best_energy:
            best_positions = result2.x.reshape(n_atoms, 3)
            best_energy = result2.fun
    
    # Ensure minimum separation
    for _ in range(10):
        max_violation = 0.0
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                diff = best_positions[i] - best_positions[j]
                r = np.linalg.norm(diff)
                if r < 0.5 and r > 1e-10:
                    push = (0.5 - r) * diff / r
                    best_positions[i] += 0.5 * push
                    best_positions[j] -= 0.5 * push
                    max_violation = max(max_violation, 0.5 - r)
        
        if max_violation < 1e-6:
            break
    
    return best_positions
# EVOLVE-BLOCK-END
