"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _compute_radius(pos):
    """Compute the actual packing radius from positions."""
    n = len(pos)
    if n == 0:
        return 0.0
    # Distance to boundary
    dist_boundary = 0.5 - np.max(np.abs(pos), axis=1)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances using vectorized computation
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = float(np.min(dist))
    return min(min_pair / 2.0, min_boundary)

def _negative_radius(pos_flat):
    """Objective function: negative of the packing radius."""
    pos = pos_flat.reshape(-1, 2)
    r = _compute_radius(pos)
    return -r

def _initial_hex_grid(n):
    """Create initial hexagonal lattice packing."""
    # Hexagonal lattice with spacing 1
    rows = int(np.ceil(np.sqrt(n * 2 / 3)))
    cols = int(np.ceil(n / rows))
    
    points = []
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            x = col * 1.0
            y = row * np.sqrt(3) / 2
            if row % 2 == 1:
                x += 0.5
            points.append([x, y])
            count += 1
        if count >= n:
            break
    
    points = np.array(points[:n])
    
    # Center the lattice
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    points -= center
    
    # Scale to fit in [-0.5, 0.5]^2 with margin
    max_coord = np.max(np.abs(points))
    if max_coord > 0:
        scale = 0.45 / max_coord
        points *= scale
    
    return points

def _initial_random(n, seed=42):
    """Create initial random positions."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 2))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Try multiple initializations
    best_pos = None
    best_r = 0.0
    
    # Start with hexagonal lattice
    init_methods = [
        ("hex", _initial_hex_grid(N)),
        ("rand1", _initial_random(N, seed=1)),
        ("rand2", _initial_random(N, seed=2)),
        ("rand3", _initial_random(N, seed=42)),
    ]
    
    for name, pos in init_methods:
        if time.time() > deadline - 30:
            break
        
        pos = np.clip(pos, -0.49, 0.49)
        current_r = _compute_radius(pos)
        
        if current_r > best_r:
            best_r = current_r
            best_pos = pos.copy()
        
        # Use scipy L-BFGS-B optimization
        pos_flat = pos.flatten()
        
        def objective(x):
            if time.time() > deadline - 5:
                return 1e10
            return _negative_radius(x)
        
        try:
            result = minimize(
                objective,
                pos_flat,
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
            )
            
            if result.success or result.fun < 0:
                opt_pos = result.x.reshape(-1, 2)
                opt_pos = np.clip(opt_pos, -0.4999, 0.4999)
                opt_r = _compute_radius(opt_pos)
                
                if opt_r > best_r:
                    best_r = opt_r
                    best_pos = opt_pos.copy()
        except Exception:
            pass
    
    # Additional local refinement using simulated annealing-like approach
    if time.time() < deadline - 10 and best_pos is not None:
        pos = best_pos.copy()
        current_r = best_r
        temp = 0.001
        cooling = 0.99
        
        for iteration in range(500):
            if time.time() > deadline - 5:
                break
            
            # Random perturbation
            delta = np.random.randn(*pos.shape) * temp
            new_pos = pos + delta
            new_pos = np.clip(new_pos, -0.4999, 0.4999)
            
            new_r = _compute_radius(new_pos)
            
            # Accept if better or with probability based on temperature
            if new_r > current_r or np.random.rand() < math.exp((new_r - current_r) / (temp + 1e-15)):
                pos = new_pos
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_pos = pos.copy()
            
            temp *= cooling
    
    # Final polish with small steps
    if time.time() < deadline - 2 and best_pos is not None:
        pos = best_pos.copy()
        small_step = 0.001
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            
            # Compute gradient numerically
            grad = np.zeros_like(pos)
            eps = 1e-6
            for i in range(N):
                for j in range(2):
                    pos_plus = pos.copy()
                    pos_plus[i, j] += eps
                    pos_plus = np.clip(pos_plus, -0.4999, 0.4999)
                    r_plus = _compute_radius(pos_plus)
                    
                    pos_minus = pos.copy()
                    pos_minus[i, j] -= eps
                    pos_minus = np.clip(pos_minus, -0.4999, 0.4999)
                    r_minus = _compute_radius(pos_minus)
                    
                    grad[i, j] = (r_plus - r_minus) / (2 * eps)
            
            # Take step in gradient direction
            new_pos = pos + small_step * grad
            new_pos = np.clip(new_pos, -0.4999, 0.4999)
            new_r = _compute_radius(new_pos)
            
            if new_r > _compute_radius(pos):
                pos = new_pos
                if new_r > best_r:
                    best_r = new_r
                    best_pos = pos.copy()
            else:
                small_step *= 0.5
                if small_step < 1e-8:
                    break
    
    # Return best position found
    if best_pos is None:
        return _initial_hex_grid(N)
    
    return best_pos
# EVOLVE-BLOCK-END
