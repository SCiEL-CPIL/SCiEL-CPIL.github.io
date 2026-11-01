"""Circle packing in a unit square - maximize minimum radius for 50 circles."""
import numpy as np
import time

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill

def _compute_radius(centers):
    """Compute the minimum radius for a given set of centers."""
    # Minimum distance between any two centers
    min_pair_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if d < min_pair_dist:
                min_pair_dist = d
    
    # Minimum distance to boundary
    min_boundary = np.inf
    for i in range(N):
        for k in range(2):
            d = 0.5 - np.abs(centers[i][k])
            if d < min_boundary:
                min_boundary = d
    
    return min(min_pair_dist / 2.0, min_boundary)

def _objective(centers):
    """Negative radius (we minimize negative to maximize radius)."""
    return -_compute_radius(centers)

def _hexagonal_grid(n):
    """Create a hexagonal-like grid arrangement for initial guess."""
    centers = np.zeros((n, 2))
    # Try different arrangements
    # Start with a grid and perturb
    m = int(np.ceil(np.sqrt(n)))
    cols = m
    rows = int(np.ceil(n / cols))
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x = (col + 0.5) / cols - 0.5
            y = (row + 0.5) / rows - 0.5
            # Add hexagonal offset
            if row % 2 == 1:
                x += 0.5 / cols
            centers[idx] = [x, y]
            idx += 1
    
    return centers[:n]

def _random_init(n, seed=42):
    """Random initialization within the square."""
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-0.45, 0.45, size=(n, 2))
    return centers

def _simulated_annealing(centers, max_time=100.0):
    """Run simulated annealing to improve the packing."""
    start_time = time.time()
    current = centers.copy()
    current_radius = _compute_radius(current)
    best = current.copy()
    best_radius = current_radius
    
    # Temperature schedule
    T_start = 0.05
    T_end = 0.0001
    total_steps = 50000
    
    for step in range(total_steps):
        if time.time() - start_time > max_time:
            break
        
        # Cooling schedule (exponential)
        T = T_start * (T_end / T_start) ** (step / total_steps)
        
        # Random perturbation
        candidate = current.copy()
        i = np.random.randint(N)
        delta = np.random.normal(0, T, size=2)
        candidate[i] += delta
        
        # Keep within bounds (soft constraint)
        candidate[i] = np.clip(candidate[i], -0.49, 0.49)
        
        candidate_radius = _compute_radius(candidate)
        delta_obj = current_radius - candidate_radius  # We want to maximize radius
        
        # Accept if better or with probability exp(-delta/T)
        if delta_obj < 0 or np.random.random() < np.exp(-delta_obj / T):
            current = candidate
            current_radius = candidate_radius
            
            if current_radius > best_radius:
                best = current.copy()
                best_radius = current_radius
    
    return best, best_radius

def _local_optimization(centers, max_time=20.0):
    """Local optimization using gradient-free methods."""
    start_time = time.time()
    current = centers.copy()
    current_radius = _compute_radius(current)
    best = current.copy()
    best_radius = current_radius
    
    step_size = 0.01
    patience = 5000
    no_improve = 0
    
    while time.time() - start_time < max_time:
        improved = False
        for i in range(N):
            if time.time() - start_time > max_time:
                    break
            
            # Try moving in 4 directions
            for dx in [step_size, -step_size, 0]:
                for dy in [step_size, -step_size, 0]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    candidate = current.copy()
                    candidate[i][0] += dx
                    candidate[i][1] += dy
                    
                    # Keep within bounds
                    candidate[i] = np.clip(candidate[i], -0.49, 0.49)
                    
                    candidate_radius = _compute_radius(candidate)
                    
                    if candidate_radius > current_radius:
                        current = candidate
                        current_radius = candidate_radius
                        improved = True
                        
                        if current_radius > best_radius:
                            best = current.copy()
                            best_radius = current_radius
                        break
                if improved:
                    break
        
        if not improved:
            step_size *= 0.5
            no_improve += 1
            if step_size < 1e-6 or no_improve > 20:
                break
        else:
            no_improve = 0
    
    return best, best_radius

def construct():
    """Main entry point: return centers of 50 circles maximizing minimum radius."""
    start_time = time.time()
    
    # Try multiple initializations and pick the best
    best_centers = None
    best_radius = 0.0
    
    # Initial guesses
    initializations = [
        _hexagonal_grid(N),
        _random_init(N, seed=1),
        _random_init(N, seed=2),
        _random_init(N, seed=3),
    ]
    
    for init_centers in initializations:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Simulated annealing
        sa_centers, sa_radius = _simulated_annealing(init_centers, max_time=30.0)
        
        # Local optimization
        local_centers, local_radius = _local_optimization(sa_centers, max_time=10.0)
        
        if local_radius > best_radius:
            best_radius = local_radius
            best_centers = local_centers
    
    # Final deadline check
    if time.time() - start_time > DEADLINE - 2:
        if best_centers is not None:
            return best_centers
        return _hexagonal_grid(N)
    
    return best_centers if best_centers is not None else _hexagonal_grid(N)
