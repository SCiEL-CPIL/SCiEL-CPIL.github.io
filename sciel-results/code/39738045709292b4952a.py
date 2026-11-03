"""Circle packing in perimeter-4 rectangle using simulated annealing."""
import numpy as np

N = 21

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N circles in a rectangle of perimeter 4 (width + height = 2).
    Maximize sum of radii using simulated annealing with force-based moves.
    """
    best_result = None
    best_score = -np.inf
    
    # Try multiple aspect ratios and random seeds
    aspects = [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    
    for aspect in aspects:
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        # Run simulated annealing with multiple random restarts
        for seed in range(3):
            result = _simulated_annealing(w, h, N, seed)
            if result is not None:
                score = np.sum(result[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = result
    
    if best_result is None:
        # Fallback: simple grid packing
        side = int(np.ceil(np.sqrt(N)))
        w, h = 1.0, 1.0
        cells = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            cells[i, 0] = (col + 0.5) * w / side
            cells[i, 1] = (row + 0.5) * h / side
            cells[i, 2] = 0.5 / side - 0.01
        return cells
    
    return best_result


def _simulated_annealing(w, h, N, seed=0, max_iter=5000, T0=0.1, T_end=0.001):
    """
    Simulated annealing to pack N circles in w x h rectangle.
    State: positions (N,2) and radii (N,)
    Moves: 
      1. Move a circle position
      2. Change a circle radius
      3. Move a circle position AND change radius
    Acceptance: Metropolis criterion on energy = -sum(radii) + penalty
    """
    rng = np.random.RandomState(seed)
    
    # Initialize: place circles in a jittered grid
    side = int(np.ceil(np.sqrt(N)))
    positions = np.zeros((N, 2))
    for i in range(N):
        row = i // side
        col = i % side
        positions[i, 0] = (col + 0.5) * w / side + rng.uniform(-0.05, 0.05)
        positions[i, 1] = (row + 0.5) * h / side + rng.uniform(-0.05, 0.05)
    
    # Clip to valid range
    positions[:, 0] = np.clip(positions[:, 0], 0.01, w - 0.01)
    positions[:, 1] = np.clip(positions[:, 1], 0.01, h - 0.01)
    
    # Initial radii: small, uniform
    radii = np.full(N, 0.03)
    
    # Compute initial energy
    current_energy = _energy(positions, radii, w, h)
    
    # Track best
    best_positions = positions.copy()
    best_radii = radii.copy()
    best_energy = current_energy
    
    # Simulated annealing parameters
    T = T0
    cooling_rate = (T_end / T0) ** (1.0 / max_iter)
    
    for iteration in range(max_iter):
        # Generate a random move
        move_type = rng.choice(3, p=[0.4, 0.3, 0.3])
        
        new_positions = positions.copy()
        new_radii = radii.copy()
        
        if move_type == 0:
            # Move a circle position
            i = rng.randint(N)
            new_positions[i] += rng.normal(0, 0.05, size=2)
        elif move_type == 1:
            # Change a circle radius
            i = rng.randint(N)
            new_radii[i] *= rng.uniform(0.8, 1.2)
            new_radii[i] = np.clip(new_radii[i], 0.01, min(w, h) / 2)
        else:
            # Move position and change radius
            i = rng.randint(N)
            new_positions[i] += rng.normal(0, 0.05, size=2)
            new_radii[i] *= rng.uniform(0.9, 1.1)
            new_radii[i] = np.clip(new_radii[i], 0.01, min(w, h) / 2)
        
        # Clip positions to stay in bounds (with margin for radius)
        for i in range(N):
            r = new_radii[i]
            new_positions[i, 0] = np.clip(new_positions[i, 0], r, w - r)
            new_positions[i, 1] = np.clip(new_positions[i, 1], r, h - r)
        
        # Compute new energy
        new_energy = _energy(new_positions, new_radii, w, h)
        
        # Metropolis acceptance
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            positions = new_positions
            radii = new_radii
            current_energy = new_energy
            
            # Update best
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = positions.copy()
                best_radii = radii.copy()
        
        # Cool down
        T *= cooling_rate
    
    # Verify best solution
    result = _verify_and_build(best_positions, best_radii, w, h)
    return result


def _energy(positions, radii, w, h):
    """
    Compute energy: -sum(radii) + overlap_penalty + boundary_penalty
    Lower energy is better.
    """
    n = len(radii)
    energy = -np.sum(radii)
    
    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                overlap = min_dist - dist
                energy += 50.0 * overlap * overlap
    
    # Boundary penalties
    for i in range(n):
        x, y, r = positions[i, 0], positions[i, 1], radii[i]
        if x < r:
            energy += 50.0 * (r - x) ** 2
        if x > w - r:
            energy += 50.0 * (x - (w - r)) ** 2
        if y < r:
            energy += 50.0 * (r - y) ** 2
        if y > h - r:
            energy += 50.0 * (y - (h - r)) ** 2
    
    return energy


def _verify_and_build(positions, radii, w, h):
    """
    Verify that the packing is valid (no overlaps, all in bounds).
    Return (N,3) array if valid, else None.
    """
    n = len(radii)
    eps = 1e-6
    
    # Check boundaries
    for i in range(n):
        x, y, r = positions[i, 0], positions[i, 1], radii[i]
        if r <= 0:
            return None
        if x < r - eps or x > w - r + eps:
            return None
        if y < r - eps or y > h - r + eps:
            return None
    
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - eps:
                return None
    
    # Build result
    result = np.zeros((n, 3))
    for i in range(n):
        result[i] = [positions[i, 0], positions[i, 1], radii[i]]
    return result

# EVOLVE-BLOCK-END
