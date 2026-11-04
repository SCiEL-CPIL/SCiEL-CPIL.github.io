"""Circle packing in unit square - optimized with iterative relaxation."""
import numpy as np
import time

N = 26

def _initial_grid(n):
    """Create initial grid placement."""
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y])
    # If we need more, add random positions
    while len(positions) < n:
        positions.append([np.random.rand(), np.random.rand()])
    positions = np.array(positions[:n])
    # Initial radius estimate
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            min_dist = min(min_dist, d)
    r = min(min_dist / 2, 0.5 / side) * 0.95
    return positions, r

def _check_validity(positions, radii):
    """Check if all circles are valid (in bounds and non-overlapping)."""
    x, y = positions[:, 0], positions[:, 1]
    r = radii
    
    # Check bounds
    if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
        return False
    
    # Check overlap
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < radii[i] + radii[j] - 1e-9:
                return False
    return True

def _optimize(positions, radii, max_iters=5000, lr=0.01):
    """Iteratively optimize positions and radii."""
    n = len(positions)
    positions = positions.copy()
    radii = radii.copy()
    
    best_score = np.sum(radii)
    best_positions = positions.copy()
    best_radii = radii.copy()
    
    for it in range(max_iters):
        # Compute forces
        forces = np.zeros_like(positions)
        
        # Repulsion between overlapping circles
        for i in range(n):
            for j in range(i+1, n):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-10:
                    # Push apart
                    overlap = min_dist - dist
                    force_dir = diff / dist
                    forces[i] += force_dir * overlap
                    forces[j] -= force_dir * overlap
                elif dist <= 1e-10:
                    # Random small push
                    forces[i] += np.random.randn(2) * 0.01
                    forces[j] -= np.random.randn(2) * 0.01
        
        # Boundary forces
        for i in range(n):
            # Left wall
            if positions[i, 0] < radii[i]:
                forces[i, 0] += (radii[i] - positions[i, 0]) * 2
            # Right wall
            if positions[i, 0] > 1 - radii[i]:
                forces[i, 0] -= (positions[i, 0] - (1 - radii[i])) * 2
            # Bottom wall
            if positions[i, 1] < radii[i]:
                forces[i, 1] += (radii[i] - positions[i, 1]) * 2
            # Top wall
            if positions[i, 1] > 1 - radii[i]:
                forces[i, 1] -= (positions[i, 1] - (1 - radii[i])) * 2
        
        # Update positions
        positions += lr * forces
        
        # Clamp to valid range
        for i in range(n):
            positions[i, 0] = np.clip(positions[i, 0], 0, 1)
            positions[i, 1] = np.clip(positions[i, 1], 0, 1)
        
        # Try to increase radii slightly
        # Find minimum clearance
        min_clearance = np.inf
        for i in range(n):
            # Clearance from walls
            min_clearance = min(min_clearance, positions[i, 0], 1 - positions[i, 0],
                                positions[i, 1], 1 - positions[i, 1])
            for j in range(i+1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                min_clearance = min(min_clearance, dist - radii[i] - radii[j])
        
        # Increase all radii by a small amount if there's clearance
        if min_clearance > 0:
            increase = min(min_clearance * 0.5, 0.001)
            radii += increase
        
        # Ensure radii are positive and reasonable
        radii = np.maximum(radii, 1e-6)
        radii = np.minimum(radii, 0.5)
        
        score = np.sum(radii)
        if score > best_score:
            best_score = score
            best_positions = positions.copy()
            best_radii = radii.copy()
    
    return best_positions, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    best_result = None
    best_score = -np.inf
    
    # Multiple restarts with different initializations
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Grid initialization
            positions, r = _initial_grid(N)
            radii = np.full(N, r)
        elif restart % 2 == 1:
            # Random initialization with moderate radii
            positions = np.random.rand(N, 2)
            r = 0.1 + 0.05 * np.random.rand()
            radii = np.full(N, r)
        else:
            # Grid with perturbation
            positions, r = _initial_grid(N)
            positions += 0.05 * np.random.randn(N, 2)
            positions = np.clip(positions, 0.05, 0.95)
            radii = np.full(N, r * 0.9)
        
        # Optimize
        opt_positions, opt_radii, score = _optimize(positions, radii, max_iters=3000, lr=0.005)
        
        # Check validity
        if _check_validity(opt_positions, opt_radii):
            if score > best_score:
                best_score = score
                best_result = np.column_stack([opt_positions, opt_radii])
        
        # Try with smaller learning rate for refinement
        if time.time() < deadline - 5:
            opt_positions2, opt_radii2, score2 = _optimize(opt_positions, opt_radii, max_iters=2000, lr=0.001)
            if _check_validity(opt_positions2, opt_radii2):
                if score2 > best_score:
                    best_score = score2
                    best_result = np.column_stack([opt_positions2, opt_radii2])
    
    # If no valid result found, return seed-like result
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
