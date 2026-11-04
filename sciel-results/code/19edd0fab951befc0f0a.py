"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    best_score = -np.inf
    best_result = None
    
    # Try different aspect ratios (w + h = 2)
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.67, 0.5, 2.0, 0.4, 2.5, 0.333, 3.0]
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 30:
            break
        
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 - w
        
        # Initial configuration: grid-like placement
        n_cols = max(1, int(np.round(np.sqrt(N * w / h))))
        n_rows = max(1, int(np.ceil(N / n_cols)))
        
        # Create initial positions
        positions = np.zeros((N, 2))
        radii = np.zeros(N)
        
        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            positions[i, 0] = (col + 0.5) * w / n_cols
            positions[i, 1] = (row + 0.5) * h / n_rows
        
        # Initial radii based on cell size
        cell_w = w / n_cols
        cell_h = h / n_rows
        init_r = min(cell_w, cell_h) * 0.45
        radii[:] = init_r
        
        # Gradient-based optimization
        result = _optimize(w, h, positions.copy(), radii.copy(), start_time, deadline)
        
        if result is not None:
            score = result[:, 2].sum()
            if score > best_score:
                best_score = score
                best_result = result
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result


def _optimize(w, h, positions, radii, start_time, deadline):
    """Optimize circle positions and radii using gradient descent with penalties."""
    
    N = len(radii)
    lr_pos = 0.01
    lr_rad = 0.005
    penalty_weight = 100.0
    n_iterations = 5000
    
    for it in range(n_iterations):
        if time.time() > deadline - 10:
            break
        
        # Compute gradient
        grad_pos = np.zeros_like(positions)
        grad_rad = np.zeros_like(radii)
        
        # Boundary constraints: circles must stay within [0, w] x [0, h]
        # Left wall: x_i >= r_i
        # Right wall: x_i <= w - r_i
        # Bottom wall: y_i >= r_i
        # Top wall: y_i <= h - r_i
        
        for i in range(N):
            x, y = positions[i]
            r = radii[i]
            
            # Boundary penalties
            # Left
            if x < r:
                grad_pos[i, 0] += penalty_weight * (r - x)
                grad_rad[i] += penalty_weight * (r - x)
            # Right
            if x > w - r:
                grad_pos[i, 0] -= penalty_weight * (x - (w - r))
                grad_rad[i] += penalty_weight * (x - (w - r))
            # Bottom
            if y < r:
                grad_pos[i, 1] += penalty_weight * (r - y)
                grad_rad[i] += penalty_weight * (r - y)
            # Top
            if y > h - r:
                grad_pos[i, 1] -= penalty_weight * (y - (h - r))
                grad_rad[i] += penalty_weight * (y - (h - r))
        
        # Circle-circle constraints: dist(i,j) >= r_i + r_j
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    if dist > 1e-10:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        # Push apart
                        grad_pos[i, 0] += penalty_weight * overlap * nx
                        grad_pos[i, 1] += penalty_weight * overlap * ny
                        grad_pos[j, 0] -= penalty_weight * overlap * nx
                        grad_pos[j, 1] -= penalty_weight * overlap * ny
                        # Grow radii (both contribute to overlap)
                        grad_rad[i] += penalty_weight * overlap * 0.5
                        grad_rad[j] += penalty_weight * overlap * 0.5
                    else:
                        # Nearly coincident, push in random direction
                        angle = np.random.uniform(0, 2 * np.pi)
                        nx = np.cos(angle)
                        ny = np.sin(angle)
                        overlap = min_dist
                        grad_pos[i, 0] += penalty_weight * overlap * nx
                        grad_pos[i, 1] += penalty_weight * overlap * ny
                        grad_pos[j, 0] -= penalty_weight * overlap * nx
                        grad_pos[j, 1] -= penalty_weight * overlap * ny
                        grad_rad[i] += penalty_weight * overlap * 0.5
                        grad_rad[j] += penalty_weight * overlap * 0.5
        
        # Objective: maximize sum of radii -> gradient is +1 for each radius
        grad_rad += 1.0
        
        # Update
        positions -= lr_pos * grad_pos
        radii -= lr_rad * grad_rad
        
        # Clip radii to be positive
        radii = np.maximum(radii, 1e-6)
        
        # Clip positions to be within bounds (hard constraint)
        for i in range(N):
            radii[i] = min(radii[i], w * 0.5, h * 0.5)
            positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
            positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
    
    # Final validation and cleanup
    result = np.column_stack([positions, radii])
    
    # Check validity
    valid = _is_valid(result, w, h)
    if valid:
        return result
    else:
        # Try to fix by shrinking radii
        for i in range(N):
            # Ensure circle fits in bounds
            max_r = min(positions[i, 0], w - positions[i, 0], positions[i, 1], h - positions[i, 1])
            radii[i] = min(radii[i], max_r * 0.99)
        
        result = np.column_stack([positions, radii])
        valid = _is_valid(result, w, h)
        if valid:
            return result
    
    return None


def _is_valid(result, w, h):
    """Check if the packing is valid."""
    N = len(result)
    for i in range(N):
        x, y, r = result[i]
        if r <= 0:
            return False
        if x < r - 1e-9 or x > w - r + 1e-9:
            return False
        if y < r - 1e-9 or y > h - r + 1e-9:
            return False
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = result[j, 0] - result[i, 0]
            dy = result[j, 1] - result[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < result[i, 2] + result[j, 2] - 1e-9:
                return False
    
    return True
# EVOLVE-BLOCK-END
