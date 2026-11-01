"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_initialization(n):
    """Create a hexagonal close-packed initial configuration."""
    # Hexagonal packing: rows offset
    # For n circles, we want to pack them in a hexagonal lattice
    # Radius estimate for hex packing in unit square
    # Number of rows
    rows = int(np.ceil(np.sqrt(n * np.sqrt(3) / 2)))
    cols = int(np.ceil(n / rows))
    
    # Hex spacing
    # For hex packing, horizontal spacing = 2r, vertical spacing = sqrt(3)*r
    # Offset every other row by r
    # We want to fit in [0,1]^2
    # Let's estimate r:
    # Width: cols * 2r + r (for offset) <= 1
    # Height: rows * sqrt(3)*r + r <= 1
    
    # Solve for r
    # cols * 2r + r = (2*cols + 1) * r <= 1
    # rows * sqrt(3)*r + r = (rows * sqrt(3) + 1) * r <= 1
    r_x = 1.0 / (2 * cols + 1)
    r_y = 1.0 / (rows * np.sqrt(3) + 1)
    r = min(r_x, r_y)
    
    positions = []
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            x = (2 * col + 1) * r + (r if row % 2 == 1 else 0)
            y = (row * np.sqrt(3) + 1) * r
            # Center in the square
            x = x + 0.5 - (cols * 2 * r + r) / 2
            y = y + 0.5 - (rows * np.sqrt(3) * r + r) / 2
            positions.append((x, y, r))
            count += 1
        if count >= n:
            break
    
    # If we didn't get enough, add more
    while len(positions) < n:
        positions.append((0.5, 0.5, r))
    
    return np.array(positions[:n])


def _random_initialization(n, seed=42):
    """Create a random initial configuration."""
    rng = np.random.default_rng(seed)
    # Start with small circles
    r = 0.02
    positions = []
    for _ in range(n):
        x = rng.uniform(r, 1 - r)
        y = rng.uniform(r, 1 - r)
        positions.append((x, y, r))
    return np.array(positions)


def _physics_preprocess(positions, iterations=500, dt=0.01):
    """Use physics-based push-apart to improve initial configuration."""
    pos = positions.copy()
    n = len(pos)
    
    for _ in range(iterations):
        forces = np.zeros((n, 2))
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = pos[i, 2] + pos[j, 2]
                if dist < min_dist and dist > 1e-10:
                    # Repulsive force
                    force = (min_dist - dist) / dist
                    forces[i, 0] += force * dx
                    forces[i, 1] += force * dy
                    forces[j, 0] -= force * dx
                    forces[j, 1] -= force * dy
                # Boundary forces
                for dim in range(2):
                    if pos[i, dim] < pos[i, 2]:
                        forces[i, dim] += (pos[i, 2] - pos[i, dim])
                    if pos[i, dim] > 1 - pos[i, 2]:
                        forces[i, dim] -= (pos[i, dim] - (1 - pos[i, 2]))
        
        pos[:, :2] += dt * forces
        # Project back to feasible region
        for i in range(n):
            pos[i, 0] = np.clip(pos[i, 0], pos[i, 2], 1 - pos[i, 2])
            pos[i, 1] = np.clip(pos[i, 1], pos[i, 2], 1 - pos[i, 2])
    
    return pos


def _objective(params, n):
    """Objective function: negative sum of radii plus penalty for violations."""
    pos = params.reshape(n, 3)
    x = pos[:, 0]
    y = pos[:, 1]
    r = pos[:, 2]
    
    # Penalty for boundary violations
    boundary_penalty = 0.0
    for i in range(n):
        if x[i] < r[i]:
            boundary_penalty += (r[i] - x[i]) ** 2
        if x[i] > 1 - r[i]:
            boundary_penalty += (x[i] - (1 - r[i])) ** 2
        if y[i] < r[i]:
            boundary_penalty += (r[i] - y[i]) ** 2
        if y[i] > 1 - r[i]:
            boundary_penalty += (y[i] - (1 - r[i])) ** 2
    
    # Penalty for circle overlap
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap_penalty += (min_dist - dist) ** 2
    
    # Objective: maximize sum of radii, penalize violations
    return -np.sum(r) + 100 * (boundary_penalty + overlap_penalty)


def _gradient(params, n):
    """Gradient of the objective function."""
    pos = params.reshape(n, 3)
    x = pos[:, 0]
    y = pos[:, 1]
    r = pos[:, 2]
    
    grad = np.zeros_like(params)
    
    # Gradient of -sum(r)
    grad[2::3] = -1.0
    
    # Gradient of boundary penalty
    for i in range(n):
        if x[i] < r[i]:
            grad[3*i] += 200 * (r[i] - x[i])
            grad[3*i+2] += 200 * (r[i] - x[i])
        if x[i] > 1 - r[i]:
            grad[3*i] += 200 * (x[i] - (1 - r[i]))
            grad[3*i+2] -= 200 * (x[i] - (1 - r[i]))
        if y[i] < r[i]:
            grad[3*i+1] += 200 * (r[i] - y[i])
            grad[3*i+2] += 200 * (r[i] - y[i])
        if y[i] > 1 - r[i]:
            grad[3*i+1] += 200 * (y[i] - (1 - r[i]))
            grad[3*i+2] -= 200 * (y[i] - (1 - r[i]))
    
    # Gradient of overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist and dist > 1e-10:
                penalty = 200 * (min_dist - dist)
                grad[3*i] += penalty * dx / dist
                grad[3*i+1] += penalty * dy / dist
                grad[3*i+2] += penalty
                grad[3*j] -= penalty * dx / dist
                grad[3*j+1] -= penalty * dy / dist
                grad[3*j+2] += penalty
    
    return grad


def _polish(positions, max_iterations=1000, tol=1e-8):
    """High-precision polish pass using L-BFGS-B."""
    n = len(positions)
    params = positions.flatten().copy()
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0, 0.5))  # r
    
    result = minimize(
        _objective,
        params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iterations, 'ftol': tol, 'gtol': tol}
    )
    
    return result.x.reshape(n, 3)


def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    safety_margin = 5.0  # seconds
    
    best_result = None
    best_score = -np.inf
    
    # Generate multiple initializations
    initializations = []
    
    # Hexagonal
    try:
        hex_pos = _hex_initialization(N)
        initializations.append(hex_pos)
    except:
        pass
    
    # Random with different seeds
    for seed in [42, 123, 456, 789, 101]:
        try:
            rand_pos = _random_initialization(N, seed=seed)
            initializations.append(rand_pos)
        except:
            pass
    
    # Process each initialization
    for init_pos in initializations:
        if time.time() - start_time > deadline - safety_margin:
            break
        
        # Physics pre-processing
        try:
            phys_pos = _physics_preprocess(init_pos, iterations=200, dt=0.005)
        except:
            phys_pos = init_pos
        
        # Polish
        try:
            polished = _polish(phys_pos, max_iterations=500, tol=1e-6)
            
            # Validate and compute score
            x = polished[:, 0]
            y = polished[:, 1]
            r = polished[:, 2]
            
            # Check validity
            valid = True
            for i in range(N):
                if x[i] < r[i] - 1e-9 or x[i] > 1 - r[i] + 1e-9:
                    valid = False
                    break
                if y[i] < r[i] - 1e-9 or y[i] > 1 - r[i] + 1e-9:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist < r[i] + r[j] - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_result = polished.copy()
        except:
            continue
    
    # If no valid result found, use a simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
