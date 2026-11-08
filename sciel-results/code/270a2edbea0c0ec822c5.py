"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
def _pairwise_distances(pos):
    """Vectorized pairwise squared distances and minimum distance."""
    # pos: (n, 2)
    # Compute squared distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    min_d2 = np.min(d2)
    return d2, min_d2

def _validity_check(circles, eps=1e-9):
    """Check if circles are valid: inside unit square and non-overlapping."""
    n = len(circles)
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    # Check boundaries
    if np.any(x - r < -eps) or np.any(y - r < -eps) or np.any(x + r > 1 + eps) or np.any(y + r > 1 + eps):
        return False
    # Check pairwise distances
    d2, min_d2 = _pairwise_distances(np.column_stack([x, y]))
    min_dist = np.sqrt(min_d2)
    # For each pair, distance must be >= sum of radii
    # We can check min over pairs of (dist - r_i - r_j)
    # But simpler: check all pairs
    for i in range(n):
        for j in range(i+1, n):
            if d2[i, j] < (r[i] + r[j])**2 - eps:
                return False
    return True

def _score(circles):
    """Sum of radii."""
    return np.sum(circles[:, 2])

def _clamp(circles):
    """Clamp circles to be inside unit square."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    # Ensure r > 0
    r = np.maximum(r, 1e-6)
    # Clamp positions so that circles stay inside [0,1]^2
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    return np.column_stack([x, y, r])

def _relax_and_grow(circles, max_iters=500, growth_rate=0.01, step_size=0.01):
    """Iterative relaxation with simultaneous radius growth."""
    circles = _clamp(circles.copy())
    n = len(circles)
    
    for it in range(max_iters):
        # Compute pairwise distances
        pos = circles[:, :2]
        d2, min_d2 = _pairwise_distances(pos)
        min_dist = np.sqrt(min_d2)
        
        # Check if all pairs are separated
        # For each pair (i,j), we need dist >= r_i + r_j
        # Violation = max(0, r_i + r_j - dist)
        
        # Try to grow radii
        r = circles[:, 2]
        new_r = r + growth_rate
        
        # Check if growth is valid
        pos_new = pos  # positions stay same for now
        d2_new, min_d2_new = _pairwise_distances(pos_new)
        min_dist_new = np.sqrt(min_d2_new)
        
        # For each pair, check if dist >= r_i + r_j
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                if d2_new[i, j] < (new_r[i] + new_r[j])**2:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            circles[:, 2] = new_r
        else:
            # Need to relax positions
            # Compute forces: push apart overlapping circles
            forces = np.zeros_like(pos)
            for i in range(n):
                for j in range(i+1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(d2[i, j])
                    if dist < 1e-12:
                        # Random small perturbation
                        forces[i, 0] += 0.01 * np.random.randn()
                        forces[i, 1] += 0.01 * np.random.randn()
                        forces[j, 0] -= 0.01 * np.random.randn()
                        forces[j, 1] -= 0.01 * np.random.randn()
                    else:
                        overlap = r[i] + r[j] - dist
                        if overlap > 0:
                            # Push apart proportional to overlap
                            fx = overlap * dx / dist
                            fy = overlap * dy / dist
                            forces[i, 0] += fx
                            forces[i, 1] += fy
                            forces[j, 0] -= fx
                            forces[j, 1] -= fy
            
            # Also push away from boundaries
            for i in range(n):
                if pos[i, 0] < r[i]:
                    forces[i, 0] += (r[i] - pos[i, 0])
                if pos[i, 0] > 1 - r[i]:
                    forces[i, 0] -= (pos[i, 0] - (1 - r[i]))
                if pos[i, 1] < r[i]:
                    forces[i, 1] += (r[i] - pos[i, 1])
                if pos[i, 1] > 1 - r[i]:
                    forces[i, 1] -= (pos[i, 1] - (1 - r[i]))
            
            # Update positions
            circles[:, :2] += step_size * forces
            circles = _clamp(circles)
    
    return circles

def _local_search(circles, max_iters=1000, lr=0.01):
    """Local search to maximize sum of radii."""
    circles = _clamp(circles.copy())
    n = len(circles)
    best_score = _score(circles)
    
    for it in range(max_iters):
        # Try small random perturbations
        for trial in range(5):
            trial_circles = circles.copy()
            # Perturb positions
            idx = np.random.randint(n)
            trial_circles[idx, 0] += lr * np.random.randn()
            trial_circles[idx, 1] += lr * np.random.randn()
            # Try to grow radius
            trial_circles[idx, 2] += 0.005
            trial_circles = _clamp(trial_circles)
            
            # Check validity
            if _validity_check(trial_circles):
                sc = _score(trial_circles)
                if sc > best_score:
                    circles = trial_circles
                    best_score = sc
    
    return circles

def _hexagonal_initial(n):
    """Create hexagonal packing initial configuration."""
    # Hexagonal lattice in unit square
    # Spacing s, radius r = s/2
    # For hexagonal packing, density = pi/(2*sqrt(3)) ≈ 0.9069
    # We want to fit n circles
    
    # Try different spacings
    best_circles = None
    best_score = -1
    
    for s in np.linspace(0.05, 0.3, 50):
        r = s / 2
        # Hexagonal lattice: rows offset
        pos = []
        row = 0
        y = r
        while y <= 1 - r + 1e-9:
            x_start = r
            if row % 2 == 1:
                x_start = r + s/2
            x = x_start
            while x <= 1 - r + 1e-9:
                pos.append((x, y, r))
                x += s
            y += s * np.sqrt(3) / 2
            row += 1
        
        if len(pos) >= n:
            pos = pos[:n]
            circles = np.array(pos)
            if _validity_check(circles):
                sc = _score(circles)
                if sc > best_score:
                    best_score = sc
                    best_circles = circles
    
    if best_circles is not None:
        return best_circles
    
    # Fallback to grid
    side = int(np.ceil(np.sqrt(n)))
    r = 0.5 / side - 1e-3
    cells = [((i + 0.5) / side, (j + 0.5) / side, r) for i in range(side) for j in range(side)]
    return np.array(cells[:n])

def _random_initial(n, seed=None):
    """Random initial configuration."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    circles = np.zeros((n, 3))
    circles[:, 2] = 0.02  # Small initial radius
    circles[:, 0] = rng.uniform(0.05, 0.95, n)
    circles[:, 1] = rng.uniform(0.05, 0.95, n)
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_circles = None
    best_score = -1
    
    # Try multiple initializations
    initializations = []
    
    # 1. Hexagonal
    hex_circles = _hexagonal_initial(N)
    initializations.append(hex_circles)
    
    # 2. Grid
    side = int(np.ceil(np.sqrt(N)))
    r_grid = 0.5 / side - 1e-3
    grid_cells = [((i + 0.5) / side, (j + 0.5) / side, r_grid) for i in range(side) for j in range(side)]
    initializations.append(np.array(grid_cells[:N]))
    
    # 3. Random with different seeds
    for seed in [42, 123, 456]:
        initializations.append(_random_initial(N, seed=seed))
    
    for init_circles in initializations:
        if time.time() > deadline - 20:
            break
        
        # Relax and grow
        relaxed = _relax_and_grow(init_circles, max_iters=300)
        
        # Local search
        local = _local_search(relaxed, max_iters=500)
        
        sc = _score(local)
        if sc > best_score:
            best_score = sc
            best_circles = local.copy()
    
    if best_circles is None:
        # Fallback
        best_circles = _hexagonal_initial(N)
    
    return best_circles
# EVOLVE-BLOCK-END
