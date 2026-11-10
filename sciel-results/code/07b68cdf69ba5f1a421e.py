"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for a set of circle centers."""
    # Distance to boundary
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary)
    
    if min_boundary <= 0:
        return 0.0
    
    # Pairwise distances using cdist for efficiency
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)


def _hex_init(n):
    """Create a hexagonal grid initialization with multiple configurations."""
    best_centers = None
    best_score = -1
    
    # Try different row/column configurations for hex grid
    for rows in range(4, 14):
        cols = int(np.ceil(n / rows))
        if cols < 3 or cols > 15:
            continue
        
        # Hex spacing
        dx = 1.0 / (cols + 0.5)
        dy = dx * np.sqrt(3) / 2
        
        # Center the grid
        x_start = -0.5 + dx / 2
        y_start = -0.5 + dy / 2
        
        centers = []
        for i in range(rows):
            for j in range(cols):
                if len(centers) >= n:
                    break
                x = x_start + j * dx
                y = y_start + i * dy
                # Offset odd rows
                if i % 2 == 1:
                    x += dx / 2
                centers.append([x, y])
        
        if len(centers) >= n:
            centers = np.array(centers[:n])
            # Clip to ensure within bounds
            centers = np.clip(centers, -0.5, 0.5)
            s = _compute_radius(centers)
            if s > best_score:
                best_score = s
                best_centers = centers
    
    if best_centers is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:n]
    
    return best_centers


def _local_search(centers, max_iter=2000, step_init=0.002):
    """Efficient local search using coordinate descent with adaptive steps."""
    current = centers.copy()
    current_score = _compute_radius(current)
    best = current.copy()
    best_score = current_score
    
    step = step_init
    no_improve = 0
    
    for iteration in range(max_iter):
        improved = False
        
        # Try moving each center in small steps
        for i in range(N):
            # Try multiple directions with decreasing step size
            for dx in [step, -step, 0]:
                for dy in [step, -step, 0]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_c = current.copy()
                    new_c[i] = current[i] + np.array([dx, dy])
                    
                    # Keep within bounds
                    if np.any(np.abs(new_c[i]) > 0.5):
                        continue
                    
                    new_score = _compute_radius(new_c)
                    
                    if new_score > current_score + 1e-12:
                        current = new_c
                        current_score = new_score
                        improved = True
                        
                        if current_score > best_score:
                            best = current.copy()
                            best_score = current_score
                        
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            step *= 0.7
            if step < 1e-6:
                break
    
    return best, best_score


def _nelder_mead_polish(centers, time_budget=30.0):
    """Use Nelder-Mead to polish the solution by maximizing the minimum distance."""
    start_time = time.time()
    
    def objective(x_flat):
        if time.time() - start_time > time_budget:
            return 0.0
        centers_2d = x_flat.reshape(-1, 2)
        # We want to maximize the radius, so minimize negative radius
        r = _compute_radius(centers_2d)
        return -r
    
    x0 = centers.flatten()
    
    # Use Nelder-Mead with reasonable parameters
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={
            'maxiter': 500,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
    )
    
    if result.success or result.fun < -_compute_radius(centers):
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    else:
        return centers


def _sa_improve(centers, time_budget=50.0):
    """Simulated annealing with improved efficiency."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    current = centers.copy()
    current_score = _compute_radius(current)
    best = current.copy()
    best_score = current_score
    
    T = 0.005
    step_size = 0.003
    cooling = 0.9995
    
    while T > 1e-8:
        if time.time() > deadline - 2:
            break
        
        # Batch of moves
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            
            idx = np.random.randint(N)
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
            
            new_center = current.copy()
            new_center[idx] += [dx, dy]
            new_center[idx] = np.clip(new_center[idx], -0.5, 0.5)
            
            new_score = _compute_radius(new_center)
            
            # Accept if better or with probability exp(delta/T)
            if new_score >= current_score:
                current = new_center
                current_score = new_score
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
            else:
                delta = new_score - current_score
                if np.random.random() < np.exp(delta / T):
                    current = new_center
                    current_score = new_score
        
        T *= cooling
        # Adapt step size
        if T < 0.001:
            step_size = 0.001
        elif T < 0.005:
            step_size = 0.002
    
    return best, best_score


def construct():
    start_time = time.time()
    total_time_limit = 135.0  # Leave 15s margin for safety
    
    np.random.seed(12345)
    
    # Generate multiple initial configurations
    best_centers = None
    best_score = -1
    
    # Hex init
    centers = _hex_init(N)
    score = _compute_radius(centers)
    if score > best_score:
        best_centers = centers.copy()
        best_score = score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # Run local search on the best initial
    if remaining > 20:
        polished, pol_score = _local_search(centers, max_iter=1500, step_init=0.0015)
        if pol_score > best_score:
            best_centers = polished
            best_score = pol_score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # Nelder-Mead polish
    if remaining > 25:
        nm_polished = _nelder_mead_polish(best_centers, time_budget=min(20, remaining - 5))
        nm_score = _compute_radius(nm_polished)
        if nm_score > best_score:
            best_centers = nm_polished
            best_score = nm_score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # SA improvement
    if remaining > 30:
        sa_best, sa_score = _sa_improve(best_centers, time_budget=min(40, remaining - 5))
        if sa_score > best_score:
            best_centers = sa_best
            best_score = sa_score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # Final local search polish
    if remaining > 10:
        final_polished, final_score = _local_search(best_centers, max_iter=1000, step_init=0.001)
        if final_score > best_score:
            best_centers = final_polished
            best_score = final_score
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    # Final validation check
    final_r = _compute_radius(best_centers)
    if final_r <= 0 or not np.all(np.isfinite(best_centers)):
        # Fallback to hex init
        best_centers = _hex_init(N)
    
    return best_centers
# EVOLVE-BLOCK-END
