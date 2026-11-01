"""Different strategy: simulated annealing with large perturbations + repair."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 200
HALF = 0.5


def compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > HALF + 1e-12):
        return 0.0

    min_dist_to_wall = np.min(HALF - np.abs(centers), axis=None)
    if min_dist_to_wall <= 0:
        return 0.0

    tree = cKDTree(centers)
    # Use a generous radius to catch all close pairs
    pairs = tree.query_pairs(r=0.08)
    if len(pairs) == 0:
        min_pair_dist = 1e9
    else:
        min_pair_dist = min(np.linalg.norm(centers[i] - centers[j]) for i, j in pairs)

    r = min(min_dist_to_wall, min_pair_dist / 2)
    return r


def push_apart(centers, r_target, max_rounds=30):
    """Iteratively push overlapping circles apart until no overlaps remain."""
    current = centers.copy()
    for _ in range(max_rounds):
        tree = cKDTree(current)
        pairs = tree.query_pairs(r=2 * r_target + 0.01)
        if len(pairs) == 0:
            break
        
        moved = False
        for i, j in pairs:
            diff = current[i] - current[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                current[i] += np.random.normal(0, 0.002, 2)
                current[j] -= np.random.normal(0, 0.002, 2)
                moved = True
                continue
            min_dist = 2 * r_target
            if dist < min_dist:
                push = (min_dist - dist) / 2 + 0.0003
                diff_norm = diff / dist
                current[i] += push * diff_norm
                current[j] -= push * diff_norm
                moved = True
        
        current = np.clip(current, -HALF + 1e-9, HALF - 1e-9)
        if not moved:
            break
    return current


def hex_init():
    """Generate a hexagonal lattice initialization."""
    best_points = None
    best_r = 0.0
    
    for scale in [0.072, 0.074, 0.076, 0.078, 0.080]:
        dx = scale
        dy = scale * np.sqrt(3) / 2
        
        points = []
        row = 0
        y = -HALF + dy / 2
        while y <= HALF:
            x_offset = dx / 2 if row % 2 == 1 else 0
            x = -HALF + x_offset
            while x <= HALF:
                if -HALF <= x <= HALF and -HALF <= y <= HALF:
                    points.append([x, y])
                x += dx
            y += dy
            row += 1
        
        if len(points) < N:
            continue
        
        points = np.array(points)
        # Take N closest to center
        dists = np.sum(points ** 2, axis=1)
        idx = np.argsort(dists)[:N]
        pts = points[idx]
        
        r = compute_radius(pts)
        if r > best_r:
            best_r = r
            best_points = pts.copy()
    
    if best_points is None:
        # Fallback grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
    
    return best_points


def simulate_annealing(centers, start_time, deadline):
    """Simulated annealing with large random perturbations and repair."""
    current = centers.copy()
    current_r = compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    # Annealing schedule: start with large moves, shrink over time
    n_steps = 0
    total_budget = deadline - start_time
    
    # Temperature schedule: geometric cooling
    T_start = 0.05
    T_end = 0.001
    cooling = 0.995  # per step
    
    T = T_start
    rng = np.random.RandomState(123)
    
    while T > T_end:
        if time.time() > deadline - 5:
            break
        
        n_steps += 1
        
        # Choose a perturbation size proportional to temperature
        # Large perturbations: move a random subset of points significantly
        n_move = rng.randint(5, 40)
        indices = rng.choice(N, n_move, replace=False)
        
        # Perturbation magnitude
        mag = T * rng.uniform(0.5, 1.5)
        
        new_centers = current.copy()
        for idx in indices:
            new_centers[idx] += rng.normal(0, mag, 2)
        
        # Clamp to boundary
        new_centers = np.clip(new_centers, -HALF + 1e-9, HALF - 1e-9)
        
        # Repair: push apart with current best radius as target
        # Use a slightly larger target to allow some flexibility
        target_r = max(best_r, current_r)
        repaired = push_apart(new_centers, target_r, max_rounds=15)
        
        new_r = compute_radius(repaired)
        
        # Acceptance criterion
        if new_r >= current_r:
            # Accept improvement
            current = repaired
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = current.copy()
        else:
            # Metropolis acceptance: accept with probability exp((new_r - current_r) / T)
            delta_r = new_r - current_r
            # Scale: the relevant scale for acceptance
            scale = 0.001
            prob = np.exp(delta_r / (T * scale))
            if rng.random() < prob:
                current = repaired
                current_r = new_r
        
        # Cool down
        T *= cooling
        
        # Periodically try to increase the target radius
        if n_steps % 50 == 0:
            # Try pushing with a slightly larger radius
            trial_r = best_r * 1.002
            trial = push_apart(best, trial_r, max_rounds=20)
            trial_r_actual = compute_radius(trial)
            if trial_r_actual > best_r:
                best_r = trial_r_actual
                best = trial.copy()
                current = trial.copy()
                current_r = trial_r_actual
    
    return best, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    
    # Step 1: Hexagonal initialization
    best_config = hex_init()
    best_r = compute_radius(best_config)
    
    # Step 2: Initial repair to resolve overlaps
    if time.time() < deadline - 10:
        repaired = push_apart(best_config, best_r, max_rounds=30)
        repaired_r = compute_radius(repaired)
        if repaired_r > best_r:
            best_r = repaired_r
            best_config = repaired
    
    # Step 3: Simulated annealing with large perturbations
    if time.time() < deadline - 5:
        sa_best, sa_r = simulate_annealing(best_config, time.time(), deadline)
        if sa_r > best_r:
            best_r = sa_r
            best_config = sa_best
    
    # Step 4: Final fine-tuning push
    if time.time() < deadline - 2:
        # Try to push with slightly larger radius
        for scale in [1.001, 1.002, 1.003, 1.005]:
            trial_r = best_r * scale
            trial = push_apart(best_config, trial_r, max_rounds=25)
            trial_r_actual = compute_radius(trial)
            if trial_r_actual > best_r:
                best_r = trial_r_actual
                best_config = trial.copy()
    
    # Final validation
    final_r = compute_radius(best_config)
    if final_r <= 0:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_config = grid[:N]
    
    return best_config
