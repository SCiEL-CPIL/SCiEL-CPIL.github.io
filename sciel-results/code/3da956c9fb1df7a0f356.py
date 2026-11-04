"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import KDTree


def _compute_radius(centers):
    """Compute the largest common radius for given centers in [-0.5, 0.5]^3."""
    if len(centers) == 0:
        return 0.0
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    # Pairwise distances
    tree = KDTree(centers)
    pairs = tree.query_pairs(1.0)  # all pairs within 1.0 (all pairs since max dist ~ sqrt(3)/2 < 1)
    if len(pairs) > 0:
        pair_dists = np.array([np.linalg.norm(centers[i] - centers[j]) for i, j in pairs])
        min_pair = np.min(pair_dists) / 2.0
    else:
        min_pair = 0.5
    return min(min_wall, min_pair)


def _fcc_init(n_points):
    """Generate FCC lattice points within the unit cube, then select n_points."""
    # FCC lattice: points at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) + integer translations
    # We'll generate a grid of FCC sites and keep those inside [-0.5, 0.5]^3
    points = []
    # Generate enough FCC sites
    for ix in range(-2, 4):
        for iy in range(-2, 4):
            for iz in range(-2, 4):
                # FCC basis
                for bx, by, bz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    x = ix + bx - 0.5
                    y = iy + by - 0.5
                    z = iz + bz - 0.5
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        points.append([x, y, z])
    
    points = np.array(points)
    
    # We need exactly n_points. If we have more, we need to select a subset.
    # For now, if we have at least n_points, randomly select (with fixed seed for reproducibility)
    if len(points) >= n_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), size=n_points, replace=False)
        return points[idx]
    else:
        # Pad with random points if not enough
        rng = np.random.default_rng(42)
        extra = rng.uniform(-0.45, 0.45, size=(n_points - len(points), 3))
        return np.vstack([points, extra])


def _sa_optimize(centers, time_budget=140.0, seed=42):
    """Simulated annealing optimization."""
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    # Annealing schedule
    T_start = 0.05
    T_end = 0.001
    n_iters_max = 500000
    
    for it in range(n_iters_max):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Temperature
        frac = elapsed / time_budget
        T = T_start * (T_end / T_start) ** frac
        
        # Pick a random sphere and move it
        i = rng.integers(0, N)
        old_pos = current[i].copy()
        
        # Random displacement
        step_scale = 0.05 * (T / T_start) ** 0.5
        new_pos = old_pos + rng.normal(0, step_scale, size=3)
        
        # Keep inside cube
        new_pos = np.clip(new_pos, -0.499, 0.499)
        
        # Compute new radius
        current[i] = new_pos
        new_r = _compute_radius(current)
        
        # Accept or reject
        if new_r >= current_r or rng.random() < np.exp((new_r - current_r) / T):
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = current.copy()
        else:
            current[i] = old_pos
    
    return best, best_r


def _gradient_step(centers, r, lr=0.01, n_steps=100):
    """Simple gradient ascent using pairwise repulsion and wall attraction."""
    centers = centers.copy()
    for step in range(n_steps):
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Wall forces: push inward if close to wall
        for i in range(N):
            for d in range(3):
                dist_to_wall = 0.5 - abs(centers[i, d])
                if dist_to_wall < r:
                    # Push away from wall
                    sign = 1 if centers[i, d] > 0 else -1
                    forces[i, d] += sign * (r - dist_to_wall)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 2 * r and dist > 1e-10:
                    force_mag = (2 * r - dist) / dist
                    forces[i] += force_mag * diff
                    forces[j] -= force_mag * diff
                elif dist <= 1e-10:
                    # Coincident, push apart randomly
                    forces[i] += rng.normal(0, 0.01, 3)
                    forces[j] -= rng.normal(0, 0.01, 3)
        
        # Update
        centers += lr * forces
        # Clip to cube
        centers = np.clip(centers, -0.499, 0.499)
    
    return centers


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    # Initialize with FCC lattice
    centers = _fcc_init(N)
    current_r = _compute_radius(centers)
    best = centers.copy()
    best_r = current_r
    
    # Phase 1: Gradient steps to improve initial configuration
    if time.time() < deadline - 30:
        for grad_round in range(5):
            centers = _gradient_step(centers, best_r, lr=0.005, n_steps=50)
            r = _compute_radius(centers)
            if r > best_r:
                best_r = r
                best = centers.copy()
    
    # Phase 2: Simulated annealing
    if time.time() < deadline - 10:
        sa_time_budget = deadline - time.time() - 5
        sa_centers, sa_r = _sa_optimize(best, time_budget=sa_time_budget, seed=123)
        if sa_r > best_r:
            best_r = sa_r
            best = sa_centers
    
    # Phase 3: Final gradient polish
    if time.time() < deadline - 5:
        polish_time = deadline - time.time() - 3
        n_polish_steps = max(1, int(polish_time * 100))
        for _ in range(min(n_polish_steps, 200)):
            best = _gradient_step(best, best_r, lr=0.001, n_steps=1)
            r = _compute_radius(best)
            if r > best_r:
                best_r = r
    
    # Final validation
    final_r = _compute_radius(best)
    if final_r > best_r:
        best_r = final_r
    
    # Ensure all values are finite and within bounds
    best = np.clip(best, -0.4999, 0.4999)
    best = np.nan_to_num(best, nan=0.0, posinf=0.4999, neginf=-0.4999)
    
    return best
# EVOLVE-BLOCK-END
