"""Pack 200 equal circles in a unit square to maximize common radius."""
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import KDTree

N = 200
_TIME_LIMIT = 140  # seconds, leave margin before 150s kill


def _score(points):
    """Compute the radius for a set of points in [-0.5, 0.5]^2."""
    if not np.all(np.isfinite(points)):
        return 0.0
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 0.0
    if len(points) < 2:
        min_pair = float('inf')
    else:
        d = pdist(points)
        min_pair = np.min(d)
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    min_boundary = np.min(dist_boundary)
    return min(min_pair / 2.0, min_boundary)


def _initialize_packing():
    """Generate a high-quality initial hexagonal packing."""
    best_points = None
    best_score = 0.0
    
    # Try various row counts for hexagonal packing
    for rows in range(13, 21):
        cols = N // rows
        remainder = N % rows
        
        # Calculate spacing for hexagonal packing
        # Vertical spacing: s * sqrt(3) / 2
        # Horizontal spacing: s
        # We need rows * (s * sqrt(3) / 2) <= 1.0 and cols * s <= 1.0
        s_from_width = 1.0 / cols
        s_from_height = 1.0 / (rows * np.sqrt(3) / 2)
        s = min(s_from_width, s_from_height) * 0.98
        
        points = []
        for j in range(rows):
            offset = (j % 2) * s / 2
            # Determine how many points to place in this row
            if j < remainder:
                n_in_row = cols + 1
            else:
                n_in_row = cols
            
            row_width = (n_in_row - 1) * s
            start_x = -row_width / 2 + offset
            for i in range(n_in_row):
                x = start_x + i * s
                y = -0.5 + s * np.sqrt(3) / 4 + j * s * np.sqrt(3) / 2
                points.append([x, y])
        
        if len(points) != N:
            # Adjust if we don't have exactly N points
            continue
            
        points = np.array(points)
        
        # Center the whole set
        points[:, 0] -= np.mean(points[:, 0])
        points[:, 1] -= np.mean(points[:, 1])
        
        # Scale to fit in [-0.5, 0.5]
        max_abs = np.max(np.abs(points))
        if max_abs > 0.5:
            points *= 0.5 / max_abs
            
        sc = _score(points)
        if sc > best_score:
            best_score = sc
            best_points = points.copy()
            
    if best_points is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
        
    return best_points


def _simulated_annealing(points, deadline, max_iter=20000):
    """Simulated annealing to directly optimize the non-smooth radius objective."""
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Initial temperature based on current score
    t0 = 0.005
    t_end = 0.0001
    cooling_rate = (t_end / t0) ** (1.0 / max_iter)
    
    # Use KDTree for efficient neighbor queries
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Get current score and identify bottleneck
        d = pdist(pts)
        min_pair_dist = np.min(d)
        dist_boundary = 0.5 - np.max(np.abs(pts), axis=1)
        min_boundary = np.min(dist_boundary)
        current_score = min(min_pair_dist / 2.0, min_boundary)
        
        # Temperature
        T = t0 * (cooling_rate ** it)
        
        # Find closest pair
        min_pair_idx = np.argmin(d)
        # Convert condensed index to (i, j)
        n = len(pts)
        i = int(np.floor((-1 + np.sqrt(1 + 8 * min_pair_idx)) / 2))
        j = i + 1 + min_pair_idx - i * (i + 1) // 2
        
        # Find closest point to boundary
        wall_idx = np.argmin(dist_boundary)
        
        # Determine which constraint is the bottleneck
        if min_pair_dist / 2.0 < min_boundary:
            # Closest pair is bottleneck
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Random perturbation
                pts[i] += np.random.randn(2) * T
                pts[j] -= np.random.randn(2) * T
            else:
                # Push apart with some randomness
                push_dir = diff / dist
                push_mag = T * np.random.uniform(0.5, 1.5)
                pts[i] += push_dir * push_mag
                pts[j] -= push_dir * push_mag
                # Add small random component
                pts[i] += np.random.randn(2) * T * 0.3
                pts[j] -= np.random.randn(2) * T * 0.3
        else:
            # Wall is bottleneck
            # Move the point closest to wall inward
            for k in range(2):
                if pts[wall_idx, k] < -0.5:
                    pts[wall_idx, k] += T * np.random.uniform(0.5, 1.5)
                elif pts[wall_idx, k] > 0.5:
                    pts[wall_idx, k] -= T * np.random.uniform(0.5, 1.5)
                else:
                    # Move away from nearest wall
                    if abs(pts[wall_idx, k] + 0.5) < abs(pts[wall_idx, k] - 0.5):
                        pts[wall_idx, k] += T * np.random.uniform(0.5, 1.5)
                    else:
                        pts[wall_idx, k] -= T * np.random.uniform(0.5, 1.5)
            # Add small random component
            pts[wall_idx] += np.random.randn(2) * T * 0.3
        
        # Also perturb a random few points to escape local minima
        n_perturb = np.random.randint(1, 4)
        perturb_idx = np.random.choice(N, size=n_perturb, replace=False)
        for idx in perturb_idx:
            pts[idx] += np.random.randn(2) * T * 0.5
        
        # Clamp to valid region
        pts = np.clip(pts, -0.5, 0.5)
        
        # Evaluate new score
        d_new = pdist(pts)
        min_pair_new = np.min(d_new)
        dist_boundary_new = 0.5 - np.max(np.abs(pts), axis=1)
        min_boundary_new = np.min(dist_boundary_new)
        new_score = min(min_pair_new / 2.0, min_boundary_new)
        
        # Accept or reject based on Metropolis criterion
        delta = new_score - current_score
        if delta > 0:
            # Always accept improvement
            pass
        else:
            # Accept with probability exp(delta / T)
            if np.random.rand() < np.exp(delta / T):
                pass
            else:
                # Reject: revert to previous state
                pts = best_pts.copy() if new_score < best_score else pts
        
        # Update best
        if new_score > best_score:
            best_score = new_score
            best_pts = pts.copy()
    
    return best_pts, best_score


def _gradient_refine(points, deadline, max_iter=5000):
    """Gradient-based refinement using subgradient of the radius function."""
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    lr = 0.001
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute radius and identify active constraints
        d = pdist(pts)
        min_pair_dist = np.min(d)
        dist_boundary = 0.5 - np.max(np.abs(pts), axis=1)
        min_boundary = np.min(dist_boundary)
        current_score = min(min_pair_dist / 2.0, min_boundary)
        
        # Compute subgradient
        grad = np.zeros_like(pts)
        
        if min_pair_dist / 2.0 < min_boundary:
            # Closest pair constraint is active
            min_pair_idx = np.argmin(d)
            n = len(pts)
            i = int(np.floor((-1 + np.sqrt(1 + 8 * min_pair_idx)) / 2))
            j = i + 1 + min_pair_idx - i * (i + 1) // 2
            
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                grad[i] += diff / dist
                grad[j] -= diff / dist
        else:
            # Wall constraint is active
            wall_idx = np.argmin(dist_boundary)
            for k in range(2):
                if pts[wall_idx, k] < 0:
                    grad[wall_idx, k] += 1
                else:
                    grad[wall_idx, k] -= 1
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        # Update
        pts += lr * grad
        pts = np.clip(pts, -0.5, 0.5)
        
        # Evaluate
        d_new = pdist(pts)
        min_pair_new = np.min(d_new)
        dist_boundary_new = 0.5 - np.max(np.abs(pts), axis=1)
        min_boundary_new = np.min(dist_boundary_new)
        new_score = min(min_pair_new / 2.0, min_boundary_new)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = pts.copy()
        
        # Decay learning rate
        lr *= 0.999
    
    return best_pts, best_score


def _multi_phase_optimize(points, deadline):
    """Multi-phase optimization: SA + gradient refinement."""
    # Phase 1: Simulated annealing for global exploration
    sa_pts, sa_score = _simulated_annealing(points, deadline, max_iter=15000)
    
    # Phase 2: Gradient refinement for local polishing
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        grad_pts, grad_score = _gradient_refine(sa_pts, deadline, max_iter=5000)
        if grad_score > sa_score:
            return grad_pts, grad_score
    
    return sa_pts, sa_score


def construct():
    deadline = time.time() + _TIME_LIMIT
    
    # Phase 1: Initialize with hexagonal packing
    init_pts = _initialize_packing()
    best_points = init_pts.copy()
    best_score = _score(init_pts)
    
    # Phase 2: Multi-phase optimization
    opt_pts, opt_score = _multi_phase_optimize(best_points, deadline)
    if opt_score > best_score:
        best_points = opt_pts
        best_score = opt_score
        
    # Phase 3: Final polish if time permits
    if time.time() < deadline - 3:
        # Quick SA pass with lower temperature
        sa_pts, sa_score = _simulated_annealing(best_points, deadline, max_iter=5000)
        if sa_score > best_score:
            best_points = sa_pts
            best_score = sa_score
    
    return best_points
