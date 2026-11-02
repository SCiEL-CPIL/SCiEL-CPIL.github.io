import numpy as np
from scipy.optimize import minimize
import time
import math

N = 300

def _hex_lattice_init(n_points, rotation=0.0):
    """Generate a hexagonal lattice of points inside the unit circle."""
    # Estimate spacing for n_points in area pi
    # Area per point in hex lattice ~ (sqrt(3)/2) * s^2
    # n * (sqrt(3)/2) * s^2 = pi  =>  s = sqrt(2*pi / (n*sqrt(3)))
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    
    # We need to generate a grid that covers the unit circle
    # Max range is roughly 1.0 / s
    k = int(1.0 / s) + 2
    
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            # Hex lattice coordinates
            # x = (i + 0.5 * (j % 2)) * s
            # y = j * s * sqrt(3) / 2
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) < n_points:
        # If we don't have enough points, add random ones (should not happen with k large enough)
        while len(points) < n_points:
            r = np.sqrt(np.random.random())
            a = np.random.random() * 2 * np.pi
            points = np.vstack([points, [[r * np.cos(a), r * np.sin(a)]]])
    else:
        # If we have too many, pick the closest to center or random sample
        # Sorting by distance to center ensures we fill the core first
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        points = points[indices[:n_points]]
        
    # Apply rotation
    if rotation != 0.0:
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        points = points @ R.T
        
    # Add small noise to break symmetry
    points += np.random.normal(0, s * 0.01, size=points.shape)
    
    return points

def _radial_ring_init(n_points):
    """
    Initialize points in concentric rings.
    This is often a better starting point for circle packing than hex lattice
    because it respects the circular boundary more naturally in the outer layers.
    """
    points = [[0.0, 0.0]] # Center point
    remaining = n_points - 1
    ring_radius = 0.0
    # Approximate spacing
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    
    # We will build rings until we have enough points
    # Number of points in ring k is approx 6k (hexagonal symmetry)
    # Or simply distribute remaining points into rings of increasing radius
    
    # A simple strategy:
    # 1. Estimate total area per point.
    # 2. Determine radii of rings.
    
    # Let's try a deterministic ring placement
    # Ring 0: 1 point (center)
    # Ring 1: 6 points
    # Ring 2: 12 points
    # ...
    # This is specific to hexagonal packing. For general N, we might need to adjust.
    
    # Alternative: Randomized radial initialization
    # Sort random points by radius and assign to rings?
    
    # Let's stick to a simple radial distribution
    # Divide area into rings
    # Area of ring i is proportional to (i+1)^2 - i^2 = 2i+1
    # Number of points in ring i proportional to area?
    
    # Simpler: Just place points at random angles with radii chosen to avoid overlap initially.
    # But we want a structured start.
    
    # Let's use a simple "spiral" or "rings" approach.
    # For N=300, let's just use Hex Lattice with random rotation and noise, 
    # but the inspiration says _radial_ring_init exists.
    
    # I will implement a basic radial ring initializer.
    # Assume rings with increasing number of points.
    # 1, 6, 12, 18, ...
    
    counts = [1]
    total = 1
    k = 1
    while total < n_points:
        # Add points to next ring
        # Usually 6*k points
        add = 6 * k
        if total + add > n_points:
            add = n_points - total
        counts.append(add)
        total += add
        k += 1
        
    # Trim if necessary
    while sum(counts) > n_points:
        # Remove from last ring
        last = counts[-1]
        if last > 0:
            counts[-1] -= 1
        else:
            counts.pop()
            
    # Now assign radii
    # Total area is pi.
    # Area of ring i is proportional to count_i? No, area of ring is proportional to r^2.
    # If we assume uniform density, area per point is constant.
    # Area of ring i (from r_{i-1} to r_i) = pi * r_i^2 - pi * r_{i-1}^2.
    # Let A_i be area allocated to ring i. A_i = count_i * (pi / n_points).
    # r_i^2 = r_{i-1}^2 + A_i / pi = r_{i-1}^2 + count_i / n_points.
    
    radii = []
    r_sq = 0.0
    for c in counts:
        r_sq += c / n_points
        radii.append(np.sqrt(r_sq))
        
    # Adjust outer radius to be <= 1.0
    # If r_max > 1, scale down? Or just clip?
    # Scaling down might cause overlaps.
    # Let's just generate them and rely on optimization to fix overlaps.
    # However, for a good start, we want them inside.
    # If r_max > 1, we can scale the whole configuration by 1/r_max.
    scale = 1.0
    if radii and radii[-1] > 1.0:
        scale = 1.0 / radii[-1]
        
    points = []
    for i, c in enumerate(counts):
        if c == 0: continue
        r = radii[i] * scale
        # Add a small random phase shift for each ring to break symmetry
        phase = np.random.random() * 2 * np.pi
        for j in range(c):
            theta = 2 * np.pi * j / c + phase
            points.append([r * np.cos(theta), r * np.sin(theta)])
            
    points = np.array(points)
    
    # Add small noise
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    points += np.random.normal(0, s * 0.05, size=points.shape)
    
    # Ensure inside unit circle
    norms = np.linalg.norm(points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        points[mask] = points[mask] * (0.999 / norms[mask])[:, np.newaxis]
        
    return points

def _compute_score(points):
    """Compute the packing radius r = min(min_pair_dist/2, min_wall_dist)."""
    # Wall distance
    dists_center = np.linalg.norm(points, axis=1)
    min_wall = 1.0 - np.max(dists_center)
    
    # Pairwise distances
    # For N=300, full matrix is 300x300, which is small.
    D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    np.fill_diagonal(L, np.inf)
    min_pair = np.min(L) / 2.0
    
    return min(min_pair, min_wall)

def _sa_pack(points, t_start=1.0, t_end=1e-5, alpha=0.9995, time_limit=100.0):
    """Simulated Annealing for circle packing."""
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    # Pre-compute some stats
    # We need to update the minimum distance efficiently.
    # For simplicity and speed in Python, we will re-calculate the min distance
    # involving the moved point.
    
    t = t_start
    best_points = current_points.copy()
    best_score = current_score
    
    start_time = time.time()
    
    # Estimate max step size based on initial spacing
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    while t > t_end:
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random point to move
        i = np.random.randint(N)
        
        # Generate a random move
        # Step size scales with temperature
        step_size = s * np.sqrt(t / t_start) * 2.0
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        
        new_points = current_points.copy()
        new_points[i] += [dx, dy]
        
        # Check if point is outside unit circle
        # If it is, the wall distance becomes negative or small.
        # We can just calculate the new score.
        
        # Efficient score update:
        # Only distances involving point i change.
        # Old min_pair might have been (i, j).
        # New min_pair is min(old_min_pair_except_i, new_dists_from_i)
        
        # To keep it simple and robust, we just recalculate the full score
        # if N is small enough. N=300 -> 45k pairs. NumPy is fast.
        # Let's try to be a bit smarter:
        
        # 1. Update wall distance for i
        new_dist_center_i = np.linalg.norm(new_points[i])
        new_min_wall = 1.0 - new_dist_center_i
        
        # 2. Update pair distances involving i
        # Distances from i to all others
        diffs = new_points - new_points[i]
        dists_i = np.linalg.norm(diffs, axis=1)
        dists_i[i] = np.inf
        new_min_pair_involving_i = np.min(dists_i) / 2.0
        
        # 3. We need the global min pair distance.
        # We don't track the second min easily.
        # Given N=300, let's just compute the full min pair dist.
        # It's ~0.1ms in NumPy.
        D = new_points[:, np.newaxis, :] - new_points[np.newaxis, :, :]
        L = np.linalg.norm(D, axis=2)
        np.fill_diagonal(L, np.inf)
        new_min_pair_global = np.min(L) / 2.0
        
        # 4. Global min wall dist
        dists_center_all = np.linalg.norm(new_points, axis=1)
        new_min_wall_global = 1.0 - np.max(dists_center_all)
        
        new_score = min(new_min_pair_global, new_min_wall_global)
        
        delta = new_score - current_score
        
        if delta > 0:
            # Accept
            current_points = new_points
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        else:
            # Metropolis criterion
            if np.random.random() < np.exp(delta / t):
                current_points = new_points
                current_score = new_score
        
        t *= alpha
        
    return best_points, best_score

def _soft_min_obj_and_grad(x, beta):
    """
    Compute the negative soft-min value and its gradient.
    """
    C = x.reshape(N, 2)
    
    # Pairwise distances
    D = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    
    L_safe = np.where(L > 1e-12, L, 1.0)
    x_pairs = L_safe / 2.0
    
    # Wall distances
    L_wall = np.linalg.norm(C, axis=1)
    x_walls = 1.0 - L_wall
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    x_pairs_flat = x_pairs[i_idx, j_idx]
    
    x_all = np.concatenate([x_pairs_flat, x_walls])
    
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    P = exp_terms / Z
    
    val = min_x - (1.0 / beta) * np.log(Z)
    
    grad = np.zeros_like(C)
    
    P_pairs = P[:len(i_idx)]
    
    D_flat = D[i_idx, j_idx]
    L_flat = L[i_idx, j_idx]
    L_flat_safe = np.where(L_flat > 1e-12, L_flat, 1.0)
    
    U = D_flat / L_flat_safe[:, np.newaxis]
    
    W = P_pairs * 0.5
    
    np.add.at(grad, i_idx, W[:, np.newaxis] * U)
    np.add.at(grad, j_idx, -W[:, np.newaxis] * U)
    
    P_walls = P[len(i_idx):]
    
    L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)
    U_walls = -C / L_wall_safe[:, np.newaxis]
    
    grad += P_walls[:, np.newaxis] * U_walls
    
    return -val, -grad

def _polish_lbfgs(points, beta, max_iter=200, time_limit=10.0):
    """Run L-BFGS-B optimization with a time limit."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        return _soft_min_obj_and_grad(x, beta)
    
    try:
        res = minimize(obj_and_grad, points.flatten(), method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-12})
        if res.success or res.status in [1, 2, 4]:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return points

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin before 150s kill
    
    # 1. Initialize with Radial Rings (better for circular boundary)
    # Fallback to Hex if Radial fails or gives bad score
    np.random.seed(42) # Deterministic for reproducibility in this run
    init_points = _radial_ring_init(N)
    
    # Ensure initial validity (push inside if needed)
    norms = np.linalg.norm(init_points, axis=1)
    mask = norms > 0.999
    if np.any(mask):
        init_points[mask] = init_points[mask] * (0.999 / norms[mask])[:, np.newaxis]
        
    # 2. Optimization Phase
    # We replace SA with a sequence of L-BFGS optimizations with increasing beta.
    # This is faster and often more effective for this problem size.
    
    current_points = init_points.copy()
    best_score = _compute_score(current_points)
    best_points = current_points.copy()
    
    # Annealing schedule for beta
    # Start with low beta (smooth) to resolve overlaps, increase to sharpen
    betas = [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 10:
            break
            
        # Allocate time based on remaining budget
        remaining_time = time_limit - 5 - (time.time() - start_time)
        # Give more time to earlier, smoother stages? Or later, sharper stages?
        # Later stages need precision.
        # Let's give equal time to each, or slightly more to the last few.
        time_step = min(10.0, remaining_time / (len(betas) - betas.index(beta)))
        
        current_points = _polish_lbfgs(current_points, beta, max_iter=200, time_limit=time_step)
        
        score = _compute_score(current_points)
        if score > best_score:
            best_score = score
            best_points = current_points.copy()
            
    # Final check
    final_score = _compute_score(best_points)
    if final_score > best_score:
        best_score = final_score
        
    # Ensure all points are inside the unit circle
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            best_points[mask] = best_points[mask] * (0.999999 / norms[mask])[:, np.newaxis]
            
    return best_points
