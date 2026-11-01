import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initialization: Concentric Rings
    # We want to place N points.
    # Ring k (k>=1) has 6k points. Ring 0 has 1 point.
    # Total points for K rings: 1 + 6 * (1 + 2 + ... + K) = 1 + 3K(K+1)
    # We need to distribute N points.
    
    points = []
    # Center point
    points.append([0.0, 0.0])
    remaining = N - 1
    
    k = 1
    radii = []
    # Estimate initial radii. The area of the circle is pi.
    # Approximate radius of ring k is proportional to sqrt(k).
    # Let's just assign them evenly in area initially.
    
    while remaining > 0:
        num_in_ring = min(6 * k, remaining)
        # If the last ring is partial, we still place them on a circle
        angles = np.linspace(0, 2 * np.pi, num_in_ring, endpoint=False)
        # Initial radius estimate: sqrt(k) * scale
        # Scale factor to fit in unit circle roughly
        # Max k will be around sqrt(N/3)
        max_k_est = np.sqrt(N / 3.0)
        r_est = np.sqrt(k) / max_k_est_est
        # Wait, I need to know max_k first.
        break # Let's do a cleaner init
        
    # Cleaner Init
    points = np.zeros((N, 2))
    # Assign indices to rings
    idx = 0
    points[idx] = [0, 0]
    idx += 1
    k = 1
    # Determine how many rings we need
    # 1 + 3K(K+1) >= N
    # 3K^2 ~ N => K ~ sqrt(N/3)
    K_max = int(np.ceil(np.sqrt(N / 3.0)))
    
    # Pre-calculate points per ring
    pts_per_ring = {}
    rem = N - 1
    k = 1
    while rem > 0:
        cap = 6 * k
        take = min(cap, rem)
        pts_per_ring[k] = take
        rem -= take
        k += 1
    
    max_k = k - 1
    
    # Generate points
    idx = 1
    for k in range(1, max_k + 1):
        num = pts_per_ring[k]
        # Angle offset to stagger rings (hexagonal packing style)
        offset = (k % 2) * (np.pi / num) if num > 0 else 0
        angles = np.linspace(0, 2 * np.pi, num, endpoint=False) + offset
        # Radius: distribute rings evenly in area
        # Area up to ring k is proportional to k^2
        # So radius r_k ~ sqrt(k)
        # Normalize so that r_max ~ 0.9
        r = 0.9 * np.sqrt(k) / np.sqrt(max_k)
        for a in angles:
            points[idx, 0] = r * np.cos(a)
            points[idx, 1] = r * np.sin(a)
            idx += 1
            
    # 2. Optimization
    # Objective: Maximize sum of inverse powers of distances
    # This pushes points apart.
    # Constraint: Points must stay within radius R_max = 1 - r_min
    
    p_power = 12  # Higher power focuses more on the minimum distance
    epsilon = 1e-6
    
    # Initial radius calculation
    def calc_radius(pts):
        # Pairwise distances
        # (N, N) matrix
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary distance
        norms = np.linalg.norm(pts, axis=1)
        min_bound_dist = np.min(1.0 - norms)
        
        return min(min_pair_dist / 2.0, min_bound_dist)
    
    current_r = calc_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    # Learning rate
    lr = 0.005
    lr_decay = 0.999
    
    steps = 0
    max_steps = 20000
    
    while steps < max_steps and time.time() < deadline:
        # Compute distances
        # To save memory/time, we can compute gradient analytically
        # Grad of sum_{i<j} d_ij^-p wrt x_i
        # d/dx_i ( (x_i - x_j)^2 )^-p/2 = -p/2 * ( (x_i - x_j)^2 )^(-p/2 - 1) * 2(x_i - x_j)
        # = -p * (x_i - x_j) * d_ij^(-p - 2)
        
        # Vectorized gradient calculation
        # diff[i, j] = x_i - x_j
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, 1.0) # Avoid div by 0, we'll mask later
        
        # Clip distances to avoid overflow
        dists_sq = np.maximum(dists_sq, epsilon ** 2)
        dists = np.sqrt(dists_sq)
        
        # Weight matrix W[i, j] = p * d_ij^(-p - 2)
        # We want sum_{j != i} p * (x_i - x_j) / d_ij^(p+2)
        
        # d_ij^(-p-2)
        inv_d_p2 = dists ** (-(p_power + 2))
        np.fill_diagonal(inv_d_p2, 0.0)
        
        # Gradient component for x_i
        # grad_x_i = sum_j p * (x_i - x_j) * inv_d_p2[i, j]
        # This is (p * inv_d_p2) @ (-diff) ?? No.
        # sum_j (x_i - x_j) * w_ij = x_i * sum_j w_ij - sum_j x_j w_ij
        # This is a matrix multiplication.
        
        W = p_power * inv_d_p2
        
        # grad_i = sum_j W[i, j] * (x_i - x_j)
        # Note: W is symmetric.
        # sum_j W[i, j] x_i = x_i * sum_j W[i, j]
        # sum_j W[i, j] x_j = (W @ x)_i
        
        sum_W = np.sum(W, axis=1)
        Wx = W @ points
        
        grad = points * sum_W[:, np.newaxis] - Wx
        
        # Update
        points += lr * grad
        
        # Projection: Keep points inside circle of radius R_bound
        # R_bound = 1 - current_r
        # Using current_r ensures we don't violate the current best minimum radius constraint
        # If a point goes out, we project it back.
        R_bound = 1.0 - current_r
        norms = np.linalg.norm(points, axis=1)
        mask = norms > R_bound
        if np.any(mask):
            # Scale down
            points[mask] *= (R_bound / norms[mask])[:, np.newaxis]
            
        # Calculate new radius
        new_r = calc_radius(points)
        
        if new_r > best_r:
            best_r = new_r
            best_points = points.copy()
            # If we improved, maybe increase LR slightly? No, standard decay is safer.
            # But if we are stuck, we might need noise.
            
        # Accept the move if it improves the "potential" or just always accept in gradient ascent?
        # Since we are maximizing a surrogate, and the surrogate is correlated with the true objective,
        # we should probably accept moves that increase the surrogate.
        # However, the projection can reduce the surrogate.
        # Let's stick to simple gradient ascent on the surrogate, but ensure we track the true best.
        
        # Decay learning rate
        lr *= lr_decay
        
        steps += 1
        
        # If stuck, add noise?
        # Let's check if progress has stalled.
        if steps % 1000 == 0:
            # Optional: Add small noise to escape local minima
            # points += np.random.normal(0, 0.001, points.shape)
            # This might break validity if not projected carefully.
            pass
            
    return best_points

# Helper to ensure valid output
def construct():
    # Wrap the logic above to ensure no exceptions
    try:
        return _construct_impl()
    except Exception as e:
        # Fallback to hexagonal lattice if optimization fails
        return _hex_lattice_fallback()

def _hex_lattice_fallback():
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

# Refine the main implementation to be self-contained and efficient
def _construct_impl():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Initialization: Concentric Rings
    points = np.zeros((N, 2))
    idx = 0
    points[idx] = [0, 0]
    idx += 1
    
    rem = N - 1
    k = 1
    pts_per_ring = {}
    while rem > 0:
        cap = 6 * k
        take = min(cap, rem)
        pts_per_ring[k] = take
        rem -= take
        k += 1
        
    max_k = k - 1
    
    idx = 1
    for k in range(1, max_k + 1):
        num = pts_per_ring[k]
        offset = (k % 2) * (np.pi / num) if num > 0 else 0
        angles = np.linspace(0, 2 * np.pi, num, endpoint=False) + offset
        r = 0.9 * np.sqrt(k) / np.sqrt(max_k)
        for a in angles:
            points[idx, 0] = r * np.cos(a)
            points[idx, 1] = r * np.sin(a)
            idx += 1
            
    # 2. Optimization
    p_power = 12
    epsilon = 1e-6
    
    def calc_radius(pts):
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        norms = np.linalg.norm(pts, axis=1)
        min_bound_dist = np.min(1.0 - norms)
        return min(min_pair_dist / 2.0, min_bound_dist)
    
    current_r = calc_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    lr = 0.005
    lr_decay = 0.999
    
    steps = 0
    max_steps = 20000
    
    while steps < max_steps and time.time() < deadline:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, 1.0)
        dists_sq = np.maximum(dists_sq, epsilon ** 2)
        dists = np.sqrt(dists_sq)
        
        inv_d_p2 = dists ** (-(p_power + 2))
        np.fill_diagonal(inv_d_p2, 0.0)
        
        W = p_power * inv_d_p2
        sum_W = np.sum(W, axis=1)
        Wx = W @ points
        
        grad = points * sum_W[:, np.newaxis] - Wx
        
        points += lr * grad
        
        R_bound = 1.0 - current_r
        norms = np.linalg.norm(points, axis=1)
        mask = norms > R_bound
        if np.any(mask):
            points[mask] *= (R_bound / norms[mask])[:, np.newaxis]
            
        new_r = calc_radius(points)
        
        if new_r > best_r:
            best_r = new_r
            best_points = points.copy()
            
        lr *= lr_decay
        steps += 1
        
    return best_points

# Final wrapper
def construct():
    try:
        return _construct_impl()
    except Exception:
        return _hex_lattice_fallback()
