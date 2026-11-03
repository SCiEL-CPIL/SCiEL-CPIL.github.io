import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    best_points = None
    best_radius = 0.0

    # Helper to compute radius
    def get_radius(pts):
        # pts shape (N, 2)
        # Boundary check
        min_boundary = 0.5 - np.max(np.abs(pts), axis=1)
        min_b = np.min(min_boundary)
        
        # Pairwise distances
        # d_ij = norm(pts_i - pts_j)
        # We want min d_ij / 2
        # Vectorized:
        # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # dists = np.sqrt(np.sum(diff**2, axis=-1))
        # min_d = np.min(dists[dists > 1e-9]) # Exclude self
        
        # More efficient:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        # Set diagonal to inf to ignore self-distance
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_b, min_pair)

    def optimize(pts, max_time=10.0):
        pts = pts.copy()
        r = get_radius(pts)
        last_improve = time.time()
        step_size = 0.001
        noise = 0.001
        
        # Gradient ascent on the minimum radius
        # The minimum radius is determined by the "tightest" constraint.
        # We push against constraints that are active (distance <= 2r + eps) or boundary <= r + eps
        
        while time.time() < start_time + max_time:
            if time.time() > last_improve + 5.0:
                # Stagnation: add noise or break
                noise = max(noise * 0.9, 0.0001)
                if noise < 0.0001:
                    break
                last_improve = time.time()
            
            # Calculate distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            
            # Boundary distances
            dists_b = 0.5 - np.abs(pts)
            
            # Determine active constraints
            # We want to increase r.
            # If dists[i,j] < 2*r, push i and j apart.
            # If dists_b[i,k] < r, push i towards center.
            
            # For efficiency, we only consider constraints that are "close" to the current minimum
            # or we can just sum forces from all violations.
            
            # Let's use a "push" force for all pairs closer than 2*r * 1.05
            threshold = 2.0 * r * 1.05
            forces = np.zeros_like(pts)
            
            # Pairwise forces
            # dists < threshold
            mask = dists < threshold
            # We need to sum (diff / dists) for each i
            # diff is (N, N, 2)
            # We want sum over j for each i
            # force_i = sum_{j != i, d_ij < thresh} (pts_i - pts_j) / d_ij
            # Note: diff[i,j] = pts_i - pts_j
            
            # To avoid division by zero, dists is inf on diagonal.
            # Masked sum
            # inv_dists = np.where(mask, 1.0/dists, 0.0)
            # forces = np.sum(diff * inv_dists[..., np.newaxis], axis=1)
            
            # This is O(N^2) memory and compute. N=200 -> 40k elements. Fine.
            inv_dists = np.zeros_like(dists)
            inv_dists[mask] = 1.0 / dists[mask]
            forces += np.sum(diff * inv_dists[..., np.newaxis], axis=1)
            
            # Boundary forces
            # If dists_b[i,k] < r * 1.05, push towards center (negative of pts_i[k])
            # dists_b is (N, 2)
            mask_b = dists_b < r * 1.05
            # force = -pts * mask_b / dists_b (approx)
            # Actually, gradient of (0.5 - |x|) is -sign(x).
            # We want to move in direction that increases distance to boundary.
            # If x > 0, move left (negative). If x < 0, move right (positive).
            # So force is -sign(pts).
            # Weighted by how close it is?
            # Let's just add a constant push if violated.
            forces[mask_b] -= np.sign(pts[mask_b])
            
            # Normalize forces?
            # forces /= (np.linalg.norm(forces, axis=1, keepdims=True) + 1e-9)
            
            # Update
            pts += step_size * forces
            pts += noise * np.random.randn(*pts.shape)
            
            # Keep inside square (hard constraint for validity, though we push back)
            pts = np.clip(pts, -0.49, 0.49)
            
            new_r = get_radius(pts)
            if new_r > r:
                r = new_r
                last_improve = time.time()
                step_size *= 0.99
                noise *= 0.98
            else:
                # Revert or keep? 
                # Since we are doing gradient ascent on a non-smooth function, 
                # small steps might decrease r due to noise.
                # If we revert, we might get stuck. 
                # Let's keep the point but reduce step size if we stagnate.
                step_size *= 0.95
                noise *= 0.95

            if time.time() > deadline:
                break
                
        return pts, r

    # Generate Hexagonal Lattice Initializations
    # Hexagonal packing is dense.
    # We need 200 points.
    # Approx side length of square for hex grid:
    # Area per point ~ 1/sqrt(3) * (2r)^2 ? No, just fit in unit square.
    
    def generate_hex_init(jitter):
        # Create a hex grid covering the square
        # Hex lattice vectors: (1, 0), (0.5, sqrt(3)/2)
        # Scale to fit in [-0.5, 0.5]
        
        # Number of rows/cols
        # Let's try to fit as many as possible, then select 200 closest to center?
        # Or just generate a grid of ~200 points.
        
        # Simpler: Generate a grid of points in a hex pattern within the square.
        # Spacing s.
        # Number of points ~ Area / (s^2 * sqrt(3)/2) = 1 / (s^2 * 0.866)
        # 200 = 1 / (s^2 * 0.866) => s^2 = 1 / (200 * 0.866) => s = 1/sqrt(173.2) = 0.0758
        
        s = 0.075
        points = []
        y = -0.5
        row = 0
        while y <= 0.5:
            x_offset = (row % 2) * (s * 0.5)
            x = -0.5 + x_offset
            while x <= 0.5:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += s
            y += s * np.sqrt(3) / 2
            row += 1
        
        points = np.array(points)
        
        # If we have more than 200, randomly sample 200? 
        # Or take the first 200? 
        # Random sampling might break symmetry. 
        # Let's sort by distance to center and take 200 closest to ensure compactness.
        if len(points) > N:
            dists_to_center = np.sum(points**2, axis=1)
            indices = np.argsort(dists_to_center)[:N]
            points = points[indices]
        elif len(points) < N:
            # Pad with random points if grid was too sparse
            while len(points) < N:
                points = np.vstack([points, np.random.uniform(-0.4, 0.4, 2)])
        
        # Add jitter
        points += jitter * np.random.randn(*points.shape)
        
        # Clip to square
        points = np.clip(points, -0.49, 0.49)
        
        return points

    # Try multiple random initializations
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Different jitter levels
        jitter = 0.01 + 0.02 * (i % 5)
        init_pts = generate_hex_init(jitter)
        
        # Optimize
        opt_pts, r = optimize(init_pts, max_time=4.0)
        
        if r > best_radius:
            best_radius = r
            best_points = opt_pts
            # If we found a very good radius, we can reduce restarts
            if best_radius > 0.036:
                 # Continue to see if we can beat it, but maybe fewer restarts?
                 pass

    if best_points is None:
        # Fallback to grid if something went wrong
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]

    return best_points
