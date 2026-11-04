import numpy as np

N = 300

def construct():
    # 1. Initialize with concentric rings
    # We need to choose the number of rings and points per ring.
    # A standard hexagonal-like ring structure has 1, 6, 12, 18...
    # Sum 1 + 6*sum(1..m) = 1 + 3m(m+1)
    # For m=9, sum=271. We have 29 points left.
    # Let's create 10 rings.
    # Ring 0: 1 point
    # Ring 1: 6 points
    # ...
    # Ring 9: 54 points
    # Total 271.
    # Add 29 points to Ring 10? Or distribute?
    # Let's just use a simple distribution:
    # We want roughly equal angular spacing.
    
    points = []
    
    # Center
    points.append([0.0, 0.0])
    
    # Define ring radii and counts
    # We will optimize these later, but need a start.
    # Let's aim for 10 rings.
    # Counts: 1, 6, 12, 18, 24, 30, 36, 42, 48, 43 (sum=260? No)
    # 1+6+12+18+24+30+36+42+48 = 217.
    # 300 - 217 = 83.
    # This is getting complicated. Let's just use a grid of rings.
    
    # Better Init:
    # Use the "Lai et al" known structure approximation? 
    # No, we must generate it.
    
    # Let's use a spiral or grid mapping?
    # Actually, a very effective simple init is a hexagonal grid 
    # projected onto the circle, but "ring" init is better for local search 
    # because the topology is already close to optimal (central symmetry).
    
    # Let's try to fit 300 points into rings with counts:
    # 1, 6, 12, 18, 24, 30, 36, 42, 48, 43?
    # Sum = 1 + 6*sum(1..9) + 43 = 1 + 270 + 43 = 314. Too many.
    # Remove some from outer rings?
    
    # Let's use a deterministic distribution:
    # k rings.
    # points in ring k is proportional to k.
    # n_k = c * k.
    # sum c*k = 300.
    # c * m(m+1)/2 = 300.
    # If m=10, c * 55 = 300 => c = 5.45.
    # n_1 = 5.45 -> 5
    # n_2 = 10.9 -> 11
    # ...
    
    # Let's just hardcode a good ring distribution for 300.
    # From literature, optimal packings often have rings with counts:
    # 1, 6, 12, 18, 24, 30, 36, 42, 48, 43 (last one is 300 - 257 = 43)
    # Sum 1..9 hex = 271. 300-271=29.
    # So Ring 10 has 29 points.
    
    ring_counts = [1, 6, 12, 18, 24, 30, 36, 42, 48, 29]
    
    # Initial radii: roughly linear
    # R_k = (k / (m+0.5)) * (1 - r_est)
    # r_est ~ 0.05
    # Max R ~ 0.95
    # R_k = 0.95 * k / 10
    
    m = len(ring_counts)
    radii = np.linspace(0.1, 0.9, m)
    # Adjust radii to be more spread out?
    # Actually, we will optimize the positions.
    
    pts = []
    for k, count in enumerate(ring_counts):
        r = radii[k]
        if count == 1:
            pts.append([0,0])
        else:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            # Offset angles for even k to interleave?
            # Standard hex packing offsets alternate rings by pi/count
            offset = np.pi / count if k % 2 == 1 else 0
            angles += offset
            xs = r * np.cos(angles)
            ys = r * np.sin(angles)
            for x, y in zip(xs, ys):
                pts.append([x, y])
                
    pts = np.array(pts)
    
    # Ensure we have exactly N points
    if len(pts) > N:
        pts = pts[:N]
    elif len(pts) < N:
        # Add random points? Should not happen with this logic.
        pass
        
    # 2. Local Search Optimization
    # We want to maximize r = min(min_dist/2, 1 - max_radius)
    # Let's define a function to calculate the score.
    
    def get_score(points):
        # Boundary constraint
        dist_center = np.linalg.norm(points, axis=1)
        min_boundary = 1.0 - np.max(dist_center)
        
        # Pairwise distance
        # N=300, 45k pairs.
        # Use broadcasting for speed? 
        # 300x300 matrix is 90k floats. Fast.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf to ignore self-distance
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        return min(min_boundary, min_pair_dist / 2.0)
    
    current_score = get_score(pts)
    
    # Hill Climbing
    # Step size decay
    step_size = 0.05
    alpha = 0.99
    
    for i in range(5000): # 5000 iterations should be fast enough
        # Calculate gradient-like direction?
        # Simpler: Random perturbation of a random point.
        idx = np.random.randint(N)
        orig_pt = pts[idx].copy()
        
        # Move in random direction
        angle = np.random.uniform(0, 2*np.pi)
        pts[idx][0] += step_size * np.cos(angle)
        pts[idx][1] += step_size * np.sin(angle)
        
        new_score = get_score(pts)
        
        if new_score > current_score:
            current_score = new_score
        else:
            pts[idx] = orig_pt
            
        # Decay step size
        step_size *= alpha
        
        # If step size is too small, break or restart?
        # 5000 * 0.99^5000 is very small.
        # Let's monitor improvement.
        
    return pts

# EVOLVE-BLOCK-START
import numpy as np

N = 300

def construct():
    # 1. Initialize with concentric rings
    # Distribution: 1, 6, 12, 18, 24, 30, 36, 42, 48, 29
    ring_counts = [1, 6, 12, 18, 24, 30, 36, 42, 48, 29]
    
    # Initial radii: geometric or linear?
    # Linear is a safe start.
    m = len(ring_counts)
    # We want the outer ring to be close to 1.
    # But we don't know the optimal radius yet.
    # Let's start with radii: 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.85, 0.9, 0.95, 0.98
    # This is a guess. The optimizer will fix it.
    
    radii = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.85, 0.92, 0.96, 0.99])
    
    pts = []
    for k, count in enumerate(ring_counts):
        r = radii[k]
        if count == 1:
            pts.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            # Offset alternating rings for better packing
            if k % 2 == 1:
                angles += np.pi / count
            xs = r * np.cos(angles)
            ys = r * np.sin(angles)
            for x, y in zip(xs, ys):
                pts.append([x, y])
                
    pts = np.array(pts, dtype=float)
    
    # Ensure exactly N points
    if len(pts) != N:
        # Fallback to hexagonal grid if ring counts didn't sum to N
        # (This shouldn't happen with the hardcoded list, but just in case)
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                pts = points[np.argsort(radius, kind="stable")[:N]]
                break
            spacing *= 0.98

    # 2. Optimization: Local Search
    # We maximize r = min( 1 - max(|c_i|), min_{i<j} |c_i - c_j| / 2 )
    
    def calculate_score(points):
        # Boundary term
        max_r = np.max(np.linalg.norm(points, axis=1))
        boundary_score = 1.0 - max_r
        
        # Pairwise term
        # N=300 is small enough for O(N^2)
        # Calculate all pairwise distances
        # Using broadcasting
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Mask diagonal
        dists[np.diag_indices_from(dists)] = np.inf
        min_dist = np.min(dists)
        pair_score = min_dist / 2.0
        
        return min(boundary_score, pair_score)

    current_score = calculate_score(pts)
    
    # Hill Climbing with adaptive step size
    # Use a simple random walk
    step_size = 0.02
    no_improve_count = 0
    max_no_improve = 100
    
    for i in range(10000):
        if no_improve_count > max_no_improve:
            break
            
        # Pick a random point
        idx = np.random.randint(N)
        orig = pts[idx].copy()
        
        # Random perturbation
        theta = np.random.uniform(0, 2*np.pi)
        pts[idx][0] += step_size * np.cos(theta)
        pts[idx][1] += step_size * np.sin(theta)
        
        new_score = calculate_score(pts)
        
        if new_score > current_score:
            current_score = new_score
            no_improve_count = 0
        else:
            pts[idx] = orig
            no_improve_count += 1
            
        # Decay step size
        step_size *= 0.995
        
        # If step size is too small, it won't help much
        if step_size < 1e-6:
            break
            
    return pts
# EVOLVE-BLOCK-END
