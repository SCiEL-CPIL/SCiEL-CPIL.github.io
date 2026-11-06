import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # Estimate spacing for N=300 in unit disk
    # Area of disk = pi. Area per circle ~ pi * r^2.
    # N * pi * r^2 <= pi => r <= 1/sqrt(N) ~ 0.0577
    # Hexagonal packing density is higher.
    # Let's generate a grid and filter.
    
    def generate_lattice():
        spacing = 0.05
        points = []
        # Generate enough points
        max_r = 1.2
        # Hex grid
        # y from -max_r to max_r
        # x from -max_r to max_r
        # For each row j, x offset is (j%2)*spacing/2
        # dy = spacing * sqrt(3)/2
        
        # We need roughly 300 points.
        # Let's just use a dense grid and pick the best 300 closest to center.
        # This ensures we start with a valid-ish configuration.
        
        # Better: Generate lattice, keep those inside 1 - spacing, then if < N, shrink spacing.
        # If > N, keep closest N.
        
        for attempt in range(20):
            s = 0.05 * (0.95 ** attempt)
            dy = s * np.sqrt(3) / 2
            rows = []
            # Determine range
            max_j = int(1.2 / dy) + 2
            max_i = int(1.2 / s) + 2
            
            for j in range(-max_j, max_j + 1):
                for i in range(-max_i, max_i + 1):
                    x = i * s + (j % 2) * s / 2
                    y = j * dy
                    r = np.hypot(x, y)
                    if r < 1.0 - s/2: # Ensure inside
                        rows.append((x, y, r))
            
            if len(rows) >= N:
                # Sort by distance to center and take first N
                rows.sort(key=lambda t: t[2])
                pts = np.array([t[:2] for t in rows[:N]])
                return pts
            
        # Fallback: Random
        return np.random.randn(N, 2) * 0.8

    pts = generate_lattice()
    
    # Add small noise to break symmetry
    pts += np.random.randn(N, 2) * 0.001
    
    def compute_score(p):
        # Returns min distance / 2 and min boundary dist
        # p is (N, 2)
        # Boundary
        dists_center = np.linalg.norm(p, axis=1)
        min_boundary = 1.0 - np.max(dists_center)
        
        # Pairwise distances
        # Efficient min distance
        # For N=300, 45k pairs. Can do O(N^2) easily.
        # diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        # np.fill_diagonal(dists, np.inf)
        # min_pair = np.min(dists) / 2
        
        # Faster:
        # Use scipy if available? No, stick to numpy.
        # 300x300 matrix is small.
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair = np.sqrt(np.min(dists_sq)) / 2
        
        return min(min_pair, min_boundary)

    current_score = compute_score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    # Local Search
    # Strategy: 
    # 1. Compute forces.
    # 2. Move.
    # 3. If score improves, accept. Else, maybe revert or reduce step.
    
    step_size = 0.005
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Reduce step size and reset counter
            step_size *= 0.8
            no_improve_count = 0
            if step_size < 1e-5:
                break
            
        # Compute forces
        # Force from neighbors: repel if close
        # Force from boundary: push in if close to edge
        
        # We want to maximize the min distance.
        # Let's identify the "bottleneck" distance r_b = current_score
        # We want to increase r_b.
        # Points that are at distance < r_b + delta from each other or boundary are "active".
        
        r_b = current_score
        delta = 0.01 # Look-ahead window
        
        forces = np.zeros_like(pts)
        
        # Boundary force
        dists_center = np.linalg.norm(pts, axis=1)
        # If dist > 1 - r_b, push in
        # Target: 1 - r_b
        # Force direction: -c / |c|
        # Magnitude: proportional to violation
        violation_b = np.maximum(0, dists_center - (1.0 - r_b))
        # Only apply if violation is small? No, always push in if outside ideal radius
        # Actually, if we are strictly inside, we might want to push out to maximize radius?
        # No, the constraint is |c| <= 1 - r.
        # To maximize r, we want |c| to be as large as possible? No, r is the radius of the circles.
        # The constraint is |c_i| + r <= 1.
        # So we want to place points such that the minimum (1 - |c_i|) is maximized.
        # This means we want |c_i| to be as small as possible?
        # Wait.
        # Score = min( min_pair_dist/2, min_boundary_dist ).
        # min_boundary_dist = min_i (1 - |c_i|).
        # To maximize this, we want |c_i| to be small.
        # So points should be pulled towards the center?
        # But they also need to be far from each other.
        # So there is a balance.
        
        # Let's define the force more carefully.
        # We want to increase r.
        # Let r_new = r + dr.
        # We need all pairs to be > 2*r_new and all |c_i| < 1 - r_new.
        
        # Simple heuristic:
        # For each pair (i,j) with d_ij < 2*r_b + delta:
        # Push i away from j, j away from i.
        # For each i with 1-|c_i| < r_b + delta:
        # Push i towards center.
        
        # This is O(N^2).
        
        # Compute pairwise distances again
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, np.inf)
        
        # Threshold
        thresh = 2.0 * r_b + delta
        
        # For each i
        for i in range(N):
            # Find neighbors within thresh
            # mask = dists[i, :] < thresh
            # neighbors = np.where(mask)[0]
            # 
            # Vector sum
            # f = sum (pts[i] - pts[j]) / dists[i,j] * weight
            
            # Boundary
            d_c = np.linalg.norm(pts[i])
            b_thresh = 1.0 - r_b + delta
            if d_c > b_thresh:
                # Push towards center
                # Strength proportional to violation
                viol = d_c - b_thresh
                forces[i] -= (pts[i] / d_c) * viol * 10.0 # Scaling factor
            
            # Neighbors
            # To speed up, we can just sum over all j where dist < thresh
            # But loop is slow. Vectorize per i?
            
            # Let's try to vectorize the force calculation.
            
        # Vectorized Force Calculation
        # We want F_i = sum_j G(d_ij) * (p_i - p_j)
        # where G(d) is 0 if d > thresh, else some repulsion.
        
        # Let's use a simple repulsion: 1/d if d < thresh, else 0.
        # This is strong for very close points.
        
        # Mask for active pairs
        active_mask = dists < thresh
        # Set diagonal to 0 to avoid self-repulsion
        np.fill_diagonal(active_mask, False)
        
        # Weights
        # w_ij = 1 / dists[i,j] for active, else 0
        # Avoid div by 0 (diagonal is masked)
        weights = np.zeros_like(dists)
        # Only compute for active to save time? 
        # dists is 300x300, small.
        weights[active_mask] = 1.0 / dists[active_mask]
        
        # Force from neighbors:
        # F_i = sum_j weights[i,j] * (p_i - p_j)
        # = sum_j weights[i,j] * p_i - sum_j weights[i,j] * p_j
        # = p_i * sum_j weights[i,j] - sum_j weights[i,j] * p_j
        
        w_sum = np.sum(weights, axis=1) # (N,)
        # weighted_points = weights @ pts # (N, 2)
        # F_neighbor = pts * w_sum[:, np.newaxis] - weighted_points
        
        # Boundary Force
        # F_bound_i = -k * p_i / |p_i| * violation_i
        dists_center = np.linalg.norm(pts, axis=1)
        b_viol = np.maximum(0, dists_center - (1.0 - r_b + delta))
        # Direction is -p_i / |p_i|
        # If dist is 0, direction is 0.
        with np.errstate(divide='ignore', invalid='ignore'):
            dir_bound = pts / dists_center[:, np.newaxis]
        dir_bound[~np.isfinite(dir_bound)] = 0.0
        
        F_bound = -dir_bound * b_viol[:, np.newaxis] * 5.0 # Tuning constant
        
        F_neighbor = pts * w_sum[:, np.newaxis] - (weights @ pts)
        
        total_force = F_neighbor + F_bound
        
        # Normalize forces to prevent huge jumps
        force_norms = np.linalg.norm(total_force, axis=1)
        max_norm = np.max(force_norms)
        if max_norm > 0:
            total_force = total_force / max_norm * step_size
        else:
            break # No force, stuck?
            
        # Update
        new_pts = pts + total_force
        
        # Project back to disk if any point went outside (safety)
        new_dists = np.linalg.norm(new_pts, axis=1)
        mask_out = new_dists > 1.0
        if np.any(mask_out):
            new_pts[mask_out] = new_pts[mask_out] / new_dists[mask_out, np.newaxis] * 1.0
            
        new_score = compute_score(new_pts)
        
        if new_score > current_score:
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If we are stuck, maybe try a random kick?
            if no_improve_count % 10 == 0:
                # Random perturbation
                kick_idx = np.random.choice(N, 5, replace=False)
                pts[kick_idx] += np.random.randn(5, 2) * 0.01
                # Re-clamp to disk
                d = np.linalg.norm(pts[kick_idx], axis=1)
                pts[kick_idx] = pts[kick_idx] / d[:, np.newaxis] * np.minimum(d, 1.0)[:, np.newaxis]
                current_score = compute_score(pts)
                # If this kick made it worse, we'll just let the loop handle it
                # But we don't revert immediately, we try to fix it with forces.
                
    return best_pts
