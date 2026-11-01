import numpy as np
import time
from scipy.spatial import KDTree

N = 200

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 145.0  # Leave 5s margin for safety
    
    def hex_init():
        # Generate a hexagonal grid
        # Approximate spacing for N points in unit square
        # Area per point ~ 1/N. Hex cell area ~ s^2 * sqrt(3)/2
        # s ~ sqrt(2 / (N * sqrt(3)))
        s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
        
        # Generate points in a large range
        n_x = int(1.0 / s) + 2
        n_y = int(1.0 / s) + 2
        
        pts = []
        for i in range(n_y):
            for j in range(n_x):
                x = j * s + (i % 2) * s * 0.5
                y = i * s * np.sqrt(3.0) / 2.0
                pts.append((x, y))
        
        pts = np.array(pts)
        
        # Shift to center roughly at origin
        pts[:, 0] -= 0.5
        pts[:, 1] -= 0.5
        
        # Filter points inside [-0.5, 0.5]
        mask = (pts[:, 0] >= -0.5) & (pts[:, 0] <= 0.5) & (pts[:, 1] >= -0.5) & (pts[:, 1] <= 0.5)
        pts = pts[mask]
        
        # If we have too many, take the first N (or random)
        if len(pts) > N:
            # Random selection to avoid bias, or just slice
            # Random is better for diversity
            idx = np.random.choice(len(pts), N, replace=False)
            pts = pts[idx]
        elif len(pts) < N:
            # Add random points to fill
            extra = np.random.uniform(-0.5, 0.5, size=(N - len(pts), 2))
            pts = np.vstack([pts, extra])
            
        return pts

    def compute_score(pts):
        """Compute the minimum radius for a given set of points."""
        if len(pts) < 2:
            return 0.5
        # Distances between all pairs
        # Using KDTree for efficiency
        tree = KDTree(pts)
        # Query for 2 nearest neighbors (self and 1 other)
        dists, _ = tree.query(pts, k=2)
        min_dist = np.min(dists[:, 1])
        
        # Distances to boundaries
        # x in [-0.5, 0.5], so dist to wall is min(0.5 - x, x + 0.5)
        dist_x = np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5)
        dist_y = np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
        min_wall_dist = np.min(np.minimum(dist_x, dist_y))
        
        return min(min_dist / 2.0, min_wall_dist)

    def relax(pts, steps=5000, initial_temp=0.01, cooling=0.999):
        """Force-directed relaxation."""
        pts = pts.copy()
        temp = initial_temp
        
        for step in range(steps):
            if time.time() - start_time > timeout:
                break
                
            tree = KDTree(pts)
            # Get 2 nearest neighbors
            dists, indices = tree.query(pts, k=2)
            
            # Calculate forces
            forces = np.zeros_like(pts)
            
            for i in range(N):
                # Neighbor forces
                # We only consider the closest neighbor to create a "rigid" packing feel
                # Or sum over all neighbors? Summing over all is O(N^2) if not careful, but KDTree gives all.
                # Let's use a repulsive force from ALL points within a certain radius to ensure global separation.
                # But for packing, the closest neighbor is the main constraint.
                
                # Simple approach: Repulsion from closest neighbor
                j = indices[i, 1]
                diff = pts[i] - pts[j]
                dist = dists[i, 1]
                if dist > 1e-9:
                    # Inverse square repulsion
                    f_mag = 1.0 / (dist * dist)
                    forces[i] += f_mag * diff / dist
                
                # Wall forces
                # Push away from walls if close
                # Use a soft potential: if d < d_thresh, force = k * (d_thresh - d)
                # Or just hard clamp and let neighbor forces do the work?
                # Better: Add repulsion from walls proportional to how close it is.
                
                # Distance to left wall: x + 0.5
                # Distance to right wall: 0.5 - x
                # Distance to bottom wall: y + 0.5
                # Distance to top wall: 0.5 - y
                
                # We want to maximize min distance.
                # If a point is close to a wall, it constrains the radius.
                # So we push it away from the wall.
                
                dx_left = pts[i, 0] + 0.5
                dx_right = 0.5 - pts[i, 0]
                dy_bottom = pts[i, 1] + 0.5
                dy_top = 0.5 - pts[i, 1]
                
                min_dx = min(dx_left, dx_right)
                min_dy = min(dy_bottom, dy_top)
                
                # If close to wall, push away
                # Threshold could be related to current estimated radius
                # Let's use a fixed threshold for stability, or adaptive.
                # For now, simple linear repulsion if < 0.05
                thresh = 0.05
                k_wall = 10.0
                
                if dx_left < thresh:
                    forces[i, 0] += k_wall * (thresh - dx_left)
                if dx_right < thresh:
                    forces[i, 0] -= k_wall * (thresh - dx_right)
                if dy_bottom < thresh:
                    forces[i, 1] += k_wall * (thresh - dy_bottom)
                if dy_top < thresh:
                    forces[i, 1] -= k_wall * (thresh - dy_top)
            
            # Update positions
            # Normalize forces to prevent huge jumps
            force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
            force_norms[force_norms == 0] = 1.0
            normalized_forces = forces / force_norms
            
            pts += temp * normalized_forces
            
            # Clamp to boundaries
            pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
            pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
            
            # Cool down
            temp *= cooling
            
        return pts

    # Main loop with multiple restarts
    best_pts = None
    best_score = 0.0
    
    # Try a few different initializations
    for restart in range(5):
        if time.time() - start_time > timeout:
            break
            
        if restart == 0:
            pts = hex_init()
        else:
            # Perturb the best solution found so far
            if best_pts is not None:
                pts = best_pts + np.random.normal(0, 0.01, size=(N, 2))
                pts = np.clip(pts, -0.5, 0.5)
            else:
                pts = hex_init()
        
        # Relax
        # Use more steps for the first restart, fewer for subsequent if time is tight
        steps = 2000 if restart == 0 else 1000
        pts = relax(pts, steps=steps, initial_temp=0.02, cooling=0.9995)
        
        # Final polish: high precision local search
        # Use a finer step size
        pts = relax(pts, steps=500, initial_temp=0.005, cooling=0.9999)
        
        score = compute_score(pts)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            
            # If we are close to the known best, try to polish further
            if best_score > 0.036:
                 # Do another pass with very small temp
                 pts = relax(pts, steps=1000, initial_temp=0.001, cooling=0.99999)
                 score = compute_score(pts)
                 if score > best_score:
                     best_score = score
                     best_pts = pts.copy()

    return best_pts
# EVOLVE-BLOCK-END
