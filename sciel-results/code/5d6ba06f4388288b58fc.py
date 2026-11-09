import numpy as np
import time

N = 50

def construct():
        # 1. Initialization
        # Use a 7x7 grid for 49 points, and add a 50th point.
        # Where to put the 50th? 
        # If we put it in the middle, it clashes with the center point.
        # Maybe remove the center point of the 7x7 grid and put 2 points there?
        # Or just start with a random jittered 7x7 grid?
        
        # Let's try a hexagonal initialization.
        # Generate a large hex grid and select 50 points.
        
        # Simpler: Random initialization inside the square.
        # Then optimize.
        
        def init_grid():
            # 7x7 grid
            m = 7
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            # grid has 49 points.
            # Add a 50th point. 
            # Let's add it at (0, 0) and remove the center point of the grid?
            # The center of the 7x7 grid is at (0,0).
            # So replace the center point with two points? No, we need 50 distinct points.
            # Let's just add a point at a random position near the center and let the optimizer fix it.
            # Or, better: initialize with 50 random points.
            
            # Actually, a 7x7 grid is so good that we should preserve it.
            # Let's take the 7x7 grid.
            # We need 1 more point.
            # The "holes" in a 7x7 grid are the centers of the squares formed by 4 adjacent circles.
            # There are 6x6 = 36 such holes.
            # The radius of a circle that fits in a hole of a 7x7 grid (spacing 1/7) is:
            # dist to center = (1/7) * sqrt(2)/2 = 1/(7*sqrt(2)) approx 0.101.
            # But the existing circles have radius 1/14 approx 0.071.
            # The distance from hole center to existing circle centers is 1/(7*sqrt(2)).
            # If we place a new circle there, its distance to existing circles is 1/(7*sqrt(2)).
            # This is LARGER than 1/7. So it fits?
            # Wait. The existing circles are at the corners of the square cell.
            # Distance from corner to center is $d = \sqrt{(1/14)^2 + (1/14)^2} = 1/14 * \sqrt{2}$.
            # The new circle at the center has distance $1/14 * \sqrt{2}$ to the 4 surrounding circles.
            # This is $> 1/7$? No. $1/14 * 1.414 = 0.101$. $1/7 = 0.142$.
            # So the distance is SMALLER than the grid spacing.
            # So the new circle would overlap with the surrounding ones if the radius was 1/14.
            # To fit, the radius must be at most half the min distance.
            # Min distance would be $1/14 * \sqrt{2} \approx 0.101$.
            # Radius $\approx 0.0505$.
            # This is much worse than 0.071.
            
            # So simply adding a point to a 7x7 grid is bad.
            # We need to DISTORT the grid.
            
            # Let's use a random initialization.
            np.random.seed(42)
            pts = np.random.rand(N, 2) - 0.5
            return pts

        pts = init_grid()
        
        # 2. Optimization
        deadline = time.time() + 110
        step_size = 0.01
        min_dist = 0.0
        
        while time.time() < deadline:
            # Compute pairwise distances
            # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            # dist = np.sqrt(np.sum(diff**2, axis=-1))
            # np.fill_diagonal(dist, np.inf)
            
            # Efficient distance calculation
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=-1)
            np.fill_diagonal(dist_sq, np.inf)
            dist = np.sqrt(dist_sq)
            
            # Boundary distances
            # dist_x = 0.5 - np.abs(pts[:, 0])
            # dist_y = 0.5 - np.abs(pts[:, 1])
            # boundary_dist = np.minimum(dist_x, dist_y)
            
            # Current min radius
            r_pair = np.min(dist) / 2
            r_bound = np.min(0.5 - np.abs(pts[:, 0]))
            r_bound_y = np.min(0.5 - np.abs(pts[:, 1]))
            r_bound = min(r_bound, r_bound_y)
            r = min(r_pair, r_bound)
            
            if r > min_dist:
                min_dist = r
                best_pts = pts.copy()
            
            # Find active constraints
            # Pairs with dist < r_pair + tol
            tol = 1e-6
            active_pairs = dist < (2 * r + tol) # If r is limited by pairs
            # Actually, we want to maximize the min.
            # Let's just push all pairs apart.
            
            # Gradient of sum of 1/dist?
            # Or just move points apart if they are close.
            
            # Simple Repulsion
            forces = np.zeros_like(pts)
            
            # Pair repulsion
            # Only consider pairs that are close to the minimum to save time?
            # Or all pairs? N=50 is small. 1225 pairs.
            
            # Force magnitude proportional to 1/d^2
            # Avoid division by zero
            safe_dist = np.where(dist < 1e-6, 1e-6, dist)
            force_mag = 1.0 / (safe_dist**2)
            
            # Direction
            # force_vec = diff / dist^3 ? (Coulomb)
            # Let's use force_vec = diff / dist^2 (direction * 1/d)
            # We want to push i away from j.
            # F_i += (p_i - p_j) / |p_i - p_j|^2
            
            # diff is p_i - p_j
            # force_vec = diff / dist^3 (so that magnitude is 1/d^2)
            # dist is matrix. diff is tensor.
            # force_vec = diff / (dist**3)[:,:,np.newaxis]
            # Sum over j for each i.
            
            # Handle dist=0
            mask = dist < 1e-6
            # If dist is 0, force is 0 (or random).
            
            force_vec = np.zeros_like(diff)
            # Avoid div by zero
            dist_safe = np.where(dist < 1e-6, 1e-6, dist)
            force_vec = diff / (dist_safe**3)[:,:,np.newaxis]
            # Zero out diagonal
            np.fill_diagonal(force_vec[:, :, 0], 0)
            np.fill_diagonal(force_vec[:, :, 1], 0)
            
            forces += np.sum(force_vec, axis=1)
            
            # Boundary repulsion
            # If p_x > 0.5 - r, push left.
            # If p_x < -0.5 + r, push right.
            # We don't know r yet, so use a fixed margin or current r.
            # Let's use current r.
            
            margin = r
            # X boundary
            # Right wall: x > 0.5 - margin
            # Force left: -1
            # Left wall: x < -0.5 + margin
            # Force right: +1
            
            # Soft repulsion: 
            # f_x = -sign(x) * max(0, |x| - (0.5 - margin))**2 ?
            
            # Let's just hard clip and add a small random jitter if stuck?
            # No, let's use gradient.
            
            # Boundary force
            # dx = pts[:, 0]
            # dy = pts[:, 1]
            
            # Force from right wall
            f_x = np.zeros(N)
            f_y = np.zeros(N)
            
            # Right wall: x > 0.5 - r
            # dist_to_wall = 0.5 - x
            # if dist < r, force = (r - dist) * k
            # Let's use k=100
            
            k_wall = 100.0
            dist_right = 0.5 - pts[:, 0]
            dist_left = 0.5 + pts[:, 0]
            dist_top = 0.5 - pts[:, 1]
            dist_bottom = 0.5 + pts[:, 1]
            
            # Push away from walls if too close
            # If dist < r, push in.
            # Force magnitude = (r - dist) * k_wall
            
            f_x -= np.where(dist_right < r, (r - dist_right) * k_wall, 0)
            f_x += np.where(dist_left < r, (r - dist_left) * k_wall, 0)
            f_y -= np.where(dist_top < r, (r - dist_top) * k_wall, 0)
            f_y += np.where(dist_bottom < r, (r - dist_bottom) * k_wall, 0)
            
            forces[:, 0] += f_x
            forces[:, 1] += f_y
            
            # Update
            # Normalize forces to prevent huge steps
            norm_forces = np.linalg.norm(forces, axis=1, keepdims=True)
            norm_forces = np.where(norm_forces < 1e-6, 1e-6, norm_forces)
            forces = forces / norm_forces * step_size
            
            pts += forces
            
            # Clip to square
            pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
            pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
            
            # Reduce step size over time?
            # step_size *= 0.999
            
        return best_pts
