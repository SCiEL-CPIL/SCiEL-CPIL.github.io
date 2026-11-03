"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
import time


def construct():
    """
    Place 50 equal non-overlapping circles in the unit square [-0.5, 0.5]^2
    to maximize the common radius.
    
    Uses a physics-based approach:
    1. Initialize points with a hexagonal lattice pattern
    2. Simulate repulsive forces between circles and attractive forces toward center
    3. Use simulated annealing to escape local minima
    4. Final refinement with gradient-based optimization
    
    The key insight is to model each circle as a particle with:
    - Strong repulsion when too close (like hard-core exclusion)
    - Weak attraction to keep them inside the square
    - Temperature annealing to find global optimum
    """
    
    def initial_hexagonal(n):
        """Generate n points in a hexagonal arrangement within [-0.5, 0.5]^2."""
        points = []
        
        # Hexagonal lattice parameters
        # For ~50 points, try different configurations
        # Target: roughly 7-8 columns, 7-8 rows
        
        best_points = None
        best_min_dist = -1
        
        # Try multiple random rotations and offsets
        for seed in range(100):
            rng = np.random.RandomState(seed)
            angle = rng.uniform(0, np.pi / 6)
            offset_x = rng.uniform(-0.03, 0.03)
            offset_y = rng.uniform(-0.03, 0.03)
            
            # Estimate spacing
            s = 1.0 / np.sqrt(n) * rng.uniform(0.98, 1.05)
            
            # Generate hexagonal lattice
            row_spacing = s * np.sqrt(3) / 2
            col_spacing = s
            
            # Determine number of rows and columns needed
            n_rows = int(np.ceil(1.0 / row_spacing)) + 2
            n_cols = int(np.ceil(1.0 / col_spacing)) + 2
            
            lattice_points = []
            for row in range(n_rows):
                y = -0.5 + row_spacing * (row + 0.5) + offset_y
                if y < -0.5 or y > 0.5:
                    continue
                row_offset = col_spacing / 2 if row % 2 else 0
                for col in range(n_cols):
                    x = -0.5 + col_spacing * (col + 0.5) + row_offset + offset_x
                    if x < -0.5 or x > 0.5:
                        continue
                    lattice_points.append([x, y])
            
            lattice_points = np.array(lattice_points)
            
            if len(lattice_points) < n:
                continue
            
            # Rotate around center
            if abs(angle) > 1e-10:
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                lattice_points = lattice_points @ rot.T
            
            # Keep only points inside square
            mask = (np.abs(lattice_points[:, 0]) <= 0.5) & (np.abs(lattice_points[:, 1]) <= 0.5)
            lattice_points = lattice_points[mask]
            
            if len(lattice_points) < n:
                continue
            
            # Select n points - prefer those with good local spacing
            # Use a greedy approach: pick points that maximize minimum distance to already picked
            selected = [lattice_points[0]]
            for _ in range(n - 1):
                # For each candidate, compute min distance to selected
                dists = cdist(lattice_points, np.array(selected))
                min_dists = dists.min(axis=1)
                
                # Exclude already selected
                for idx in range(len(selected)):
                    min_dists[idx] = 0
                
                # Pick the one with max min-distance
                best_idx = np.argmax(min_dists)
                selected.append(lattice_points[best_idx])
            
            points_arr = np.array(selected)
            
            # Compute minimum distance
            if len(points_arr) == n:
                min_d = pdist(points_arr).min()
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_points = points_arr.copy()
        
        if best_points is None:
            # Fallback: grid
            m = int(np.ceil(np.sqrt(n)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            return grid[:n]
        
        return best_points
    
    # Stage 1: Get initial configuration
    points = initial_hexagonal(N)
    
    # Stage 2: Physics-based optimization with simulated annealing
    # Model: each point experiences forces from:
    # 1. Repulsion from other points (inverse square law, strong when close)
    # 2. Attraction to keep inside square (quadratic penalty near boundaries)
    
    def compute_forces(pts, rep_strength=100.0, attr_strength=50.0, min_dist_threshold=0.08):
        """Compute net force on each point."""
        n = len(pts)
        forces = np.zeros_like(pts)
        
        for i in range(n):
            # Repulsion from other points
            for j in range(n):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                # Strong repulsion when close
                if dist < min_dist_threshold:
                    # Force proportional to 1/dist^2, pointing away
                    force_mag = rep_strength / (dist * dist)
                    forces[i] += force_mag * (diff / dist)
                else:
                    # Weaker repulsion
                    force_mag = rep_strength * 0.01 / (dist * dist)
                    forces[i] += force_mag * (diff / dist)
            
            # Attraction to center / boundary penalty
            # Push away from boundaries
            for k in range(2):
                coord = pts[i, k]
                dist_to_bound = 0.5 - abs(coord)
                if dist_to_bound < 0.1:
                    # Stronger push away from boundary when close
                    force_mag = attr_strength * (1.0 - dist_to_bound / 0.1) ** 2
                    # Direction: away from boundary
                    if coord > 0:
                        forces[i, k] -= force_mag
                    else:
                        forces[i, k] += force_mag
                else:
                    # Weak attraction to center
                    forces[i, k] -= attr_strength * 0.01 * coord
        
        return forces
    
    def score_config(pts):
        """Compute the packing quality (minimum radius)."""
        # Pairwise distances
        d = pdist(pts)
        min_pair_dist = d.min() if len(d) > 0 else 0
        
        # Boundary distances
        b = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
        min_bound_dist = b.min()
        
        return min(min_pair_dist / 2, min_bound_dist)
    
    # Simulated annealing
    np.random.seed(42)
    current_pts = points.copy()
    current_score = score_config(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    # Annealing schedule
    T_start = 0.01
    T_end = 1e-6
    n_iterations = 5000
    cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)
    
    for it in range(n_iterations):
        T = T_start * (cooling_rate ** it)
        
        # Compute forces
        forces = compute_forces(current_pts)
        
        # Move points
        step_size = 0.001 + T * 0.01
        new_pts = current_pts + step_size * forces
        
        # Add random perturbation
        perturbation = T * 0.05
        new_pts += np.random.normal(0, perturbation, size=new_pts.shape)
        
        # Keep inside square
        eps = 1e-9
        new_pts[:, 0] = np.clip(new_pts[:, 0], -0.5 + eps, 0.5 - eps)
        new_pts[:, 1] = np.clip(new_pts[:, 1], -0.5 + eps, 0.5 - eps)
        
        # Evaluate new configuration
        new_score = score_config(new_pts)
        
        # Accept if better or with probability exp((new_score - current_score) / T)
        if new_score >= current_score:
            current_pts = new_pts
            current_score = new_score
        else:
            delta = new_score - current_score
            if np.random.random() < np.exp(delta / (T + 1e-10)):
                current_pts = new_pts
                current_score = new_score
        
        # Track best
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
        
        # Early stopping if no improvement for a while
        if it % 1000 == 0 and it > 0:
            pass  # Could add early stopping logic
    
    points = best_pts
    
    # Stage 3: Final gradient-based refinement
    # Use L-BFGS to maximize the minimum distance
    from scipy.optimize import minimize
    
    def objective(x):
        pts = x.reshape(-1, 2)
        d = pdist(pts)
        b = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
        all_dists = np.concatenate([d, b])
        
        # Smooth minimum
        t = 0.005
        z = all_dists / t
        z_max = z.max()
        log_sum = z_max + np.log(np.sum(np.exp(z - z_max)))
        soft_min = -t * log_sum
        return -soft_min
    
    def gradient(x):
        pts = x.reshape(-1, 2)
        d = pdist(pts)
        b = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
        all_dists = np.concatenate([d, b])
        
        t = 0.005
        z = all_dists / t
        z_max = z.max()
        exp_z = np.exp(z - z_max)
        probs = exp_z / exp_z.sum()
        
        n = len(pts)
        grad = np.zeros_like(pts)
        
        # Pairwise distances
        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                d_ij = np.linalg.norm(diff)
                if d_ij < 1e-10:
                    d_ij = 1e-10
                w = probs[pair_idx] / t
                grad[i] += w * (-diff / d_ij)
                grad[j] += w * (diff / d_ij)
                pair_idx += 1
        
        # Boundary distances
        for i in range(n):
            bx = 0.5 - abs(pts[i, 0])
            by = 0.5 - abs(pts[i, 1])
            t_b = 0.005
            zb = np.array([bx, by]) / t_b
            zb_max = zb.max()
            exp_zb = np.exp(zb - zb_max)
            probs_b = exp_zb / exp_zb.sum()
            
            dx0 = -np.sign(pts[i, 0]) if abs(pts[i, 0]) > 1e-10 else 0
            dx1 = -np.sign(pts[i, 1]) if abs(pts[i, 1]) > 1e-10 else 0
            
            grad[i, 0] += (probs_b[0] / t_b) * dx0
            grad[i, 1] += (probs_b[1] / t_b) * dx1
        
        return -grad.flatten()
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                      options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
    
    points = result.x.reshape(-1, 2)
    
    # Final clip to ensure inside square
    eps = 1e-9
    points[:, 0] = np.clip(points[:, 0], -0.5 + eps, 0.5 - eps)
    points[:, 1] = np.clip(points[:, 1], -0.5 + eps, 0.5 - eps)
    
    return points

# EVOLVE-BLOCK-END
