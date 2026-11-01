"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    def compute_score(points):
        """Compute the packing radius for a set of points."""
        # Distance from origin
        r_orig = np.hypot(points[:, 0], points[:, 1])
        # Distance to boundary
        boundary_margin = 1.0 - r_orig
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        return min(boundary_margin.min(), min_pair_dist / 2.0)

    def init_hexagonal(n, seed=42):
        """Initialize n points on a perturbed hexagonal lattice."""
        rng = np.random.RandomState(seed)
        # Estimate spacing for n points in unit circle
        # Area per point ~ pi/n, spacing ~ sqrt(pi/n)
        spacing = np.sqrt(np.pi / n) * 0.95
        
        # Generate hexagonal lattice points
        points = []
        # Range of indices
        max_k = int(2.0 / spacing) + 3
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= (1.0 - spacing * 0.6) ** 2:
                    points.append([x, y])
        
        points = np.array(points)
        
        # If we have more than n points, randomly select n
        if len(points) > n:
            indices = rng.choice(len(points), size=n, replace=False)
            points = points[indices]
        
        # If we have fewer, pad with random points
        if len(points) < n:
            extra = rng.uniform(-0.9, 0.9, size=(n - len(points), 2))
            # Ensure inside circle
            r = np.hypot(extra[:, 0], extra[:, 1])
            mask = r > 0.85
            while mask.any():
                extra[mask] = rng.uniform(-0.85, 0.85, size=(mask.sum(), 2))
                r = np.hypot(extra[:, 0], extra[:, 1])
                mask = r > 0.85
            points = np.vstack([points, extra])
        
        # Add small random perturbation to break symmetry
        points = points + rng.normal(0, spacing * 0.05, size=points.shape)
        
        # Ensure all points are inside the circle
        r = np.hypot(points[:, 0], points[:, 1])
        for i in range(len(points)):
            if r[i] > 0.95:
                points[i] *= 0.95 / r[i]
        
        return points

    def physics_refine(points, max_iters=5000, dt=0.001, cooling=0.9995):
        """Physics-based refinement using repulsive forces."""
        points = points.copy()
        n = len(points)
        
        # Force parameters
        repulsion_strength = 1.0
        wall_strength = 5.0
        damping = 0.95
        
        velocities = np.zeros_like(points)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            forces = np.zeros_like(points)
            
            # Pairwise repulsion
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, 1.0)
            
            # Only consider close pairs for efficiency
            min_dist = 0.02
            mask = dist < min_dist * 5
            np.fill_diagonal(mask, False)
            
            for i in range(n):
                for j in range(n):
                    if mask[i, j]:
                        d = dist[i, j]
                        if d > 1e-10:
                            # Repulsive force
                            force = repulsion_strength / (d * d)
                            forces[i] += (diff[i, j] / d) * force
            
            # Wall repulsion
            r = np.hypot(points[:, 0], points[:, 1])
            wall_margin = 1.0 - r
            for i in range(n):
                if wall_margin[i] < 0.1:
                    force = wall_strength * (0.1 - wall_margin[i])
                    if r[i] > 1e-10:
                        forces[i] -= (points[i] / r[i]) * force
            
            # Update velocities and positions
            velocities = velocities * damping + forces * dt
            points += velocities * dt
            
            # Ensure points stay inside the circle
            r = np.hypot(points[:, 0], points[:, 1])
            for i in range(n):
                if r[i] > 0.99:
                    points[i] *= 0.99 / r[i]
            
            # Cool down
            dt *= cooling
        
        return points

    def local_optimize(points, max_iters=2000):
        """Local optimization using gradient-based method."""
        points = points.copy()
        n = len(points)
        
        step_size = 0.005
        best_score = compute_score(points)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Compute gradient of the score
            # We want to maximize min(min_pair_dist/2, boundary_margin)
            # Use subgradient approach
            
            # Find critical pairs
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            
            r_orig = np.hypot(points[:, 0], points[:, 1])
            boundary_margin = 1.0 - r_orig
            
            current_score = min(boundary_margin.min(), np.min(dist) / 2.0)
            
            # Determine which constraint is active
            if boundary_margin.min() <= np.min(dist) / 2.0:
                # Boundary constraint is active
                active_idx = np.argmin(boundary_margin)
                grad = np.zeros_like(points)
                if r_orig[active_idx] > 1e-10:
                    grad[active_idx] = -points[active_idx] / r_orig[active_idx]
            else:
                # Pair constraint is active
                min_dist_val = np.min(dist)
                active_pairs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if abs(dist[i, j] - min_dist_val) < 1e-10:
                            active_pairs.append((i, j))
                
                grad = np.zeros_like(points)
                for i, j in active_pairs:
                    d = dist[i, j]
                    if d > 1e-10:
                        direction = (points[i] - points[j]) / d
                        grad[i] += direction
                        grad[j] -= direction
                
                # Normalize
                norm = np.linalg.norm(grad)
                if norm > 1e-10:
                    grad /= norm
            
            # Update
            new_points = points + step_size * grad
            
            # Ensure inside circle
            r = np.hypot(new_points[:, 0], new_points[:, 1])
            for i in range(n):
                if r[i] > 0.995:
                    new_points[i] *= 0.995 / r[i]
            
            new_score = compute_score(new_points)
            
            if new_score > best_score:
                points = new_points
                best_score = new_score
            else:
                step_size *= 0.95
        
        return points

    # Main algorithm
    best_points = None
    best_score = 0.0
    
    # Try multiple initializations
    for seed in range(5):
        if time.time() > deadline - 10.0:
            break
        
        points = init_hexagonal(N, seed=seed)
        score = compute_score(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Physics refinement
        if time.time() < deadline - 15.0:
            refined = physics_refine(points, max_iters=3000)
            refined_score = compute_score(refined)
            
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined.copy()
        
        # Local optimization
        if time.time() < deadline - 10.0:
            optimized = local_optimize(best_points, max_iters=1000)
            opt_score = compute_score(optimized)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = optimized.copy()
    
    # Final verification
    if best_points is not None:
        r = np.hypot(best_points[:, 0], best_points[:, 1])
        for i in range(N):
            if r[i] > 1.0:
                best_points[i] *= 0.999 / r[i]
    
    if best_points is None:
        # Fallback: simple hexagonal lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        best_points = points[np.argsort(radius, kind="stable")[:N]]
    
    return best_points
# EVOLVE-BLOCK-END
