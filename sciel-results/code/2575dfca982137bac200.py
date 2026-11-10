import numpy as np
import time

N = 300


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    def score(points):
        """Compute the packing score for given points."""
        if len(points) != N:
            return -1.0
        if np.any(~np.isfinite(points)):
            return -1.0
        # Check container constraint
        dists_to_center = np.linalg.norm(points, axis=1)
        if np.any(dists_to_center >= 1.0):
            return -1.0
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        min_container = np.min(1.0 - dists_to_center)
        return min(min_pair, min_container)

    def hex_lattice_init():
        """Generate initial hexagonal lattice points."""
        # Estimate spacing for N points in unit circle
        # Area of unit circle = pi, each circle occupies ~pi*r^2
        # For hexagonal packing, density ~ 0.9069
        # N * pi * r^2 <= 0.9069 * pi => r <= sqrt(0.9069/N)
        r_est = np.sqrt(0.9069 / N)
        spacing = 2 * r_est * 1.05  # slight expansion
        
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0]
        # Sort by distance from center to prefer central points
        idx = np.argsort(radius)
        points = points[idx[:N]]
        return points

    def optimize(points, max_iters=2000, step_scale=1.0):
        """Local optimization using repulsive forces."""
        points = points.copy().astype(float)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Compute forces
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            
            # Repulsive force: stronger for closer pairs
            # Force magnitude ~ 1/d^3 to avoid overlap
            force_mag = np.where(d < 0.5, 1.0 / (d**3 + 1e-10), 0.0)
            force_mag = np.maximum(force_mag, 0.01)
            forces = (diff * force_mag[:, :, np.newaxis]) / d[:, :, np.newaxis]
            net_force = np.sum(forces, axis=1)
            
            # Attractive force toward center to keep points in container
            center_force = -points * 0.5
            
            # Combined force
            total_force = net_force + center_force
            
            # Adaptive step size
            max_force = np.max(np.linalg.norm(total_force, axis=1)) + 1e-10
            step = step_scale / max_force * 0.1
            step = min(step, 0.05)  # cap step size
            
            # Update positions
            points += total_force * step
            
            # Project back into unit circle
            dists = np.linalg.norm(points, axis=1)
            mask = dists > 0.999
            if np.any(mask):
                points[mask] *= 0.999 / dists[mask]
            
            # Check for improvement
            if iteration % 50 == 0:
                s = score(points)
                if s > 0.053:
                    break
        
        return points

    def simulated_annealing(points, n_iter=5000, t_start=0.01, t_end=0.0001):
        """Simulated annealing for global optimization."""
        points = points.copy().astype(float)
        best_points = points.copy()
        best_score = score(points)
        
        for i in range(n_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Temperature schedule
            t = t_start * (t_end / t_start) ** (i / max(n_iter, 1))
            
            # Random perturbation
            noise = np.random.normal(0, t, size=points.shape)
            new_points = points + noise
            
            # Project into unit circle
            dists = np.linalg.norm(new_points, axis=1)
            mask = dists > 0.999
            if np.any(mask):
                new_points[mask] *= 0.999 / dists[mask]
            
            # Compute scores
            s_new = score(new_points)
            s_old = best_score if i == 0 else score(points)
            
            # Accept if better or with probability exp((s_new - s_old)/t)
            if s_new >= s_old or np.random.random() < np.exp((s_new - s_old) / max(t, 1e-10)):
                points = new_points
                if s_new > best_score:
                    best_score = s_new
                    best_points = points.copy()
        
        return best_points

    # Initialize with hexagonal lattice
    points = hex_lattice_init()
    best_points = points.copy()
    best_score = score(points)
    
    # Local optimization
    points = optimize(points, max_iters=1000)
    s = score(points)
    if s > best_score:
        best_score = s
        best_points = points.copy()
    
    # Simulated annealing with multiple restarts
    n_restarts = 3
    for r in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb initial solution
        if r == 0:
            sa_points = best_points.copy()
        else:
            sa_points = hex_lattice_init()
            # Add random perturbation
            sa_points += np.random.normal(0, 0.01, size=sa_points.shape)
            dists = np.linalg.norm(sa_points, axis=1)
            mask = dists > 0.999
            if np.any(mask):
                sa_points[mask] *= 0.999 / dists[mask]
        
        sa_points = simulated_annealing(sa_points, n_iter=3000)
        s = score(sa_points)
        if s > best_score:
            best_score = s
            best_points = sa_points.copy()
    
    # Final local optimization
    if time.time() < deadline - 10.0:
        final_points = optimize(best_points, max_iters=500)
        s = score(final_points)
        if s > best_score:
            best_points = final_points
            best_score = s
    
    return best_points
