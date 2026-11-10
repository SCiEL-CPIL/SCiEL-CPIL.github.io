import numpy as np
import time
import random

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Best known radius for N=50 is ~0.125825
    # We aim for 0.1255 to be safe but better than seed (~0.1235)
    r_target = 0.1255
    
    best_centers = None
    best_score = -np.inf
    
    # Helper to compute the actual score (radius) for a given set of centers
    def compute_radius(centers):
        if len(centers) != N:
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        
        # Boundary check
        dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
        r_boundary = np.min(dists_boundary)
        if r_boundary <= 0:
            return 0.0
        
        # Pairwise distances
        # Using broadcasting for (N, 1, 2) - (N, 2, 1) is not quite right for pairwise
        # Better:
        dx = centers[:, 0:1] - centers[:, 0:1].T
        dy = centers[:, 1:2] - centers[:, 1:2].T
        dists = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        
        return min(r_boundary, r_pair)

    # Helper for SA energy function
    def calculate_energy(centers, r):
        # Pairwise overlaps
        dx = centers[:, 0:1] - centers[:, 0:1].T
        dy = centers[:, 1:2] - centers[:, 1:2].T
        dists = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dists, np.inf)
        
        overlaps = np.maximum(0, 2*r - dists)
        E_pair = np.sum(overlaps**2)
        
        # Boundary violations
        dists_boundary = np.hypot(centers[:, 0], centers[:, 1])
        boundary_violations = np.maximum(0, dists_boundary + r - 1.0)
        E_bound = np.sum(boundary_violations**2)
        
        return E_pair + E_bound

    # Generate initial hexagonal lattice points
    def get_initial_points(r_init):
        # Spacing slightly less than 2*r_init to ensure we have enough points
        # We want to start with a valid configuration for a smaller radius
        # and then expand.
        # Let's use a grid that fits ~50 points in the unit circle.
        # Area of unit circle ~ 3.14.
        # Area per point ~ 3.14 / 50 ~ 0.063.
        # Hex cell area = sqrt(3)/2 * s^2 ~ 0.866 * s^2.
        # 0.866 * s^2 ~ 0.063 => s^2 ~ 0.0727 => s ~ 0.27.
        # So r ~ 0.135. This is too high for a rigid grid to fit 50 points?
        # Let's just generate a grid and filter.
        
        s = 0.24  # Corresponds to r=0.12
        k = int(1.0 / s) + 2
        points = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                points.append((x, y))
        points = np.array(points)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Keep points well inside
        points = points[radius <= 1.0 - r_init]
        
        if len(points) < N:
            # If not enough, reduce spacing
            s *= 0.95
            return get_initial_points(r_init)
        
        # Randomly select N points from the valid ones
        # Or select the ones closest to center?
        # Random selection helps break symmetry and might lead to better packing
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]

    # Simulated Annealing
    def anneal(initial_centers, r_target, time_limit):
        current_centers = initial_centers.copy()
        current_energy = calculate_energy(current_centers, r_target)
        
        best_centers = current_centers.copy()
        best_energy = current_energy
        
        T = 0.05  # Initial temperature
        alpha = 0.99  # Cooling rate
        step_size = 0.02
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # Decay temperature
            T *= alpha
            if T < 1e-6:
                break
            
            # Try a move
            idx = np.random.randint(N)
            old_x, old_y = current_centers[idx]
            
            # Random move
            new_x = old_x + np.random.normal(0, step_size)
            new_y = old_y + np.random.normal(0, step_size)
            
            # Keep within bounds roughly to avoid huge energy spikes
            # The energy function will handle it, but clipping helps convergence
            new_x = np.clip(new_x, -1.0, 1.0)
            new_y = np.clip(new_y, -1.0, 1.0)
            
            current_centers[idx] = [new_x, new_y]
            new_energy = calculate_energy(current_centers, r_target)
            
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_centers = current_centers.copy()
            else:
                # Revert
                current_centers[idx] = [old_x, old_y]
                
            # If we found a valid packing, we can stop early
            if best_energy < 1e-8:
                break
                
        return best_centers, best_energy

    # Main loop with restarts
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial points
        # Use a slightly smaller radius for initialization to ensure validity
        r_init = 0.12
        init_centers = get_initial_points(r_init)
        
        # Run SA
        time_left = deadline - time.time() - 5.0  # Leave 5s margin
        if time_left < 1.0:
            break
            
        centers, energy = anneal(init_centers, r_target, time_left)
        
        # Check the actual radius of the best found
        actual_r = compute_radius(centers)
        
        if actual_r > best_score:
            best_score = actual_r
            best_centers = centers
            
        # If we found a valid packing for r_target, we might be able to push higher?
        # For now, if we beat the seed significantly, we are good.
        # The seed is ~0.1235. r_target is 0.1255.
        # If actual_r >= 0.125, we have a good score.
        
        if best_score > 0.125:
            # We have a good result. We could try to optimize further for a higher r,
            # but given time constraints, we stick with this.
            # However, we can try one more time with a higher r if we have time.
            pass

    if best_centers is None:
        # Fallback to seed-like behavior
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

    return best_centers
