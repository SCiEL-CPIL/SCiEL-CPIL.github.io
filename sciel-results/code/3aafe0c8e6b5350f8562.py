import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Best known radius for N=50 is ~0.125825
    # We aim to find a valid packing close to this.
    # The seed (hexagonal) is around 0.1235.
    # We use SA to improve from a good initial guess.
    
    best_centers = None
    best_score = -np.inf
    
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
        dx = centers[:, 0:1] - centers[:, 0:1].T
        dy = centers[:, 1:2] - centers[:, 1:2].T
        dists = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        
        return min(r_boundary, r_pair)

    def calculate_energy(centers, r):
        # Energy is sum of squared overlaps + squared boundary violations
        # This is a smooth surrogate for the min-distance objective.
        dx = centers[:, 0:1] - centers[:, 0:1].T
        dy = centers[:, 1:2] - centers[:, 1:2].T
        dists = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dists, np.inf)
        
        # Overlaps: max(0, 2*r - d)^2
        overlaps = np.maximum(0, 2*r - dists)
        E_pair = np.sum(overlaps**2)
        
        # Boundary: max(0, |c| + r - 1)^2
        dists_boundary = np.hypot(centers[:, 0], centers[:, 1])
        boundary_violations = np.maximum(0, dists_boundary + r - 1.0)
        E_bound = np.sum(boundary_violations**2)
        
        return E_pair + E_bound

    def get_initial_points(r_init):
        # Generate a hexagonal lattice and select N points closest to center
        # Spacing s implies min dist s. So r_init = s/2.
        # We want to start with a valid config for r_init.
        s = 2 * r_init
        k = int(1.0 / s) + 2
        points = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                points.append((x, y))
        points = np.array(points)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Keep points well inside: |c| + r <= 1 => |c| <= 1 - r
        points = points[radius <= 1.0 - r_init]
        
        if len(points) < N:
            # If not enough, reduce spacing (increase density)
            # This implies we might need to be more careful or just use a smaller r_init
            return get_initial_points(r_init * 0.9)
        
        # Select N points closest to origin
        idx = np.argsort(radius, kind="stable")
        return points[idx[:N]]

    def anneal(initial_centers, r_target, time_limit):
        current_centers = initial_centers.copy()
        current_energy = calculate_energy(current_centers, r_target)
        
        best_centers = current_centers.copy()
        best_energy = current_energy
        
        T = 0.02  # Initial temperature
        alpha = 0.995  # Cooling rate
        step_size = 0.05
        
        end_time = time.time() + time_limit
        
        # Adaptive step size? Keep it simple for now.
        
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
            # The energy function handles it, but clipping helps convergence
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
                
            # If we found a valid packing (energy ~ 0), we can stop early
            # or continue to polish. Since we are maximizing r, 
            # finding a valid packing for r_target is a success.
            if best_energy < 1e-8:
                # We might try to increase r_target slightly? 
                # No, we fix r_target for this anneal run.
                # We break to save time for other restarts or higher r targets.
                break
                
        return best_centers, best_energy

    # Main loop with restarts and increasing target radius
    
    # Start with a conservative target and try to push it up
    # Best known is 0.125825. Let's try 0.1250, 0.1255, 0.1258
    # But we must ensure we return a valid result even if we fail to reach high targets.
    
    # Strategy:
    # 1. Get a valid initial packing with r=0.12 (easy).
    # 2. Try to anneal to r=0.125. If successful, record it.
    # 3. Try to anneal to r=0.1255. If successful, record it.
    # 4. Try to anneal to r=0.1258. If successful, record it.
    # 5. If time permits, do local polish on the best found.
    
    # Actually, a better strategy is to run SA with a fixed high target (e.g. 0.1258)
    # starting from a good initial guess. If it fails (energy > 0), the resulting
    # centers might still be a valid packing for a lower radius.
    # We should evaluate the actual radius of the result.
    
    r_targets = [0.1258, 0.1255, 0.1250, 0.1245]
    
    for r_target in r_targets:
        if time.time() > deadline - 10.0:
            break
            
        # Generate initial points for a slightly smaller radius to ensure validity
        # If we start with r_init = r_target * 0.95, we are likely valid.
        r_init = r_target * 0.95
        init_centers = get_initial_points(r_init)
        
        # Run SA
        time_left = deadline - time.time() - 5.0
        if time_left < 1.0:
            break
            
        # Split time among a few restarts for this target?
        # Or just one long run?
        # Let's do 2 restarts per target.
        num_restarts = 2
        time_per_restart = time_left / num_restarts
        
        for _ in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Re-generate initial points with some randomness?
            # The hex lattice is deterministic. Let's add small noise.
            noisy_init = init_centers + np.random.normal(0, 0.01, init_centers.shape)
            
            centers, energy = anneal(noisy_init, r_target, time_per_restart)
            
            # Check the actual radius of the best found
            actual_r = compute_radius(centers)
            
            if actual_r > best_score:
                best_score = actual_r
                best_centers = centers
                
            # If we achieved the target, we can stop trying this target
            # and move to the next (higher) one? 
            # Wait, r_targets is descending. So if we succeed at 0.1258, we are done?
            # No, we want the *best* possible. 0.1258 is the highest we try.
            # If we succeed at 0.1258, we should stop and maybe polish.
            if actual_r >= r_target - 1e-6:
                break
        
        # If we succeeded at the highest target, break out of target loop
        if best_score >= r_targets[0] - 1e-6:
            break

    # Final polish: if we have time, run a local optimization (L-BFGS-B or similar)
    # or a very fine-grained SA on the best result found.
    # Given the complexity, a simple gradient-free local search with very small steps
    # can often squeeze out a bit more.
    
    if best_centers is not None and time.time() < deadline - 2.0:
        # Local polish
        current_centers = best_centers.copy()
        current_r = best_score
        
        # Try to increase r slightly?
        # Or just minimize the "gap" to a slightly higher r?
        # Let's try to optimize for r_target = current_r + 0.0001
        r_polish_target = current_r + 0.0005
        
        # Use a very short SA with small steps
        T = 0.005
        step_size = 0.005
        end_time = deadline - 1.0
        
        while time.time() < end_time:
            idx = np.random.randint(N)
            old_x, old_y = current_centers[idx]
            
            new_x = old_x + np.random.normal(0, step_size)
            new_y = old_y + np.random.normal(0, step_size)
            
            new_x = np.clip(new_x, -1.0, 1.0)
            new_y = np.clip(new_y, -1.0, 1.0)
            
            current_centers[idx] = [new_x, new_y]
            
            # We want to maximize the minimum radius.
            # This is equivalent to minimizing energy for a target r = current_r + epsilon.
            # If energy is 0, we increased r. If not, we might have decreased it.
            # Let's just compute the actual radius and accept if it's better.
            # This is a "greedy" local search.
            
            actual_r = compute_radius(current_centers)
            
            if actual_r > current_r:
                current_r = actual_r
                best_score = current_r
                best_centers = current_centers.copy()
            else:
                # Revert
                current_centers[idx] = [old_x, old_y]
            
            # Decay T? Not really used in greedy. Just keep step size small.
            # Maybe reduce step size over time.
            step_size *= 0.999

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
