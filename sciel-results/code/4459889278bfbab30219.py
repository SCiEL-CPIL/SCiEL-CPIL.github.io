import numpy as np
import time

N = 50
DEADLINE = 115.0  # Safety margin from 120s

def construct():
    start_time = time.time()
    
    # Helper to generate initial hexagonal lattice points
    def get_hex_initial():
        spacing = 0.25
        points = []
        k = 10
        for i in range(-k, k+1):
            for j in range(-k, k+1):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if x**2 + y**2 <= 0.9:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            return np.random.uniform(-0.8, 0.8, (N, 2))
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:N]
        return points[indices]

    # Compute the score (radius) for a given set of centers
    def compute_radius(centers):
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary distances
        norms = np.linalg.norm(centers, axis=1)
        min_bound_dist = np.min(1.0 - norms)
        
        return min(min_pair_dist, min_bound_dist) / 2.0

    # Simulated Annealing
    def simulated_annealing(initial_centers, max_time=100.0):
        centers = initial_centers.copy()
        current_radius = compute_radius(centers)
        best_centers = centers.copy()
        best_radius = current_radius
        
        # Initial temperature: allow moves that reduce radius by up to ~10%
        T = 0.05 * current_radius
        T_min = 1e-6
        cooling_rate = 0.9995
        
        step_size = 0.1
        step_decay = 0.999
        min_step = 0.001
        
        rng = np.random.default_rng()
        
        while time.time() - start_time < max_time:
            # Pick a random circle to move
            idx = rng.integers(N)
            # Random direction and magnitude
            angle = rng.uniform(0, 2 * np.pi)
            magnitude = rng.uniform(0, step_size)
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
            
            new_centers = centers.copy()
            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
            
            # Check if still inside container (with small tolerance)
            if np.linalg.norm(new_centers[idx]) > 1.0:
                continue
            
            new_radius = compute_radius(new_centers)
            delta = new_radius - current_radius
            
            if delta > 0 or rng.random() < np.exp(delta / T):
                centers = new_centers
                current_radius = new_radius
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = centers.copy()
            
            # Cooling
            T *= cooling_rate
            step_size *= step_decay
            if step_size < min_step:
                step_size = min_step
                # If we're stuck, restart with a higher temperature occasionally
                if rng.random() < 0.01:
                    T = 0.02 * best_radius
                    step_size = 0.05
        
        return best_centers, best_radius

    # Generate multiple diverse initial configurations
    def generate_initials(num_initials=5):
        initials = []
        # Hex lattice
        initials.append(get_hex_initial())
        
        # Random uniform
        for _ in range(num_initials - 1):
            pts = np.random.uniform(-0.9, 0.9, (N, 2))
            # Scale to fit inside
            norms = np.linalg.norm(pts, axis=1)
            max_norm = np.max(norms)
            if max_norm > 0.9:
                pts *= 0.9 / max_norm
            initials.append(pts)
        
        return initials

    # Run SA on multiple initial configurations
    best_overall = None
    best_r_overall = 0.0
    
    initials = generate_initials(5)
    
    # Allocate time per initial configuration
    time_per_init = (DEADLINE - (time.time() - start_time)) / len(initials)
    
    for i, init_c in enumerate(initials):
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5:
            break
        
        # Add small noise to break symmetry
        noisy_init = init_c + np.random.normal(0, 0.01, init_c.shape)
        norms = np.linalg.norm(noisy_init, axis=1)
        scale = np.where(norms > 0.9, 0.9 / norms, 1.0)
        noisy_init *= scale[:, np.newaxis]
        
        result_centers, result_r = simulated_annealing(noisy_init, max_time=time_per_init - 2)
        
        if result_r > best_r_overall:
            best_r_overall = result_r
            best_overall = result_centers
        
        # If we have significant time left and haven't found a good solution, try more
        if i < len(initials) - 1 and time.time() - start_time < DEADLINE - 10:
            # Generate a new random initial
            new_init = np.random.uniform(-0.8, 0.8, (N, 2))
            norms = np.linalg.norm(new_init, axis=1)
            scale = np.where(norms > 0.8, 0.8 / norms, 1.0)
            new_init *= scale[:, np.newaxis]
            initials.append(new_init)
    
    # Final fallback
    if best_overall is None:
        return get_hex_initial()
    
    # Verify and return
    return best_overall

if __name__ == "__main__":
    c = construct()
    if len(c) == N:
        min_dist = 1e10
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(c[i] - c[j])
                if d < min_dist:
                    min_dist = d
        min_bound = 1e10
        for i in range(N):
            d = 1 - np.linalg.norm(c[i])
            if d < min_bound:
                min_bound = d
        r = min(min_dist, min_bound) / 2
        print(f"Radius: {r}")
