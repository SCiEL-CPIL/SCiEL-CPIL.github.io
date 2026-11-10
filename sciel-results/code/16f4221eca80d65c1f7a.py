import numpy as np
import time
import random

N = 50

def construct():
    start_time = time.time()
    max_time = 115.0
    
    def calculate_score(centers):
        # Calculate min distance between pairs and to boundary
        # Vectorized
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        min_boundary = np.min(1.0 - np.linalg.norm(centers, axis=1))
        return min(min_pair, min_boundary)

    def run_optimized_sa(initial_centers, time_budget):
        centers = initial_centers.copy()
        current_score = calculate_score(centers)
        best_score = current_score
        best_centers = centers.copy()
        
        T = 0.005 # Start low for directed moves? Or high?
        # Let's start moderate
        T = 0.01
        cooling_rate = 0.999
        step_size = 0.02
        
        no_improve_count = 0
        end_time = start_time + time_budget
        
        while time.time() < end_time:
            if time.time() > start_time + max_time:
                break
                
            # Calculate forces/directions based on current configuration
            # Find nearest neighbor for each point to determine repulsion direction
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Nearest neighbor index
            nn_idx = np.argmin(dists, axis=1)
            
            # Direction to move: away from nearest neighbor, and towards center if close to boundary
            moves = np.zeros((N, 2))
            for i in range(N):
                j = nn_idx[i]
                # Vector from i to j
                vec_to_nn = centers[j] - centers[i]
                # Move away from j
                norm_vec = np.linalg.norm(vec_to_nn)
                if norm_vec > 1e-9:
                    moves[i] -= vec_to_nn / norm_vec
                
                # Boundary repulsion
                dist_to_center = np.linalg.norm(centers[i])
                if dist_to_center > 0.9:
                    # Push towards center
                    if dist_to_center > 1e-9:
                        moves[i] -= centers[i] / dist_to_center * (dist_to_center - 0.9)
            
            # Normalize moves to have consistent magnitude
            move_norms = np.linalg.norm(moves, axis=1)
            for i in range(N):
                if move_norms[i] > 1e-9:
                    moves[i] /= move_norms[i]
            
            # Select a random point to move
            idx = np.random.randint(N)
            
            # Combine directed move with random noise
            noise = np.random.normal(0, step_size * 0.5, 2)
            new_point = centers[idx] + moves[idx] * step_size + noise
            
            # Keep inside circle
            norm = np.linalg.norm(new_point)
            if norm > 0.99:
                new_point = new_point / norm * 0.99
                
            old_point = centers[idx]
            centers[idx] = new_point
            
            new_score = calculate_score(centers)
            
            delta_E = new_score - current_score
            
            if delta_E > 0:
                accept = True
            else:
                if T > 1e-8:
                    p = np.exp(delta_E / T)
                    accept = (np.random.rand() < p)
                else:
                    accept = False
            
            if accept:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_centers = centers.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                centers[idx] = old_point
                no_improve_count += 1
            
            # Cooling
            T *= cooling_rate
            
            # Reheating logic
            if no_improve_count > 1000:
                T = 0.05
                step_size = 0.05
                no_improve_count = 0
            elif no_improve_count > 500:
                T = max(T, 0.02)
                step_size = max(step_size, 0.03)
            
            # Adapt step size
            if no_improve_count < 100:
                step_size = max(0.001, step_size * 0.99)
            else:
                step_size = min(0.1, step_size * 1.01)
                
            if T < 1e-8:
                T = 0.001 # Floor

        return best_centers, best_score

    # Initialization: Hexagonal Grid
    # Generate a dense hex grid and pick the N closest to origin
    spacing = 0.2
    k = int(1.0 / spacing) + 3
    points = []
    for j in range(-k, k+1):
        for i in range(-k, k+1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 0.95:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) >= N:
        # Sort by distance to origin
        dists = np.linalg.norm(points, axis=1)
        sorted_indices = np.argsort(dists)
        init_centers = points[sorted_indices[:N]]
    else:
        init_centers = np.random.uniform(-0.8, 0.8, (N, 2))
        
    # Run the optimized SA
    best_centers, best_score = run_optimized_sa(init_centers, max_time - 5.0)
    
    # Final check
    if best_centers is None or len(best_centers) != N:
        best_centers = np.random.uniform(-0.9, 0.9, (N, 2))
        
    # Ensure validity
    norms = np.linalg.norm(best_centers, axis=1)
    for i in range(N):
        if norms[i] > 1.0:
            best_centers[i] = best_centers[i] / norms[i] * 0.99
            
    return best_centers
