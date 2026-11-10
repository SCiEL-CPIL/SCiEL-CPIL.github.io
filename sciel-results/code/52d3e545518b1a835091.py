import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initialization: Hexagonal lattice
    def init_hex():
        points = []
        # Estimate spacing for N points in unit circle
        # Area = pi. N * pi * r^2 = pi * density. 
        # Let's just generate a grid and filter.
        spacing = 0.2 # Initial guess
        while True:
            candidates = []
            k = 10
            for i in range(-k, k+1):
                for j in range(-k, k+1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3) / 2
                    if x*x + y*y <= 1.0:
                        candidates.append([x, y])
            candidates = np.array(candidates)
            if len(candidates) >= N:
                # Select N points closest to center to start compactly
                dists = np.linalg.norm(candidates, axis=1)
                indices = np.argsort(dists)[:N]
                return candidates[indices]
            spacing *= 0.95
            
    centers = init_hex()
    
    # 2. Helper to calculate score
    def calc_score(centers):
        # Check boundary
        dists_center = np.linalg.norm(centers, axis=1)
        min_boundary = 1.0 - np.max(dists_center)
        
        # Check pairwise distances
        # Use broadcasting for all pairs
        # d_ij = || c_i - c_j ||
        # We only need min
        # For N=50, 1225 pairs.
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_boundary, min_pair)
    
    best_score = calc_score(centers)
    best_centers = centers.copy()
    
    # 3. Optimization Loop
    # Parameters
    step_size = 0.01
    reheat_threshold = 0.0001
    stuck_count = 0
    max_stuck = 100
    temperature = 1.0
    
    while time.time() < deadline:
        current_score = calc_score(centers)
        
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
            stuck_count = 0
            # Adapt step size
            step_size = max(0.001, step_size * 0.9)
        else:
            stuck_count += 1
            # If stuck, reheat
            if stuck_count > max_stuck:
                # Add noise to escape local minimum
                noise = np.random.normal(0, 0.05, centers.shape)
                centers = centers + noise
                # Project back inside if violated? Or just let the forces handle it.
                # Better to keep them inside during noise?
                # Let's just reset step size and increase noise
                step_size = 0.05
                stuck_count = 0
                max_stuck = 50 # Try harder next time
                continue
            
            # Decay step size slowly if not improving but not fully stuck
            step_size = max(0.001, step_size * 0.99)

        # Calculate forces
        # Force from boundary: If close to boundary, push in.
        # Force from other circles: If close, push out.
        
        forces = np.zeros_like(centers)
        
        # Boundary forces
        dists_center = np.linalg.norm(centers, axis=1)
        # Push inward if dist > 1 - margin
        margin = 0.1
        for i in range(N):
            if dists_center[i] > 1.0 - margin:
                # Push towards center
                forces[i] -= centers[i] / dists_center[i] * (dists_center[i] - (1.0 - margin))
        
        # Pairwise forces
        # Only consider close pairs to save time?
        # For N=50, all pairs is fast enough.
        for i in range(N):
            for j in range(i+1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                # We want to maximize min distance.
                # If dist is small, push apart strongly.
                # Let's define a "desired" distance based on current best?
                # Or just a repulsion potential.
                # V(r) = 1/r^2 or similar.
                # Let's use a force that is zero if dist > 2*best_score, and grows if closer.
                target_dist = 2.0 * best_score
                if dist < target_dist:
                    # Push apart
                    force_mag = (target_dist - dist) / dist # Normalized direction * deficit
                    # Scale force
                    forces[i] += diff * force_mag
                    forces[j] -= diff * force_mag
        
        # Normalize forces to avoid huge steps
        force_norms = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_norms)
        if max_force > 0:
            forces = forces / max_force * step_size
        else:
            # If no force, jiggle
            forces = np.random.normal(0, step_size, centers.shape)
            
        # Update positions
        centers = centers + forces
        
        # Ensure points stay inside? 
        # The boundary force should handle it, but if step is too big, they might jump out.
        # Clip them back? No, that creates bias.
        # Just let the next iteration's boundary force correct it.
        
    return best_centers
