"""
Packing 50 equal circles in a unit circle.
Uses a force-directed / repulsion-based approach with a deadline guard.
"""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 120s
    
    # Best known radius for 50 circles: ~0.1258254895304
    # We'll try to approach or exceed this.
    
    def score(positions):
        """Compute the packing radius."""
        # Min distance between any pair
        n = len(positions)
        min_pair_dist = float('inf')
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_pair_dist:
                    min_pair_dist = d
        # Min distance to boundary
        min_boundary = float('inf')
        for i in range(n):
            r = np.linalg.norm(positions[i])
            if r < min_boundary:
                min_boundary = r
        return min(min_pair_dist / 2.0, 1.0 - min_boundary)
    
    def validate(positions):
        """Check if positions are valid (finite, within bounds)."""
        if not np.all(np.isfinite(positions)):
            return False
        if positions.shape != (N, 2):
            return False
        # Check all points are inside or on the unit circle
        if np.any(np.linalg.norm(positions, axis=1) > 1.0 + 1e-10):
            return False
        return True
    
    # Initial placement: use a hexagonal-like grid as starting point
    def initial_placement():
        # Place points in a hexagonal pattern within the unit circle
        positions = np.zeros((N, 2))
        # Use concentric rings
        ring_radii = [0.0]
        ring_counts = [1]
        r = 0.15
        while sum(ring_counts) < N:
            # Number of points on a ring of radius r
            count = max(1, int(2 * np.pi * r / (2 * r)))  # rough estimate
            count = max(6, int(2 * np.pi * r / (2 * 0.12)))
            ring_radii.append(r)
            ring_counts.append(count)
            r += 0.12
        
        # Adjust to get exactly N points
        idx = 0
        positions[idx] = [0.0, 0.0]
        idx += 1
        for ring_i, (radius, count) in enumerate(zip(ring_radii[1:], ring_counts[1:])):
            for k in range(count):
                if idx >= N:
                    break
                angle = 2 * np.pi * k / count
                positions[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
                idx += 1
            if idx >= N:
                break
        
        # Ensure we have exactly N points
        if idx < N:
            # Fill remaining with random positions
            for k in range(idx, N):
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, 0.9)
                positions[k] = [radius * np.cos(angle), radius * np.sin(angle)]
        
        return positions[:N]
    
    # Force-directed relaxation
    def relax(positions, max_iters=5000):
        positions = positions.copy()
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            forces = np.zeros_like(positions)
            
            # Repulsion between all pairs
            for i in range(N):
                for j in range(i+1, N):
                    diff = positions[i] - positions[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-10:
                        # Add small random perturbation
                        diff = np.random.randn(2) * 1e-5
                        dist = np.linalg.norm(diff)
                    # Force is proportional to 1/dist^2, pushing apart
                    force_mag = 1.0 / (dist * dist)
                    force = force_mag * diff / dist
                    forces[i] += force
                    forces[j] -= force
            
            # Attraction toward the center (to keep points inside)
            for i in range(N):
                r = np.linalg.norm(positions[i])
                if r > 0.9:
                    # Push back toward center
                    forces[i] -= 0.5 * positions[i] / r
            
            # Update positions
            positions += lr * forces
            
            # Project back into the unit circle
            for i in range(N):
                r = np.linalg.norm(positions[i])
                if r > 1.0:
                    positions[i] = positions[i] / r
            
            # Decay learning rate
            lr *= 0.999
        
        return positions
    
    # Simulated annealing
    def simulated_annealing(positions, max_iters=20000):
        positions = positions.copy()
        best_pos = positions.copy()
        best_score = score(positions)
        
        T = 0.05  # initial temperature
        T_min = 0.0001
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Propose a move: random perturbation
            new_positions = positions.copy()
            idx = np.random.randint(N)
            perturbation = np.random.randn(2) * T
            new_positions[idx] += perturbation
            
            # Keep inside the circle
            r = np.linalg.norm(new_positions[idx])
            if r > 1.0:
                new_positions[idx] = new_positions[idx] / r
            
            new_score = score(new_positions)
            
            if new_score >= best_score:
                best_score = new_score
                best_pos = new_positions.copy()
            
            # Accept or reject
            if new_score > score(positions):
                positions = new_positions
            else:
                delta = new_score - score(positions)
                if np.random.rand() < np.exp(delta / T):
                    positions = new_positions
            
            T = max(T * cooling, T_min)
        
        return best_pos, best_score
    
    # Main algorithm
    best_positions = initial_placement()
    best_score = score(best_positions)
    
    # Multiple restarts with different initializations
    n_restarts = 0
    while time.time() < deadline:
        n_restarts += 1
        
        # Generate initial position
        if n_restarts == 1:
            positions = best_positions.copy()
        else:
            positions = initial_placement()
            # Add some randomness
            positions += np.random.randn(N, 2) * 0.05
            # Project into circle
            for i in range(N):
                r = np.linalg.norm(positions[i])
                if r > 1.0:
                    positions[i] = positions[i] / r
        
        # Relax
        positions = relax(positions, max_iters=3000)
        s = score(positions)
        if s > best_score:
            best_score = s
            best_positions = positions.copy()
        
        # Simulated annealing
        if time.time() < deadline - 10:
            positions, s = simulated_annealing(positions, max_iters=10000)
            if s > best_score:
                best_score = s
                best_positions = positions.copy()
    
    # Final validation
    if not validate(best_positions):
        # Fallback: return a simple valid configuration
        best_positions = initial_placement()
        for i in range(N):
            r = np.linalg.norm(best_positions[i])
            if r > 1.0:
                best_positions[i] = best_positions[i] / r
    
    return best_positions
# EVOLVE-BLOCK-END
