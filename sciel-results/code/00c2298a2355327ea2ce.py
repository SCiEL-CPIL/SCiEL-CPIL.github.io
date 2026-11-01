"""Program for packing 200 equal circles in a unit square using Force-Directed Relaxation."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Initialize with a jittered grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, size=centers.shape)
    centers = np.clip(centers, -0.499, 0.499)

    current_centers = centers.copy()
    
    # Parameters
    max_iter = 3000
    learning_rate = 0.01
    decay = 0.999
    wall_strength = 5.0
    wall_zone = 0.05
    
    for i in range(max_iter):
        forces = np.zeros_like(current_centers)
        
        # Pairwise Repulsion
        diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)
        
        dists = np.sqrt(dists_sq)
        # Softening to avoid division by zero issues, though diagonal is 1
        inv_dists_cubed = 1.0 / (dists_sq * dists + 1e-8)
        
        # Force on i from j
        # F = sum_j (c_i - c_j) / |c_i - c_j|^3
        forces += np.sum(diff * inv_dists_cubed[:, :, np.newaxis], axis=1)
        
        # Wall Repulsion
        # Push inward if within wall_zone
        # X-axis
        left_margin = current_centers[:, 0] + 0.5
        right_margin = 0.5 - current_centers[:, 0]
        
        forces[:, 0] += np.maximum(0, wall_zone - left_margin) * wall_strength
        forces[:, 0] -= np.maximum(0, wall_zone - right_margin) * wall_strength
        
        # Y-axis
        bottom_margin = current_centers[:, 1] + 0.5
        top_margin = 0.5 - current_centers[:, 1]
        
        forces[:, 1] += np.maximum(0, wall_zone - bottom_margin) * wall_strength
        forces[:, 1] -= np.maximum(0, wall_zone - top_margin) * wall_strength
        
        # Limit force magnitude to prevent instability
        force_mags = np.linalg.norm(forces, axis=1, keepdims=True)
        max_force = 0.5
        # Normalize if too large
        scale = np.where(force_mags > max_force, max_force / force_mags, 1.0)
        forces *= scale
        
        # Update
        current_centers += learning_rate * forces
        current_centers = np.clip(current_centers, -0.5, 0.5)
        
        # Decay
        learning_rate *= decay
        
        # Periodic noise to escape local minima
        if i % 500 == 0 and i > 0:
            noise_scale = 0.005 * (1.0 - i / max_iter)
            current_centers += rng.normal(0, noise_scale, size=current_centers.shape)
            current_centers = np.clip(current_centers, -0.5, 0.5)

    return current_centers
# EVOLVE-BLOCK-END
