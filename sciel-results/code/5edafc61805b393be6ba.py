"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    
    # 1. Initialization: Random points in a slightly smaller box to avoid immediate boundary issues
    # Then we will push them out.
    # Or better: start with a grid to give it structure, then MD.
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pos = grid[:N].copy()
    
    # Add small random jitter to break symmetry
    pos += np.random.uniform(-0.01, 0.01, pos.shape)
    
    # 2. Molecular Dynamics Simulation
    # We use a Morse-like or Lennard-Jones-like potential to push spheres apart.
    # However, simple "repulsion" isn't enough to keep them in the box.
    # We need a hard wall constraint or a soft wall potential.
    # Let's use a soft wall potential that becomes very strong near the boundary.
    
    # Parameters
    dt = 0.001
    temp = 0.05
    cooling_rate = 0.999
    
    # Potential: 
    # Pairwise: U_pair = 1/r^12 (strong repulsion) or similar. 
    # Actually, for packing, we want to maximize min distance. 
    # Minimizing sum of 1/r^k is a standard approach (Thurston's algorithm variant).
    # Let's use U = sum_{i<j} 1/|r_i - r_j|^4 + wall_penalty.
    # Wall penalty: sum_i sum_k (1/2 - |c_ik|)^-4 if close to wall? 
    # Better: keep them inside with a force that pushes back if they cross, 
    # but since we want to maximize radius, we want them to touch the walls.
    # So we want to maximize the minimum of (min_pair_dist/2, min_wall_dist).
    
    # Let's define an objective function to minimize:
    # f(pos) = sum_{i<j} (1/d_ij)^2 + sum_{i,k} (1/wall_dist_ik)^2
    # Minimizing this pushes all distances up.
    
    def compute_forces_and_energy(pos):
        n = len(pos)
        forces = np.zeros_like(pos)
        energy = 0.0
        
        # Pairwise forces
        # Vectorized calculation
        # d_ij = pos[i] - pos[j]
        # This is O(N^2). N=100 is small enough for pure Python loops in MD? 
        # 100*100/2 = 5000 pairs. 10k steps * 5000 = 50M ops. Python might be slow.
        # Let's use NumPy broadcasting for speed.
        
        # d: (N, N, 3)
        d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(d**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero
        
        dist = np.sqrt(dist_sq)
        
        # Potential V = 1/dist^4 -> Force = -grad V = 2 * d / dist^6
        # Wait, V = dist^-4. dV/ddist = -4 dist^-5. Force = -dV/ddist * (d/dist) = 4 dist^-5 * d/dist = 4 d / dist^6.
        # Let's check: U = 1/r^4. F = -grad U = - (d/dr (r^-4)) * r_hat = -(-4 r^-5) * r_hat = 4 r^-5 r_hat.
        # r_hat = d / r. So F = 4 d / r^6.
        
        # We want to MINIMIZE energy, so we move AGAINST the force?
        # No, standard MD: F = -grad U. Acceleration = F/m.
        # If U is high when close, grad points towards high U (inwards). F points outwards.
        # So we should add F to velocity.
        
        inv_dist6 = dist_sq ** -3  # dist^-6
        # Force on i from j: F_ij = 4 * d_ij / dist^6
        # But we have d_ij = pos[i] - pos[j].
        # Force on i is sum_j F_ij.
        # F_ij = 4 * (pos[i] - pos[j]) / |pos[i]-pos[j]|^6
        # This is repulsive.
        
        force_pair = 4.0 * d * inv_dist6[:, :, np.newaxis]
        # Sum over j for each i
        forces += np.sum(force_pair, axis=1)
        
        energy += np.sum(inv_dist6) # Rough energy proxy
        
        # Wall forces
        # We want spheres to stay in [-0.5, 0.5].
        # If a sphere is near the wall, we want to push it in.
        # But we also want to maximize the distance to the wall.
        # So we should treat the wall as a repulsive surface too?
        # If we treat the wall as repulsive, spheres will move away from it, reducing the radius?
        # No, the radius is limited by the *minimum* distance to the wall.
        # If we push them away from the wall, the min distance to the wall increases, UNLESS they are pushed against each other first.
        # Actually, we want to maximize min(all pair dists, all wall dists).
        # Minimizing sum(1/pair_dist^4 + 1/wall_dist^4) is a good proxy.
        
        # Wall distance for each coordinate
        # dist_to_wall = 0.5 - |pos|
        # If dist_to_wall < 0, it's outside. We must prevent that.
        # Let's add a huge penalty if outside, or just clamp.
        # For the force, let's define a soft wall at 0.5.
        # V_wall = 1 / (0.5 - |x|)^4 if |x| < 0.5, else huge.
        # This is tricky to vectorize with the absolute value and sign.
        
        # Alternative: Just clamp positions to [-0.5, 0.5] after each step, and don't add wall force?
        # If we clamp, the "energy" from the wall isn't in the gradient, so they won't actively seek to touch the wall.
        # They will just cluster in the center.
        # To make them touch the walls, we need to encourage them to spread out.
        # The pairwise repulsion will spread them out. If the box is small enough, they will hit the walls.
        # If we just clamp, the ones at the edge will be stuck, and the ones inside will push them?
        # No, if we clamp, the force from the inside pushes them against the wall.
        # This effectively creates a "hard wall" potential.
        # Is this sufficient to maximize the radius?
        # The radius is determined by the *tightest* constraint.
        # If they are all pushed to the walls, the radius is determined by the gaps between them and the walls.
        # Since they are pushed against the walls, the wall distance is small for some, but maybe not the minimum.
        # Actually, if they are in equilibrium with hard walls, the force from the other spheres balances the wall normal force.
        # This is a standard way to solve packing problems.
        
        return forces, energy

    # Velocity Verlet or simple Euler?
    # Let's use simple velocity Verlet for stability.
    vel = np.zeros_like(pos)
    
    # Precompute some constants
    # We need to be careful with the time step.
    # Let's run for a fixed number of steps, checking time.
    
    for step in range(5000):
        if time.time() > deadline - 5.0:
            break
            
        forces, energy = compute_forces_and_energy(pos)
        
        # Update velocity
        vel += forces * dt
        vel *= (1.0 - 0.01) # Friction/Damping to help convergence
        
        # Update position
        pos += vel * dt
        
        # Clamp to box
        pos = np.clip(pos, -0.5, 0.5)
        
        # Cooling (if we were doing Simulated Annealing, we'd add noise. 
        # Here we are doing gradient descent with damping, which is effectively annealing if we add noise.
        # Without noise, it might get stuck in local minima.
        # Let's add small random noise to help escape local minima (Simulated Annealing style).
        # Noise magnitude decreases over time.
        noise_scale = 0.01 * (0.99 ** (step / 100.0))
        if noise_scale > 1e-5:
            pos += np.random.normal(0, noise_scale, pos.shape)
            pos = np.clip(pos, -0.5, 0.5)
            
    # 3. Local Refinement (Coordinate Descent)
    # The MD result is a good starting point, but we need to maximize the exact minimum distance.
    # We can do a local search: for each sphere, move it in a small step that increases the minimum distance.
    
    def get_min_dist(pos):
        # Min pair dist
        d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(d**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        min_pair_dist = np.sqrt(np.min(dist_sq))
        
        # Min wall dist
        wall_dists = 0.5 - np.abs(pos)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    current_score = get_min_dist(pos)
    
    # Coordinate Descent / Hill Climbing
    # Try moving each point in random directions and keep if it improves the score.
    # This is expensive to evaluate score each time (O(N^2)).
    # N=100, 100 points. 1000 iterations * 100 points * 100 pairs = 1M ops. Fast enough.
    
    step_size = 0.01
    for _ in range(2000):
        if time.time() > deadline - 2.0:
            break
            
        improved = False
        for i in range(N):
            # Try 8 random directions
            for _dir in range(8):
                direction = np.random.normal(size=3)
                direction /= np.linalg.norm(direction)
                
                new_pos = pos.copy()
                new_pos[i] += direction * step_size
                new_pos[i] = np.clip(new_pos[i], -0.5, 0.5)
                
                new_score = get_min_dist(new_pos)
                if new_score > current_score + 1e-9:
                    pos = new_pos
                    current_score = new_score
                    improved = True
                    break # Move to next sphere
            
        if not improved:
            # Try smaller step size? Or restart with noise?
            # Let's just break if no improvement in a full pass.
            break
            
    return pos
# EVOLVE-BLOCK-END
