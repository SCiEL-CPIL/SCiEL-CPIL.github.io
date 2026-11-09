"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    np.random.seed(42)
    
    def calc_radius(p):
        """Calculate the maximum radius for a given configuration."""
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        wall_dists = 0.5 - np.max(np.abs(p), axis=1)
        min_wall_dist = np.min(wall_dists)
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    def repulsive_energy(p, r_target):
        """
        Calculate repulsive energy for particle system.
        Energy is high when particles are too close or too close to walls.
        """
        # Pairwise repulsion: use inverse square potential
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Repulsive force: 1/d^2 for d < 2*r_target
        pair_energy = np.zeros_like(dists)
        mask = dists < 2 * r_target
        pair_energy[mask] = 1.0 / (dists[mask] ** 2)
        total_pair_energy = np.sum(pair_energy)
        
        # Wall repulsion
        wall_dists = 0.5 - np.max(np.abs(p), axis=1)
        wall_mask = wall_dists < r_target
        wall_energy = np.zeros_like(wall_dists)
        wall_energy[wall_mask] = 1.0 / (wall_dists[wall_mask] ** 2)
        total_wall_energy = np.sum(wall_energy)
        
        return total_pair_energy + total_wall_energy
    
    def metropolis_monte_carlo(pos, r_target, time_limit, T_initial=0.01, T_final=0.0001):
        """
        Metropolis-Hastings Monte Carlo to equilibrate particle system.
        """
        current_energy = repulsive_energy(pos, r_target)
        best_energy = current_energy
        best_pos = pos.copy()
        current_pos = pos.copy()
        
        # Adaptive temperature schedule
        steps = 50000
        T = T_initial
        T_end = T_final
        alpha = (T_end / T) ** (1.0 / steps)
        
        for step in range(steps):
            if time.time() > time_limit:
                break
            
            # Random move: pick a particle and move it
            i = np.random.randint(N)
            move_size = 0.03 * (T / T_initial + 0.1)
            move = np.random.uniform(-move_size, move_size, 2)
            
            new_pos = current_pos.copy()
            new_pos[i] += move
            
            # Check bounds
            if np.any(new_pos[i] > 0.5) or np.any(new_pos[i] < -0.5):
                continue
            
            new_energy = repulsive_energy(new_pos, r_target)
            delta_e = new_energy - current_energy
            
            # Metropolis criterion
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                current_pos = new_pos
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = current_pos.copy()
            
            T *= alpha
        
        return best_pos
    
    def gradient_descent_polish(pos, r_target, time_limit):
        """
        Gradient descent to polish the configuration.
        Uses repulsive forces to push particles apart.
        """
        current_pos = pos.copy()
        lr = 0.001
        
        for step in range(20000):
            if time.time() > time_limit:
                break
            
            # Compute forces
            p = current_pos
            forces = np.zeros_like(p)
            
            # Pairwise forces
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Force: -1/d^3 * diff (repulsive)
            mask = dists < 2 * r_target
            for i in range(N):
                for j in range(N):
                    if i != j and mask[i, j]:
                        d = dists[i, j]
                        f = -1.0 / (d ** 3)
                        forces[i] += f * diff[i, j]
            
            # Wall forces
            wall_dists = 0.5 - np.max(np.abs(p), axis=1)
            wall_mask = wall_dists < r_target
            for i in range(N):
                if wall_mask[i]:
                    wd = wall_dists[i]
                    # Push away from wall
                    for dim in range(2):
                        if p[i, dim] > 0:
                            forces[i, dim] -= 1.0 / (wd ** 2)
                        else:
                            forces[i, dim] += 1.0 / (wd ** 2)
            
            # Update positions
            current_pos += lr * forces
            current_pos = np.clip(current_pos, -0.5, 0.5)
        
        return current_pos
    
    # Generate diverse initial configurations
    configs = []
    
    # Grid
    m = 8
    n = 7
    ticks_x = (np.arange(m) + 0.5) / m - 0.5
    ticks_y = (np.arange(n) + 0.5) / n - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    pos_grid = grid[:N].copy()
    pos_grid += np.random.uniform(-0.01, 0.01, pos_grid.shape)
    pos_grid = np.clip(pos_grid, -0.5, 0.5)
    configs.append(pos_grid)
    
    # Random
    pos_rand = np.random.uniform(-0.4, 0.4, (N, 2))
    configs.append(pos_rand)
    
    # Hexagonal
    pos_hex = np.zeros((N, 2))
    idx = 0
    for row in range(10):
        for col in range(10):
            if idx >= N: break
            x = col * 0.11 - 0.5
            y = row * 0.095 - 0.5
            if row % 2 == 1:
                x += 0.055
            pos_hex[idx] = [x, y]
            idx += 1
    pos_hex += np.random.uniform(-0.005, 0.005, pos_hex.shape)
    pos_hex = np.clip(pos_hex, -0.5, 0.5)
    configs.append(pos_hex)
    
    # Best overall
    best_overall_r = -np.inf
    best_overall_pos = None
    
    # For each config, try to maximize radius using binary search + MC
    for i, config in enumerate(configs):
        if time.time() > deadline - 15:
            break
        
        # Start with current radius
        current_r = calc_radius(config)
        
        # Try to increase radius using MC
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
            
            # Target radius: slightly higher than current
            target_r = current_r * 1.001
            
            # MC phase
            mc_time_limit = time.time() + 25.0
            pos_mc = metropolis_monte_carlo(config, target_r, mc_time_limit)
            
            # Gradient polish
            gd_time_limit = time.time() + 10.0
            pos_final = gradient_descent_polish(pos_mc, target_r, gd_time_limit)
            
            new_r = calc_radius(pos_final)
            
            if new_r > current_r:
                current_r = new_r
                config = pos_final.copy()
            else:
                break
        
        # Update best
        final_r = calc_radius(config)
        if final_r > best_overall_r:
            best_overall_r = final_r
            best_overall_pos = config.copy()
    
    if best_overall_pos is None:
        return configs[0]
    
    return best_overall_pos
# EVOLVE-BLOCK-END
