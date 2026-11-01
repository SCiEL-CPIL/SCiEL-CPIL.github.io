"""
Strategy: Physics-Based Molecular Dynamics with Simulated Annealing.
Instead of optimizing a smooth approximation of the min-distance (soft-min),
we simulate a system of particles with strong repulsive forces.
We use a potential that goes to infinity as distance approaches 0, 
mimicking hard spheres.
Simulated Annealing allows the system to escape local minima by 
occasionally accepting worse configurations (higher potential energy).
"""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def _score(pos):
        """
        Compute the actual packing score.
        """
        if pos.shape != (N, 3):
            return -1.0
        if not np.all(np.isfinite(pos)):
            return -1.0
        
        # Pairwise distances
        d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(d**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0) 
        min_pair_dist = np.sqrt(np.min(dist_sq))
        
        # Wall distances
        wall_dists = 0.5 - np.abs(pos)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def _potential_energy(pos, k_rep=1000.0, k_wall=1000.0):
        """
        Calculate potential energy and forces.
        We use a repulsive potential V(r) = k * (1/r)^n for r < r_c, and 0 otherwise?
        Actually, for hard spheres, we want to maximize the minimum distance.
        A common trick is to minimize the sum of inverse powers of distances.
        E = sum_{i<j} 1/d_ij^p + sum_i 1/wall_dist_i^p
        
        However, if any distance is 0, this explodes.
        To handle this, we can use a smooth approximation or just rely on the 
        fact that MD with strong repulsion will separate them.
        
        Let's use a simpler approach:
        Force between i and j: F_ij = k_rep * (1/d_ij^2) * (r_i - r_j) / d_ij  (for d_ij < some threshold)
        Force from wall: F_wall = k_wall * (1/wall_dist^2) * sign(pos)
        
        Actually, a better objective for MD in packing is to minimize:
        E = sum_{i<j} max(0, (2r - d_ij))^2 + sum_i max(0, (r - wall_dist_i))^2
        But we don't know r.
        
        Alternative: Minimize sum of inverse distances.
        E = sum_{i<j} 1/d_ij + sum_i 1/wall_dist_i
        Gradient:
        dE/dd_ij = -1/d_ij^2
        Force on i from j: F = -dE/dr_i = -dE/dd_ij * d(d_ij)/d(r_i)
        d(d_ij)/d(r_i) = (r_i - r_j) / d_ij
        So F_ij on i = -(-1/d_ij^2) * (r_i - r_j)/d_ij = (1/d_ij^3) * (r_i - r_j)
        This pushes i away from j.
        
        Wall force:
        E_wall = 1/wall_dist
        dE/dwall_dist = -1/wall_dist^2
        wall_dist = 0.5 - x (for x>0) -> d(wall_dist)/dx = -1
        dE/dx = -1/wall_dist^2 * (-1) = 1/wall_dist^2
        Force F_x = -dE/dx = -1/wall_dist^2
        If x > 0, wall is at 0.5. Force should be negative (towards center).
        If x < 0, wall is at -0.5. wall_dist = 0.5 - |x| = 0.5 + x.
        d(wall_dist)/dx = 1.
        dE/dx = -1/wall_dist^2 * 1 = -1/wall_dist^2.
        Force F_x = -dE/dx = 1/wall_dist^2.
        So Force is always towards the center.
        
        Let's implement this.
        """
        # Calculate distances
        d = pos[:, np.newaxis, :] - pos[np.newaxis, :]
        dist_sq = np.sum(d**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        
        # Avoid division by zero
        dist[dist < 1e-12] = 1e-12
        
        # Pairwise forces
        # F_ij on i is proportional to (r_i - r_j) / d_ij^3
        # We sum over j != i
        # Using broadcasting:
        # inv_d3 = 1 / dist^3
        inv_d3 = 1.0 / (dist_sq * np.sqrt(dist_sq))
        np.fill_diagonal(inv_d3, 0.0)
        
        # Force matrix: F[i, j] = inv_d3[i, j] * (pos[i] - pos[j])
        # This is (N, N, 3)
        # We want sum_j F[i, j]
        # F[i, j] = inv_d3[i, j] * (pos[i] - pos[j])
        # = inv_d3[i, j] * pos[i] - inv_d3[i, j] * pos[j]
        
        # Sum over j:
        # sum_j inv_d3[i, j] * pos[i] - sum_j inv_d3[i, j] * pos[j]
        # = pos[i] * sum_j inv_d3[i, j] - sum_j (inv_d3[i, j] * pos[j])
        
        sum_inv_d3 = np.sum(inv_d3, axis=1) # (N,)
        # sum_j inv_d3[i, j] * pos[j] is matrix multiplication of inv_d3 (N,N) and pos (N,3)
        weighted_pos_sum = inv_d3 @ pos # (N, 3)
        
        pair_forces = pos * sum_inv_d3[:, np.newaxis] - weighted_pos_sum
        
        # Wall forces
        # wall_dist = 0.5 - |pos|
        abs_pos = np.abs(pos)
        wall_dist = 0.5 - abs_pos
        wall_dist[wall_dist < 1e-12] = 1e-12
        
        # Force = -1/wall_dist^2 * sign(pos)
        # sign(pos) is 1 if pos>0, -1 if pos<0
        signs = np.sign(pos)
        # If pos is 0, sign is 0. Force should be 0 or random? 
        # If pos is 0, wall_dist is 0.5. Force is 0.
        # But wait, if pos is close to 0.5, wall_dist is small.
        # Force magnitude = 1/wall_dist^2
        # Direction = towards center = -sign(pos)
        # So Force = -sign(pos) / wall_dist^2
        
        wall_forces = -signs / (wall_dist**2)
        
        total_forces = pair_forces + wall_forces
        
        # Calculate Energy (for annealing acceptance)
        # E = sum 1/d + sum 1/wall_dist
        pair_energy = np.sum(1.0 / dist)
        wall_energy = np.sum(1.0 / wall_dist)
        energy = pair_energy + wall_energy
        
        return energy, total_forces

    def _init_grid():
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def _init_bcc():
        m = 4
        pts = []
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    pts.append([i, j, k])
                    pts.append([i + 0.5, j + 0.5, k + 0.5])
        pts = np.array(pts)
        pts /= m
        pts -= 0.5
        idx = np.random.choice(len(pts), N, replace=False)
        return pts[idx]

    # Main MD Loop
    best_pos = None
    best_score = -1.0
    
    inits = [
        _init_grid(),
        _init_bcc(),
        np.random.uniform(-0.45, 0.45, (N, 3))
    ]
    
    for init_idx, init_pos in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        pos = init_pos.copy()
        
        # Simulated Annealing Parameters
        # Start with high temperature to allow large moves and escape local minima
        # Cool down to fine-tune
        T_start = 0.5
        T_end = 0.01
        n_steps = 5000
        # Time budget per init: ~10s
        # 5000 steps * 100 particles * 3 dims. 
        # Force calculation is O(N^2) = 10000 ops.
        # 5000 * 10000 = 50M ops. Should be fast in numpy.
        
        # We will run for a fixed number of steps or until time limit
        max_steps = 20000
        step_size = 0.01
        cooling_rate = (T_end / T_start) ** (1.0 / max_steps)
        
        current_T = T_start
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
                
            energy, forces = _potential_energy(pos)
            
            # Proposed move: pos_new = pos + forces * step_size + noise
            # Noise scales with T
            noise = np.random.normal(0, np.sqrt(current_T) * 0.01, pos.shape)
            
            # Limit step size to avoid huge jumps
            # We can normalize forces or just use a fixed small step
            # Let's use a velocity-like update
            # v = forces
            # pos_new = pos + v * dt + noise
            # dt is effectively step_size
            pos_new = pos + forces * step_size + noise
            
            # Clip to bounds (hard constraint)
            pos_new = np.clip(pos_new, -0.5, 0.5)
            
            energy_new, _ = _potential_energy(pos_new)
            
            # Metropolis criterion
            delta_E = energy_new - energy
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / current_T):
                pos = pos_new
                
            # Cool down
            current_T *= cooling_rate
            
            # Check score periodically
            if step % 100 == 0:
                score = _score(pos)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    
        # Final polish with gradient descent on the soft-min? 
        # Or just accept the MD result.
        # MD with 1/r potential tends to equalize distances, which is good for max-min.
        
        final_score = _score(pos)
        if final_score > best_score:
            best_score = final_score
            best_pos = pos.copy()

    if best_pos is None:
        return _init_grid()
        
    return best_pos
# EVOLVE-BLOCK-END
