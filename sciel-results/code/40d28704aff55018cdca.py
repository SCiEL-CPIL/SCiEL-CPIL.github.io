"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """Place 50 equal spheres in unit cube [-0.5, 0.5]^3 to maximize minimum radius using simulated annealing."""
    
    def compute_min_radius(positions):
        """Compute the minimum radius given positions of shape (N, 3)."""
        positions = np.asarray(positions).reshape(-1, 3)
        # Pairwise distances
        from scipy.spatial.distance import pdist
        dists = pdist(positions, 'euclidean')
        min_dist = np.min(dists) if len(dists) > 0 else 1.0
        min_r_pairs = min_dist / 2.0
        
        # Wall distances: distance from each center to nearest wall
        walls = 0.5 - np.abs(positions)
        min_wall = np.min(walls)
        
        return min(min_r_pairs, min_wall)
    
    def compute_forces(positions, target_r):
        """Compute repulsive forces on each sphere to push them apart and away from walls."""
        forces = np.zeros_like(positions)
        n = len(positions)
        
        # Pairwise repulsion: if two spheres are closer than 2*target_r, push apart
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random small displacement
                    diff = np.random.randn(3) * 1e-5
                    dist = np.linalg.norm(diff)
                if dist < 2.0 * target_r:
                    # Strong repulsion when overlapping
                    overlap = 2.0 * target_r - dist
                    force_mag = 100.0 * overlap / dist
                    force = force_mag * diff
                    forces[i] += force
                    forces[j] -= force
        
        # Wall repulsion: if sphere is too close to wall, push inward
        for i in range(n):
            for k in range(3):
                # Left wall at -0.5
                if positions[i, k] < -0.5 + target_r:
                    overlap = (-0.5 + target_r) - positions[i, k]
                    forces[i, k] += 100.0 * overlap
                # Right wall at +0.5
                if positions[i, k] > 0.5 - target_r:
                    overlap = positions[i, k] - (0.5 - target_r)
                    forces[i, k] -= 100.0 * overlap
        
        return forces
    
    def simulated_annealing(positions, target_r, n_steps=5000, T_start=0.01, T_end=1e-6, cooling=0.9995):
        """Run simulated annealing to improve packing for a given target radius."""
        pos = positions.copy()
        T = T_start
        best_pos = pos.copy()
        best_r = compute_min_radius(pos)
        
        for step in range(n_steps):
            # Generate random perturbation
            delta = np.random.randn(*pos.shape) * T * 2.0
            new_pos = pos + delta
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
            # Compute new radius
            new_r = compute_min_radius(new_pos)
            
            # Accept if better or with probability exp((new_r - best_r)/T)
            if new_r >= best_r or np.random.random() < np.exp((new_r - best_r) / max(T, 1e-10)):
                pos = new_pos
                if new_r > best_r:
                    best_r = new_r
                    best_pos = pos.copy()
            
            # Cool down
            T = max(T * cooling, T_end)
        
        return best_pos, best_r
    
    def make_fcc_initial():
        """Create an FCC-inspired initial configuration for 50 spheres."""
        # FCC unit cell has 4 atoms. We'll create a 3x3x3 supercell and select 50.
        # FCC positions in a unit cell (fractional coordinates):
        # (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
        
        positions = []
        # Generate FCC lattice points in a 3x3x3 grid
        for ix in range(3):
            for iy in range(3):
                for iz in range(3):
                    # 4 FCC positions per cell
                    for fx, fy, fz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                        x = (ix + fx) / 3.0 - 0.5
                        y = (iy + fy) / 3.0 - 0.5
                        z = (iz + fz) / 3.0 - 0.5
                        if -0.49 <= x <= 0.49 and -0.49 <= y <= 0.49 and -0.49 <= z <= 0.49:
                            positions.append([x, y, z])
        
        # We have too many, randomly select 50
        np.random.seed(42)
        if len(positions) > N:
            indices = np.random.choice(len(positions), N, replace=False)
            positions = [positions[i] for i in indices]
        else:
            # Pad with random positions if needed
            while len(positions) < N:
                positions.append(np.random.uniform(-0.45, 0.45, 3).tolist())
        
        return np.array(positions[:N])
    
    def make_grid_initial():
        """Create a grid initial configuration."""
        m = 4  # 4x4x4 = 64 points, take 50
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        np.random.seed(42)
        indices = np.random.choice(len(grid), N, replace=False)
        return grid[indices]
    
    best_positions = None
    best_radius = -1.0
    
    np.random.seed(42)
    
    # Try multiple initial configurations
    initializations = [
        ("fcc", make_fcc_initial),
        ("grid", make_grid_initial),
    ]
    
    for name, init_func in initializations:
        pos0 = init_func()
        r0 = compute_min_radius(pos0)
        print(f"{name} initial: r = {r0:.10f}")
        
        # Use simulated annealing with gradually increasing target radius
        # Start from a small target and increase
        target_r = r0 * 0.95
        
        # Phase 1: Anneal at current radius to improve
        pos1, r1 = simulated_annealing(pos0, target_r, n_steps=3000, T_start=0.005, T_end=1e-7)
        print(f"{name} anneal1: r = {r1:.10f}")
        
        if r1 > best_radius:
            best_radius = r1
            best_positions = pos1
        
        # Phase 2: Try to increase target radius
        for increase_step in range(5):
            target_r = r1 * (1.0 + 0.02 * (increase_step + 1))
            pos2, r2 = simulated_annealing(pos1, target_r, n_steps=2000, T_start=0.003, T_end=1e-8)
            print(f"{name} anneal2_{increase_step}: target={target_r:.6f}, r = {r2:.10f}")
            
            if r2 > best_radius:
                best_radius = r2
                best_positions = pos2
                pos1 = pos2
                r1 = r2
        
        # Phase 3: Fine-tune with very small steps
        pos3, r3 = simulated_annealing(pos1, compute_min_radius(pos1), n_steps=5000, T_start=0.001, T_end=1e-9)
        print(f"{name} anneal3: r = {r3:.10f}")
        
        if r3 > best_radius:
            best_radius = r3
            best_positions = pos3
    
    # Final fine-tuning
    if best_positions is not None:
        final_r = compute_min_radius(best_positions)
        pos_final, r_final = simulated_annealing(best_positions, final_r, n_steps=3000, T_start=0.0005, T_end=1e-10)
        if r_final > best_radius:
            best_radius = r_final
            best_positions = pos_final
    
    # Ensure all positions are strictly inside the cube
    best_positions = np.clip(best_positions, -0.4999999, 0.4999999)
    
    print(f"Final best radius: {best_radius:.10f}")
    
    return best_positions
# EVOLVE-BLOCK-END
